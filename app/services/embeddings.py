"""The embedding seam: `gemini-embedding-001` on Vertex AI.

`Embedder` stays a protocol because the pipeline and retrieval take an embedder
rather than reaching for a global, which is what makes them testable — the
suite supplies a deterministic fake from `tests/fakes.py`.

What is gone is the *fallback*. `get_embedder()` used to return a hashing
embedder when no project was configured, and that was the worst kind of
default: ingestion completed, retrieval returned passages, citations verified,
and none of it was semantic. A paper embedded that way is also permanently
unsearchable against real vectors, because the two live in different spaces.
Configuration is now required, so that state is unreachable.
"""

from __future__ import annotations

import logging
import math
from typing import Protocol

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.db.models import EMBEDDING_DIM

logger = logging.getLogger(__name__)

GEMINI_MODEL_NAME = "gemini-embedding-001"

# 100 texts per request is a hard cap — 250 returns 400 INVALID_ARGUMENT — and
# request-rate quota bites well below it. 20 is comfortably under both, so a
# 100-chunk paper becomes five requests rather than one rejected one.
EMBED_BATCH_SIZE = 20


class EmbeddingUnavailable(Exception):
    """The embedder could not produce vectors. Transient by assumption."""

class Embedder(Protocol):
    @property
    def model_name(self) -> str:
        """Stored on the paper, so a model change is a migration rather than
        silent corruption of a mixed-vector index."""
        ...

    @property
    def default_min_similarity(self) -> float:
        """Relevance floor for this vector space.

        Cosine scores are not comparable between models, so a single global
        threshold is wrong for whichever backend is not active. Carrying it on
        the embedder means switching models cannot silently mis-tune retrieval.
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        # A zero vector makes cosine distance undefined. Anchor empty text to
        # one fixed direction so it is merely irrelevant, never NaN.
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


class GeminiEmbedder:
    """`gemini-embedding-001` at reduced output dimensionality, on Vertex AI."""

    def __init__(
        self,
        *,
        project: str,
        location: str,
        dim: int = EMBEDDING_DIM,
        batch_size: int = EMBED_BATCH_SIZE,
    ) -> None:
        self._project = project
        self._location = location
        self._dim = dim
        self._batch_size = batch_size
        self._client = None

    @property
    def model_name(self) -> str:
        return GEMINI_MODEL_NAME

    @property
    def default_min_similarity(self) -> float:
        # Measured on gemini-embedding-001 at 768 dims, normalised, over one
        # paper (7 chunks) with 4 relevant and 3 deliberately unrelated
        # queries: relevant top-1 scored 0.626..0.781, unrelated top-1 scored
        # 0.515..0.555. 0.58 sits in that gap, biased slightly towards recall.
        # A small sample — re-derive on a real corpus before trusting it.
        return 0.58

    @property
    def transport(self) -> str:
        """Which backend serves the calls. Diagnostics only — not identity."""
        return "vertex"

    def _get_client(self):
        # Shared per process: building one costs a ~12s credential and TLS
        # handshake on its first request (see app/services/genai_client.py).
        if self._client is None:
            from app.services.genai_client import get_genai_client

            self._client = get_genai_client(
                project=self._project,
                location=self._location,
            )
        return self._client

    @retry(
        retry=retry_if_exception_type(EmbeddingUnavailable),
        # Embedding quota is counted per *minute*, and the
        # API's own retry hint is around 45s. Backing off to 20s exhausts the
        # attempts inside the window and fails a job that would have succeeded
        # by simply waiting.
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=64),
        reraise=True,
        # A silent backoff looks exactly like a slow API. Logging every
        # sleep is how you tell 12s of rate limiting apart from 12s of
        # latency, which is a distinction that cost real time once.
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _embed_call(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        try:
            response = self._get_client().models.embed_content(
                model=GEMINI_MODEL_NAME,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=self._dim
                ),
            )
        except Exception as exc:
            # Rate limits are the expected transient failure here, and they
            # are worth waiting out rather than failing an ingestion job.
            raise EmbeddingUnavailable(f"{type(exc).__name__}: {exc}") from exc

        vectors = [list(item.values) for item in response.embeddings]
        if len(vectors) != len(texts):
            raise EmbeddingUnavailable(
                f"asked for {len(texts)} embeddings, received {len(vectors)}"
            )
        # gemini-embedding-001 returns unit vectors only at its native 3072
        # dimensions. Truncating to 768 (Matryoshka) leaves them un-normalised
        # — measured norm ~0.58 — so we re-normalise here. Cosine distance is
        # scale-invariant and would rank correctly either way, but unit vectors
        # are what Google's guidance specifies for reduced dimensionality.
        return [_normalise(vector) for vector in vectors]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Embed in batches the API will accept, preserving input order."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            window = texts[start : start + self._batch_size]
            vectors.extend(self._embed_call(window, task_type))
        return vectors

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Documents and queries are embedded with different task types; using
        # one for both measurably degrades retrieval.
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def get_embedder() -> Embedder:
    """The application's embedder. One backend, no branch.

    Called through the module (`embeddings.get_embedder()`) rather than
    imported by name, so the test harness has a single place to substitute a
    deterministic fake. An imported binding would have to be patched in every
    module that took a copy.
    """
    settings = get_settings()
    return GeminiEmbedder(
        project=settings.vertex_project, location=settings.vertex_location
    )
