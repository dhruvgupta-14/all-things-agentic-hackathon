"""The embedding seam.

Local development uses a deterministic hashing embedder; deployment sets
`vertex_project` and `gemini-embedding-001` takes over. Both produce unit
vectors of the same dimensionality, so cosine distance behaves the same way
and the retrieval code cannot tell them apart.

The stub is a hashing-trick embedding, not random noise: tokens map to fixed
dimensions, so texts sharing vocabulary genuinely score closer together. That
makes retrieval testable offline. It is emphatically **not** semantic — it
cannot match "car" to "automobile" — so relevance floors tuned against it will
need revisiting once real embeddings are in play.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.db.models import EMBEDDING_DIM

logger = logging.getLogger(__name__)

STUB_MODEL_NAME = "local-hashing-v1"
GEMINI_MODEL_NAME = "gemini-embedding-001"

# Two separate ceilings, both measured against the live API:
#   * 100 texts per request is a hard cap — 250 returns 400 INVALID_ARGUMENT.
#   * a free-tier key returns 429 well below that; 100 was already refused.
# 20 is comfortably under both, so a 100-chunk paper becomes five requests
# rather than one rejected one.
EMBED_BATCH_SIZE = 20


class EmbeddingUnavailable(Exception):
    """The embedder could not produce vectors. Transient by assumption."""

_TOKEN = re.compile(r"[a-z0-9]+")


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


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        # A zero vector makes cosine distance undefined. Anchor empty text to
        # one fixed direction so it is merely irrelevant, never NaN.
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


class HashingEmbedder:
    """Deterministic local embedder. Same text always yields the same vector."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return STUB_MODEL_NAME

    @property
    def default_min_similarity(self) -> float:
        # Lexical overlap scores lower and flatter than semantic similarity.
        return 0.25

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        tokens = _tokenize(text)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self._dim
            # A sign bit keeps unrelated collisions from always reinforcing.
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        return _normalise(vector)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class GeminiEmbedder:
    """`gemini-embedding-001` at reduced output dimensionality.

    Reachable two ways — an AI Studio API key, or Vertex AI with ADC. They are
    the same model, so `model_name` is deliberately identical for both: the
    vectors are interchangeable, and moving from one transport to the other
    must not make every existing paper look stale and trigger a pointless
    re-index.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
        dim: int = EMBEDDING_DIM,
        batch_size: int = EMBED_BATCH_SIZE,
    ) -> None:
        if not api_key and not project:
            raise ValueError("GeminiEmbedder needs either an api_key or a project")
        self._api_key = api_key
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
        return "vertex" if self._project else "ai-studio"

    def _get_client(self):
        if self._client is None:
            from google import genai  # imported lazily: local dev has no creds

            if self._project:
                self._client = genai.Client(
                    vertexai=True, project=self._project, location=self._location
                )
            else:
                self._client = genai.Client(api_key=self._api_key)
        return self._client

    @retry(
        retry=retry_if_exception_type(EmbeddingUnavailable),
        # The free tier caps embeddings at 100 requests per *minute*, and the
        # API's own retry hint is around 45s. Backing off to 20s exhausts the
        # attempts inside the window and fails a job that would have succeeded
        # by simply waiting.
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=64),
        reraise=True,
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
            # Rate limits are the expected failure on a free-tier key, and they
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
        # are what Google's guidance specifies for reduced dimensionality, and
        # they keep the stored vectors comparable to the local stub's.
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
    settings = get_settings()
    # Vertex first: a deployment given a project must not silently fall back to
    # a developer's personal API key.
    if settings.vertex_project:
        return GeminiEmbedder(
            project=settings.vertex_project, location=settings.vertex_location
        )
    if settings.gemini_api_key:
        return GeminiEmbedder(api_key=settings.gemini_api_key)
    logger.debug("using the local hashing embedder; retrieval will be lexical only")
    return HashingEmbedder()
