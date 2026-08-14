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

from app.config import get_settings
from app.db.models import EMBEDDING_DIM

logger = logging.getLogger(__name__)

STUB_MODEL_NAME = "local-hashing-v1"
VERTEX_MODEL_NAME = "gemini-embedding-001"

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    @property
    def model_name(self) -> str:
        """Stored on the paper, so a model change is a migration rather than
        silent corruption of a mixed-vector index."""
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


class VertexEmbedder:
    """`gemini-embedding-001` via Vertex AI, at reduced output dimensionality."""

    def __init__(self, project: str, location: str, dim: int = EMBEDDING_DIM) -> None:
        self._project = project
        self._location = location
        self._dim = dim
        self._client = None

    @property
    def model_name(self) -> str:
        return VERTEX_MODEL_NAME

    def _get_client(self):
        if self._client is None:
            from google import genai  # imported lazily: local dev has no creds

            self._client = genai.Client(
                vertexai=True, project=self._project, location=self._location
            )
        return self._client

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        response = self._get_client().models.embed_content(
            model=VERTEX_MODEL_NAME,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type, output_dimensionality=self._dim
            ),
        )
        return [list(item.values) for item in response.embeddings]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Documents and queries are embedded with different task types; using
        # one for both measurably degrades retrieval.
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def get_embedder() -> Embedder:
    settings = get_settings()
    if settings.vertex_project:
        return VertexEmbedder(settings.vertex_project, settings.vertex_location)
    logger.debug("using the local hashing embedder; retrieval will be lexical only")
    return HashingEmbedder()
