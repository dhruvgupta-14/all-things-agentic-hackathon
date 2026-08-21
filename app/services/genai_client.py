"""One Vertex client per process, not one per service instance.

Measured on this project: the first `embed_query` through a freshly built
`genai.Client` costs **12.1 seconds**; the same call on a warm client costs
**480ms**. The difference is credential resolution and the TLS handshake, and
it is paid on the client's first request, not at construction.

Nothing cached the client, and `MemoryService` and `RetrievalService` each
built their own from `get_embedder()`. So every turn paid that handshake twice
before the model was even reached — about 24 seconds of a 55-70 second turn,
spent on nothing.

The cache is keyed on project and location, so pointing the process at a
different project still produces a different client rather than silently
reusing the old one. The factory functions above it are deliberately *not*
cached: they read settings on every call, which is what lets the test harness
substitute a fake without a stale client surviving in a cache.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def get_genai_client(*, project: str, location: str):
    """The shared Vertex client for one project and location.

    Small cache: there are at most a couple of live configurations in a
    process, and an unbounded one would keep credentials alive for backends
    nobody is using any more.
    """
    from google import genai

    logger.info(
        "building Vertex client (cached per process)",
        extra={"project": project, "location": location},
    )
    return genai.Client(vertexai=True, project=project, location=location)


def reset_genai_clients() -> None:
    """Drop the cache. For tests that switch backends mid-process."""
    get_genai_client.cache_clear()
