import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.routers import feedback, health, me, memory, papers, sessions

logger = logging.getLogger(__name__)


async def _warm_model_client() -> None:
    """Pay the Vertex handshake at boot rather than on someone's first question.

    A cold `genai.Client` costs ~12 seconds on its first request — credential
    resolution and TLS, not the call itself. Without this the first turn after
    a deploy or a Cloud Run cold start wears that, which is exactly the turn
    most likely to be the one being watched.

    Failures here are logged and ignored on purpose. A warm-up is an
    optimisation; refusing to start because it did not work would turn a slow
    first turn into an outage.
    """
    settings = get_settings()
    if not (settings.vertex_project or settings.gemini_api_key):
        return  # local stubs build no client

    def _warm() -> None:
        from app.services.embeddings import get_embedder

        get_embedder().embed_query("warm up")

    try:
        # In a thread: the client is synchronous, and blocking the event loop
        # during startup would stall the health probe that says we are ready.
        await asyncio.wait_for(asyncio.to_thread(_warm), timeout=30)
        logger.info("model client warmed")
    except Exception as exc:
        logger.warning("model client warm-up skipped: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _warm_model_client()
    yield


app = FastAPI(title="Research Paper Reading Companion", lifespan=lifespan)

# `/health` is deliberately unprefixed: it is a probe, not part of the API.
app.include_router(health.router)
app.include_router(me.router)
app.include_router(papers.router)
app.include_router(sessions.router)
app.include_router(sessions.citations_router)
app.include_router(memory.router)
app.include_router(memory.turns_router)
app.include_router(feedback.router)
app.include_router(feedback.debug_router)
