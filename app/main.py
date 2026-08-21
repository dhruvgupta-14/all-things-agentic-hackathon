import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.routers import feedback, health, internal, me, memory, papers, sessions
from app.spa import mount_spa, spa_dist

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
    def _warm() -> None:
        from app.services import embeddings

        embeddings.get_embedder().embed_query("warm up")

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
# Not part of the browser API: Cloud Tasks pushes here with an OIDC token.
app.include_router(internal.router)

# Last, and deliberately so: this mounts at "/", and Starlette matches routes
# in registration order. Registered any earlier it would swallow the API — an
# unknown /api path would come back as index.html instead of a JSON 404.
mount_spa(app, spa_dist(get_settings().spa_dist_dir))
