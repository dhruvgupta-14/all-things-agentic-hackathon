from fastapi import FastAPI

from app.routers import health, me, papers, sessions

app = FastAPI(title="Research Paper Reading Companion")

# `/health` is deliberately unprefixed: it is a probe, not part of the API.
app.include_router(health.router)
app.include_router(me.router)
app.include_router(papers.router)
app.include_router(sessions.router)
app.include_router(sessions.citations_router)
