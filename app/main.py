from fastapi import FastAPI

from app.routers import feedback, health, me, memory, papers, sessions

app = FastAPI(title="Research Paper Reading Companion")

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
