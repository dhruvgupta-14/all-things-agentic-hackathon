from fastapi import FastAPI

from app.routers import health, me, papers

app = FastAPI(title="Research Paper Reading Companion")

app.include_router(health.router)
app.include_router(me.router)
app.include_router(papers.router)
