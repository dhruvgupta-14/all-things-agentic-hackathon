from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db

app = FastAPI(title="Research Paper Reading Companion")


@app.get("/health")
async def health(session: AsyncSession = Depends(get_db)):
    try:
        await session.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:
        db_status = f"error: {exc}"

    return {"database": db_status}