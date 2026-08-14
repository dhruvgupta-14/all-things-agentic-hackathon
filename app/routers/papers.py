from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import Principal, get_current_user
from app.db.base import get_db
from app.db.models import Paper, UserPaperAccess

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("")
async def list_papers(
    principal: Principal = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """Papers this user may see.

    Possessing a paper_id grants nothing: the join through `user_paper_access`
    is the only thing that makes a paper visible, and a revoked grant drops out
    here immediately (ARCHITECTURE section 4.2).
    """
    rows = await session.execute(
        select(Paper, UserPaperAccess.nickname, UserPaperAccess.last_opened_at)
        .join(UserPaperAccess, UserPaperAccess.paper_id == Paper.paper_id)
        .where(
            UserPaperAccess.user_id == principal.user_id,
            UserPaperAccess.revoked_at.is_(None),
        )
        .order_by(UserPaperAccess.last_opened_at.desc().nullslast())
    )

    return [
        {
            "paper_id": str(paper.paper_id),
            "title": paper.title,
            "nickname": nickname,
            "processing_status": paper.processing_status,
            "processing_phase": paper.processing_phase,
            "last_opened_at": last_opened_at,
        }
        for paper, nickname, last_opened_at in rows.all()
    ]
