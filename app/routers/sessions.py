"""Sessions and turns (ARCHITECTURE 15).

`user_id` appears in no path, query or body. Session ownership is asserted on
every request against the verified principal, and a session belonging to
someone else is a 404 rather than a 403 — a 403 would confirm the id is real.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import Principal, get_current_user
from app.db.base import get_db
from app.db.models import Paper, Session, UserPaperAccess
from app.services.turns import TurnPipeline

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

MAX_MESSAGE_CHARS = 8000


class CreateSessionRequest(BaseModel):
    paper_id: uuid.UUID | None = None


class TurnRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


def _serialize(conversation: Session) -> dict:
    return {
        "session_id": str(conversation.session_id),
        "activity": conversation.activity,
        "active_paper_id": (
            str(conversation.active_paper_id) if conversation.active_paper_id else None
        ),
        "active_concept_id": (
            str(conversation.active_concept_id)
            if conversation.active_concept_id
            else None
        ),
        "turn_count": conversation.turn_count,
    }


async def _owned_session(
    session_id: uuid.UUID, principal: Principal, db: AsyncSession
) -> Session:
    """Step 2 ⛨ — load and assert ownership, or 404."""
    conversation = await db.scalar(
        select(Session).where(
            Session.session_id == session_id, Session.user_id == principal.user_id
        )
    )
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No such session."
        )
    return conversation


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_session(
    body: CreateSessionRequest,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Open a session, optionally on a paper the caller may read."""
    if body.paper_id is not None:
        grant = await db.scalar(
            select(UserPaperAccess).where(
                UserPaperAccess.user_id == principal.user_id,
                UserPaperAccess.paper_id == body.paper_id,
                UserPaperAccess.revoked_at.is_(None),
            )
        )
        if grant is None:
            # Indistinguishable from a paper that does not exist.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="No such paper."
            )

    conversation = Session(user_id=principal.user_id, active_paper_id=body.paper_id)
    db.add(conversation)
    await db.commit()
    return _serialize(conversation)


@router.get("")
async def list_sessions(
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The caller's sessions, most recently active first.

    Scoped by `user_id` in the WHERE clause rather than filtered afterwards, so
    there is no arrangement of query parameters that widens it. The paper title
    is joined in because the rail renders one row per session and should not
    have to fetch each paper separately.
    """
    rows = await db.execute(
        select(Session, Paper.title)
        .outerjoin(Paper, Paper.paper_id == Session.active_paper_id)
        .where(Session.user_id == principal.user_id)
        .order_by(Session.last_activity_at.desc())
    )

    return [
        {
            **_serialize(conversation),
            "paper_title": title,
            "started_at": conversation.started_at,
            "last_activity_at": conversation.last_activity_at,
        }
        for conversation, title in rows.all()
    ]


@router.get("/{session_id}")
async def get_session(
    session_id: uuid.UUID,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return _serialize(await _owned_session(session_id, principal, db))


@router.get("/{session_id}/messages")
async def get_messages(
    session_id: uuid.UUID,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The durable transcript, for reload and for the debug strip.

    This is what makes a conversation survive a refresh: the client rebuilds
    from PostgreSQL, not from anything held in memory.
    """
    from app.services.messages import MessageService

    conversation = await _owned_session(session_id, principal, db)
    transcript = await MessageService(db).transcript(conversation.session_id)
    return [
        {
            "message_id": str(message.message_id),
            "role": message.role,
            "content": message.content,
            "turn_id": str(message.turn_id) if message.turn_id else None,
            "created_at": message.created_at,
        }
        for message in transcript
    ]


@router.post("/{session_id}/turns")
async def create_turn(
    session_id: uuid.UUID,
    body: TurnRequest,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Ask a question. Responds as an SSE stream (see `app/schemas/sse.py`)."""
    conversation = await _owned_session(session_id, principal, db)
    pipeline = TurnPipeline(db)

    return StreamingResponse(
        pipeline.run(conversation, principal.user_id, body.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Streaming through a proxy that buffers would defeat the point.
            "X-Accel-Buffering": "no",
        },
    )


citations_router = APIRouter(prefix="/api/citations", tags=["citations"])


@citations_router.get("/{turn_id}/{chunk_id}")
async def get_citation(
    turn_id: uuid.UUID,
    chunk_id: uuid.UUID,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The click-through a judge uses to verify grounding in two seconds.

    Reachable only through a `turn_retrievals` row that was actually cited, and
    only for a turn the caller owns — so a chunk id alone opens nothing.
    """
    from app.db.models import Chunk, Section, Turn, TurnRetrieval

    row = (
        await db.execute(
            select(Chunk, Section, TurnRetrieval, Paper)
            .join(TurnRetrieval, TurnRetrieval.chunk_id == Chunk.chunk_id)
            .join(Turn, Turn.turn_id == TurnRetrieval.turn_id)
            .join(Section, Section.section_id == Chunk.section_id)
            .join(Paper, Paper.paper_id == Chunk.paper_id)
            .where(
                TurnRetrieval.turn_id == turn_id,
                TurnRetrieval.chunk_id == chunk_id,
                TurnRetrieval.was_cited.is_(True),
                Turn.user_id == principal.user_id,
            )
        )
    ).first()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No such citation."
        )

    chunk, section, retrieval, paper = row
    return {
        "chunk_id": str(chunk.chunk_id),
        "paper_id": str(paper.paper_id),
        "paper_title": paper.title,
        "marker": retrieval.citation_marker,
        "content": chunk.content,
        "section_path": section.section_path,
        "section_heading": section.heading,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "similarity": retrieval.similarity,
    }
