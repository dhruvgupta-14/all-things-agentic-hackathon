"""Feedback, and the session debug strip (ARCHITECTURE 15).

Two routes that exist for the same reason: making the system's behaviour
inspectable rather than asking to be trusted. One lets a reader change it; the
other shows what it actually did.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import Principal, get_current_user
from app.db.base import get_db
from app.db.models import (
    Chunk,
    Concept,
    Feedback,
    Session,
    Turn,
    TurnRetrieval,
)
from app.services.feedback import FeedbackRejected, FeedbackService

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    feedback_type: str
    target_turn_id: uuid.UUID | None = None
    target_concept_id: uuid.UUID | None = None
    comment: str | None = Field(default=None, max_length=2000)
    preferred_style: str | None = None


@router.post("", status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    body: FeedbackRequest,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Record feedback, and say plainly whether it changed anything.

    `applied` is deliberately honest: it reports whether a standing preference
    moved, not whether the feedback was received. Telling a reader their
    thumbs-down changed how they will be taught, when it did not, is worse
    than telling them it was noted.
    """
    # Ownership: a turn or concept belonging to someone else must not be
    # nameable, so both are checked against the principal before anything is
    # written.
    if body.target_turn_id is not None:
        owned = await db.scalar(
            select(Turn.turn_id).where(
                Turn.turn_id == body.target_turn_id,
                Turn.user_id == principal.user_id,
            )
        )
        if owned is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No such turn.")

    if body.target_concept_id is not None:
        owned = await db.scalar(
            select(Concept.concept_id).where(
                Concept.concept_id == body.target_concept_id,
                Concept.user_id == principal.user_id,
            )
        )
        if owned is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No such concept.")

    try:
        recorded = await FeedbackService(db).record(
            user_id=principal.user_id,
            feedback_type=body.feedback_type,
            target_turn_id=body.target_turn_id,
            target_concept_id=body.target_concept_id,
            comment=body.comment,
            preferred_style=body.preferred_style,
        )
    except FeedbackRejected as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    await db.commit()
    return {
        "feedback_id": str(recorded.feedback_id),
        "applied": recorded.changed_preferences,
        "depth": recorded.depth,
    }


# --------------------------------------------------------------------------
# The session debug strip
# --------------------------------------------------------------------------
debug_router = APIRouter(prefix="/api/debug", tags=["debug"])


@debug_router.get("/sessions/{session_id}")
async def debug_session(
    session_id: uuid.UUID,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """What the last turn actually did — the transparency panel.

    Everything here is read from what was recorded at the time, not
    recomputed: the retrieval set with its similarities, whether memory was
    read, which callback fired or why one did not, the tools the agent chose,
    and what it cost. A demo that claims determinism should be able to show
    its own working.
    """
    conversation = await db.scalar(
        select(Session).where(
            Session.session_id == session_id, Session.user_id == principal.user_id
        )
    )
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such session.")

    turn = await db.scalar(
        select(Turn)
        .where(Turn.session_id == session_id)
        .order_by(Turn.ordinal.desc())
        .limit(1)
    )

    payload = {
        "session_id": str(session_id),
        "activity": conversation.activity,
        "turn_count": conversation.turn_count,
        "pending_quiz": conversation.pending_quiz_id is not None,
        "last_turn": None,
    }
    if turn is None:
        return payload

    rows = (
        await db.execute(
            select(TurnRetrieval, Chunk)
            .join(Chunk, Chunk.chunk_id == TurnRetrieval.chunk_id)
            .where(TurnRetrieval.turn_id == turn.turn_id)
            .order_by(TurnRetrieval.rank)
        )
    ).all()

    callback_name = None
    if turn.callback_concept_id is not None:
        callback_name = await db.scalar(
            select(Concept.canonical_name).where(
                Concept.concept_id == turn.callback_concept_id
            )
        )

    payload["last_turn"] = {
        "turn_id": str(turn.turn_id),
        "ordinal": turn.ordinal,
        "agent_action": turn.agent_action,
        "grounding_status": turn.grounding_status,
        "memory_read": turn.memory_read,
        "explanation_style": turn.explanation_style,
        "callback_concept": callback_name,
        # One of these two is always set, never both and never neither —
        # suppression is a feature and is measured.
        "callback_suppressed_reason": turn.callback_suppressed_reason,
        "tools_called": list(turn.tools_called or []),
        "input_tokens": turn.input_tokens,
        "output_tokens": turn.output_tokens,
        "latency_ms": turn.latency_ms,
        "error_code": turn.error_code,
        "retrievals": [
            {
                "rank": retrieval.rank,
                "similarity": retrieval.similarity,
                "was_cited": retrieval.was_cited,
                "citation_marker": retrieval.citation_marker,
                "paper_id": str(chunk.paper_id),
                "page_start": chunk.page_start,
            }
            for retrieval, chunk in rows
        ],
    }
    return payload


@debug_router.get("/feedback")
async def recent_feedback(
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Feedback and the turn each piece visibly changed.

    `applied_to_turn_id` is the whole point: it makes "your feedback changed
    the next answer" a join rather than a claim.
    """
    rows = (
        await db.scalars(
            select(Feedback)
            .where(Feedback.user_id == principal.user_id)
            .order_by(Feedback.created_at.desc())
            .limit(50)
        )
    ).all()
    return {
        "feedback": [
            {
                "feedback_id": str(row.feedback_id),
                "feedback_type": row.feedback_type,
                "comment": row.comment,
                "applied_to_turn_id": (
                    str(row.applied_to_turn_id) if row.applied_to_turn_id else None
                ),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
    }
