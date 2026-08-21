"""Feedback that visibly changes the next turn, and the debug strip.

The named-track requirement is not "collect feedback" — it is that feedback
*changes behaviour* and that the change is verifiable. `applied_to_turn_id` is
what makes it a join rather than a claim, so most of these are about that
column meaning what it says.
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Concept, Feedback, Session, Turn, User
from app.ingestion.concepts import normalize_name
from app.services.feedback import (
    DEFAULT_DEPTH,
    FeedbackRejected,
    FeedbackService,
    depth_instruction,
)


async def _principal(db_session: AsyncSession, signed_in: str) -> User:
    user = await db_session.scalar(select(User).where(User.auth_subject == signed_in))
    if user is None:
        user = User(auth_subject=signed_in)
        db_session.add(user)
        await db_session.flush()
    return user


async def _turn(db_session: AsyncSession, user: User) -> Turn:
    conversation = Session(user_id=user.user_id)
    db_session.add(conversation)
    await db_session.flush()
    turn = Turn(session_id=conversation.session_id, user_id=user.user_id, ordinal=0)
    db_session.add(turn)
    await db_session.flush()
    return turn


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


async def test_an_unknown_feedback_type_is_rejected(db_session: AsyncSession):
    user = await _principal(db_session, f"fb-{uuid.uuid4()}")
    turn = await _turn(db_session, user)

    with pytest.raises(FeedbackRejected):
        await FeedbackService(db_session).record(
            user_id=user.user_id,
            feedback_type="vibes_were_off",
            target_turn_id=turn.turn_id,
        )


async def test_feedback_needs_exactly_one_target(db_session: AsyncSession):
    """The CHECK constraint requires it; refusing here gives a usable error."""
    user = await _principal(db_session, f"fb-{uuid.uuid4()}")
    turn = await _turn(db_session, user)

    with pytest.raises(FeedbackRejected):
        await FeedbackService(db_session).record(
            user_id=user.user_id, feedback_type="helpful"
        )

    concept = Concept(
        user_id=user.user_id,
        canonical_name="ELBO",
        normalized_name=normalize_name("ELBO"),
    )
    db_session.add(concept)
    await db_session.flush()

    with pytest.raises(FeedbackRejected):
        await FeedbackService(db_session).record(
            user_id=user.user_id,
            feedback_type="helpful",
            target_turn_id=turn.turn_id,
            target_concept_id=concept.concept_id,
        )


# --------------------------------------------------------------------------
# What moves a preference, and what does not
# --------------------------------------------------------------------------


async def test_too_basic_moves_depth_up_one_notch(db_session: AsyncSession):
    """One notch, not to an extreme — "too basic" is not a request for a
    research seminar."""
    user = await _principal(db_session, f"fb-{uuid.uuid4()}")
    turn = await _turn(db_session, user)

    recorded = await FeedbackService(db_session).record(
        user_id=user.user_id, feedback_type="too_basic", target_turn_id=turn.turn_id
    )

    assert recorded.changed_preferences is True
    assert recorded.depth == "detailed"
    assert user.preferences["depth"] == "detailed"


async def test_too_advanced_moves_depth_down(db_session: AsyncSession):
    user = await _principal(db_session, f"fb-{uuid.uuid4()}")
    turn = await _turn(db_session, user)

    recorded = await FeedbackService(db_session).record(
        user_id=user.user_id, feedback_type="too_advanced", target_turn_id=turn.turn_id
    )

    assert recorded.depth == "introductory"


async def test_depth_does_not_run_past_the_ends(db_session: AsyncSession):
    user = await _principal(db_session, f"fb-{uuid.uuid4()}")
    turn = await _turn(db_session, user)
    service = FeedbackService(db_session)

    for _ in range(5):
        recorded = await service.record(
            user_id=user.user_id, feedback_type="too_basic", target_turn_id=turn.turn_id
        )

    assert recorded.depth == "expert"


async def test_a_thumbs_down_is_evidence_not_a_control(db_session: AsyncSession):
    """A single `not_helpful` must not silently retune how someone is taught."""
    user = await _principal(db_session, f"fb-{uuid.uuid4()}")
    turn = await _turn(db_session, user)

    recorded = await FeedbackService(db_session).record(
        user_id=user.user_id, feedback_type="not_helpful", target_turn_id=turn.turn_id
    )

    assert recorded.changed_preferences is False
    assert user.preferences.get("depth", DEFAULT_DEPTH) == DEFAULT_DEPTH


async def test_a_style_preference_needs_a_style_from_the_closed_set(
    db_session: AsyncSession,
):
    user = await _principal(db_session, f"fb-{uuid.uuid4()}")
    turn = await _turn(db_session, user)
    service = FeedbackService(db_session)

    with pytest.raises(FeedbackRejected):
        await service.record(
            user_id=user.user_id,
            feedback_type="style_preference",
            target_turn_id=turn.turn_id,
            preferred_style="interpretive_dance",
        )

    recorded = await service.record(
        user_id=user.user_id,
        feedback_type="style_preference",
        target_turn_id=turn.turn_id,
        preferred_style="numerical",
    )
    assert recorded.changed_preferences is True
    assert user.preferences["preferred_style"] == "numerical"


# --------------------------------------------------------------------------
# applied_to_turn_id — the column that makes it verifiable
# --------------------------------------------------------------------------


async def test_feedback_is_stamped_with_the_turn_it_changed(
    db_session: AsyncSession,
):
    user = await _principal(db_session, f"fb-{uuid.uuid4()}")
    first = await _turn(db_session, user)
    service = FeedbackService(db_session)

    recorded = await service.record(
        user_id=user.user_id, feedback_type="too_basic", target_turn_id=first.turn_id
    )

    later = Turn(session_id=first.session_id, user_id=user.user_id, ordinal=1)
    db_session.add(later)
    await db_session.flush()

    stamped = await service.apply_pending(user.user_id, later.turn_id)
    await db_session.flush()

    assert recorded.feedback_id in stamped
    row = await db_session.get(Feedback, recorded.feedback_id)
    assert row.applied_to_turn_id == later.turn_id


async def test_evidence_only_feedback_is_never_stamped(db_session: AsyncSession):
    """A `not_helpful` on an earlier answer did not compose this one, and
    claiming it did would make the column mean nothing."""
    user = await _principal(db_session, f"fb-{uuid.uuid4()}")
    first = await _turn(db_session, user)
    service = FeedbackService(db_session)

    recorded = await service.record(
        user_id=user.user_id, feedback_type="not_helpful", target_turn_id=first.turn_id
    )

    later = Turn(session_id=first.session_id, user_id=user.user_id, ordinal=1)
    db_session.add(later)
    await db_session.flush()
    await service.apply_pending(user.user_id, later.turn_id)
    await db_session.flush()

    row = await db_session.get(Feedback, recorded.feedback_id)
    assert row.applied_to_turn_id is None


async def test_feedback_is_stamped_once(db_session: AsyncSession):
    user = await _principal(db_session, f"fb-{uuid.uuid4()}")
    first = await _turn(db_session, user)
    service = FeedbackService(db_session)

    await service.record(
        user_id=user.user_id, feedback_type="too_basic", target_turn_id=first.turn_id
    )
    second = Turn(session_id=first.session_id, user_id=user.user_id, ordinal=1)
    db_session.add(second)
    await db_session.flush()
    await service.apply_pending(user.user_id, second.turn_id)
    await db_session.flush()

    third = Turn(session_id=first.session_id, user_id=user.user_id, ordinal=2)
    db_session.add(third)
    await db_session.flush()

    assert await service.apply_pending(user.user_id, third.turn_id) == []


async def test_stamping_does_not_reach_another_reader(db_session: AsyncSession):
    mine = await _principal(db_session, f"fb-{uuid.uuid4()}")
    theirs = User(auth_subject=f"other-{uuid.uuid4()}")
    db_session.add(theirs)
    await db_session.flush()
    their_turn = await _turn(db_session, theirs)
    service = FeedbackService(db_session)

    recorded = await service.record(
        user_id=theirs.user_id,
        feedback_type="too_basic",
        target_turn_id=their_turn.turn_id,
    )

    my_turn = await _turn(db_session, mine)
    await service.apply_pending(mine.user_id, my_turn.turn_id)
    await db_session.flush()

    row = await db_session.get(Feedback, recorded.feedback_id)
    assert row.applied_to_turn_id is None


# --------------------------------------------------------------------------
# How it reaches the agent
# --------------------------------------------------------------------------


def test_the_default_depth_adds_no_instruction():
    """Silence is the right instruction when nothing was asked for."""
    assert depth_instruction({}) is None
    assert depth_instruction(None) is None


def test_each_depth_produces_a_distinct_instruction():
    introductory = depth_instruction({"depth": "introductory"})
    expert = depth_instruction({"depth": "expert"})

    assert introductory and expert and introductory != expert
    assert "fundamentals" in introductory
    assert "mathematics" in expert


def test_a_preferred_style_reaches_the_agent():
    assert "numerical" in depth_instruction({"preferred_style": "numerical"})


# --------------------------------------------------------------------------
# The HTTP surface
# --------------------------------------------------------------------------


async def test_posting_feedback_reports_whether_it_changed_anything(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    user = await _principal(db_session, signed_in)
    turn = await _turn(db_session, user)

    changed = await client.post(
        "/api/feedback",
        json={"feedback_type": "too_basic", "target_turn_id": str(turn.turn_id)},
    )
    assert changed.status_code == 201
    assert changed.json()["applied"] is True

    noted = await client.post(
        "/api/feedback",
        json={"feedback_type": "helpful", "target_turn_id": str(turn.turn_id)},
    )
    # Honest rather than flattering: it was received, it changed nothing.
    assert noted.json()["applied"] is False


async def test_feedback_cannot_name_another_readers_turn(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    await _principal(db_session, signed_in)
    stranger = User(auth_subject=f"stranger-{uuid.uuid4()}")
    db_session.add(stranger)
    await db_session.flush()
    theirs = await _turn(db_session, stranger)

    response = await client.post(
        "/api/feedback",
        json={"feedback_type": "not_helpful", "target_turn_id": str(theirs.turn_id)},
    )

    assert response.status_code == 404


async def test_the_debug_strip_reports_what_the_last_turn_did(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    user = await _principal(db_session, signed_in)
    conversation = Session(user_id=user.user_id)
    db_session.add(conversation)
    await db_session.flush()
    db_session.add(
        Turn(
            session_id=conversation.session_id,
            user_id=user.user_id,
            ordinal=0,
            agent_action="callback",
            memory_read=True,
            grounding_status="grounded",
            tools_called=["retrieve_paper_context"],
            latency_ms=1234,
        )
    )
    await db_session.flush()

    body = (
        await client.get(f"/api/debug/sessions/{conversation.session_id}")
    ).json()

    assert body["activity"] == "FREE"
    assert body["last_turn"]["agent_action"] == "callback"
    assert body["last_turn"]["memory_read"] is True
    assert body["last_turn"]["latency_ms"] == 1234
    assert body["last_turn"]["tools_called"] == ["retrieve_paper_context"]


async def test_the_debug_strip_is_session_scoped(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    await _principal(db_session, signed_in)
    stranger = User(auth_subject=f"stranger-{uuid.uuid4()}")
    db_session.add(stranger)
    await db_session.flush()
    conversation = Session(user_id=stranger.user_id)
    db_session.add(conversation)
    await db_session.flush()

    response = await client.get(f"/api/debug/sessions/{conversation.session_id}")

    assert response.status_code == 404
