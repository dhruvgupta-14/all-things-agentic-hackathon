"""The append-only tables stay append-only, and a user can still be erased.

HANDOFF 6.1: `turns`, `observations` and `quiz_attempts` each declare
`ON DELETE CASCADE` on `user_id` and carry a `BEFORE UPDATE OR DELETE` trigger.
Both cannot hold. These tests pin the resolution from both sides — the block is
still there for every ordinary path, and the one deliberate path works.
"""

import uuid

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Concept,
    Message,
    Observation,
    Paper,
    Quiz,
    QuizAttempt,
    Session,
    Turn,
    User,
)
from app.services.erasure import erase_user


async def _user_with_a_history(session: AsyncSession) -> User:
    """A user carrying a row in every table that blocks deletion."""
    user = User(auth_subject=f"erasure-test-{uuid.uuid4()}")
    session.add(user)
    await session.flush()

    conversation = Session(user_id=user.user_id)
    session.add(conversation)
    await session.flush()

    turn = Turn(session_id=conversation.session_id, user_id=user.user_id, ordinal=0)
    session.add(turn)
    await session.flush()

    session.add(
        Message(
            session_id=conversation.session_id,
            user_id=user.user_id,
            turn_id=turn.turn_id,
            ordinal=0,
            role="user",
            content="what is the ELBO?",
        )
    )

    concept = Concept(
        user_id=user.user_id,
        canonical_name="Variational lower bound",
        normalized_name=f"variational lower bound {uuid.uuid4()}",
    )
    session.add(concept)
    await session.flush()

    session.add(
        Observation(
            user_id=user.user_id,
            concept_id=concept.concept_id,
            turn_id=turn.turn_id,
            signal_type="explicit_confusion",
            signal_source="explicit",
            weight=0.8,
        )
    )

    paper = Paper(
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        storage_uri=f"file://erasure-{uuid.uuid4()}.pdf",
        processing_status="ready",
    )
    session.add(paper)
    await session.flush()

    quiz = Quiz(
        user_id=user.user_id,
        concept_id=concept.concept_id,
        paper_id=paper.paper_id,
        question="State the ELBO.",
        rubric={"must_mention": ["KL"]},
        grounding_chunk_ids=[uuid.uuid4()],
    )
    session.add(quiz)
    await session.flush()

    session.add(
        QuizAttempt(
            quiz_id=quiz.quiz_id,
            user_id=user.user_id,
            turn_id=turn.turn_id,
            answer_text="log p(x) minus KL",
            grade="correct",
            attempt_no=1,
        )
    )
    await session.flush()
    return user


# --------------------------------------------------------------------------
# The guarantee still holds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("table", ["turns", "observations", "quiz_attempts"])
async def test_updates_are_refused_unconditionally(
    db_session: AsyncSession, table: str
):
    """UPDATE has no escape hatch at all — erasure is not a licence to rewrite."""
    user = await _user_with_a_history(db_session)

    with pytest.raises(DBAPIError, match="append-only"):
        async with db_session.begin_nested():
            await db_session.execute(
                text("SELECT set_config('app.erasure', 'on', true)")
            )
            await db_session.execute(
                text(f"UPDATE {table} SET user_id = user_id WHERE user_id = :u"),
                {"u": user.user_id},
            )


@pytest.mark.parametrize("table", ["turns", "observations", "quiz_attempts"])
async def test_ordinary_deletes_are_still_refused(db_session: AsyncSession, table: str):
    """Without the flag, nothing changed: the append-only block is intact."""
    user = await _user_with_a_history(db_session)

    with pytest.raises(DBAPIError, match="append-only"):
        async with db_session.begin_nested():
            await db_session.execute(
                text(f"DELETE FROM {table} WHERE user_id = :u"), {"u": user.user_id}
            )


async def test_deleting_a_user_without_the_flag_is_refused(db_session: AsyncSession):
    """The cascade is what used to fail, so it is what has to keep failing."""
    user = await _user_with_a_history(db_session)

    with pytest.raises(DBAPIError, match="append-only"):
        async with db_session.begin_nested():
            await db_session.execute(
                text("DELETE FROM users WHERE user_id = :u"), {"u": user.user_id}
            )


# --------------------------------------------------------------------------
# And erasure works
# --------------------------------------------------------------------------


async def test_erase_user_removes_every_dependent_row(db_session: AsyncSession):
    user = await _user_with_a_history(db_session)
    user_id = user.user_id

    erased = await erase_user(db_session, user_id)
    await db_session.flush()

    assert erased is True

    for model in (Turn, Observation, QuizAttempt, Message, Concept, Session):
        remaining = await db_session.scalar(
            select(func.count())
            .select_from(model)
            .where(model.user_id == user_id)
        )
        assert remaining == 0, f"{model.__tablename__} still holds rows"

    assert await db_session.get(User, user_id) is None


async def test_erasure_does_not_touch_anyone_else(db_session: AsyncSession):
    """Scoped counts are facts (HANDOFF 7.3) — the decoy user must survive."""
    doomed = await _user_with_a_history(db_session)
    bystander = await _user_with_a_history(db_session)

    await erase_user(db_session, doomed.user_id)
    await db_session.flush()

    assert await db_session.get(User, bystander.user_id) is not None
    surviving_turns = await db_session.scalar(
        select(func.count()).select_from(Turn).where(Turn.user_id == bystander.user_id)
    )
    assert surviving_turns == 1


async def test_the_door_closes_behind_the_erasure(db_session: AsyncSession):
    """The flag must not stay on for the rest of the transaction."""
    doomed = await _user_with_a_history(db_session)
    bystander = await _user_with_a_history(db_session)

    await erase_user(db_session, doomed.user_id)
    await db_session.flush()

    with pytest.raises(DBAPIError, match="append-only"):
        async with db_session.begin_nested():
            await db_session.execute(
                text("DELETE FROM turns WHERE user_id = :u"), {"u": bystander.user_id}
            )


async def test_erasing_an_unknown_user_reports_nothing_happened(
    db_session: AsyncSession,
):
    assert await erase_user(db_session, uuid.uuid4()) is False
