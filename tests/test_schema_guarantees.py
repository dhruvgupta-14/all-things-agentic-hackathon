"""Structural guarantees the design relies on being unrepresentable to violate.

These are database-level assertions, not application logic. Each one is a rule
the architecture document treats as load-bearing, so a migration that quietly
drops a constraint should fail the suite rather than the demo.
"""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Session, Turn, User


async def _seed(session: AsyncSession) -> tuple[User, Session, Turn]:
    user = User(auth_subject=f"schema-test-{uuid.uuid4()}")
    session.add(user)
    await session.flush()

    conversation = Session(user_id=user.user_id)
    session.add(conversation)
    await session.flush()

    turn = Turn(session_id=conversation.session_id, user_id=user.user_id, ordinal=0)
    session.add(turn)
    await session.flush()

    return user, conversation, turn


async def test_turns_are_append_only(db_session: AsyncSession):
    _, _, turn = await _seed(db_session)

    with pytest.raises(DBAPIError, match="append-only"):
        async with db_session.begin_nested():
            await db_session.execute(
                text("UPDATE turns SET ordinal = 99 WHERE turn_id = :id"),
                {"id": turn.turn_id},
            )


async def test_turns_cannot_be_deleted(db_session: AsyncSession):
    _, _, turn = await _seed(db_session)

    with pytest.raises(DBAPIError, match="append-only"):
        async with db_session.begin_nested():
            await db_session.execute(
                text("DELETE FROM turns WHERE turn_id = :id"), {"id": turn.turn_id}
            )


async def test_quiz_pending_requires_a_pending_quiz(db_session: AsyncSession):
    """State and payload cannot disagree (ARCHITECTURE 4.6)."""
    _, conversation, _ = await _seed(db_session)

    with pytest.raises(DBAPIError):
        async with db_session.begin_nested():
            await db_session.execute(
                text(
                    "UPDATE sessions SET activity = 'QUIZ_PENDING' "
                    "WHERE session_id = :id"
                ),
                {"id": conversation.session_id},
            )


async def test_citation_requires_a_marker(db_session: AsyncSession):
    """`was_cited` and `citation_marker` are set together or not at all."""
    _, _, turn = await _seed(db_session)

    with pytest.raises(DBAPIError):
        async with db_session.begin_nested():
            await db_session.execute(
                text(
                    "INSERT INTO turn_retrievals "
                    "(turn_id, chunk_id, rank, similarity, was_cited) "
                    "VALUES (:t, gen_random_uuid(), 1, 0.9, true)"
                ),
                {"t": turn.turn_id},
            )


async def test_callback_requires_a_memory_read(db_session: AsyncSession):
    """A proactive callback with no memory read behind it is not recordable."""
    _, conversation, _ = await _seed(db_session)

    with pytest.raises(DBAPIError):
        async with db_session.begin_nested():
            await db_session.execute(
                text(
                    "INSERT INTO turns "
                    "(session_id, user_id, ordinal, callback_concept_id, memory_read) "
                    "SELECT :s, user_id, 1, gen_random_uuid(), false "
                    "FROM sessions WHERE session_id = :s"
                ),
                {"s": conversation.session_id},
            )


async def test_turn_ordinals_are_unique_per_session(db_session: AsyncSession):
    """The uniqueness that makes retry writes idempotent."""
    user, conversation, _ = await _seed(db_session)

    with pytest.raises(DBAPIError):
        async with db_session.begin_nested():
            db_session.add(
                Turn(
                    session_id=conversation.session_id,
                    user_id=user.user_id,
                    ordinal=0,
                )
            )
            await db_session.flush()


async def test_vector_extension_and_hnsw_indexes_exist(db_session: AsyncSession):
    """Retrieval is silently non-functional without these."""
    extension = await db_session.scalar(
        text("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
    )
    assert extension == 1

    indexes = await db_session.scalars(
        text(
            "SELECT indexname FROM pg_indexes "
            "WHERE schemaname = 'public' AND indexdef ILIKE '%hnsw%'"
        )
    )
    assert set(indexes.all()) == {"ix_chunks_embedding_hnsw", "ix_concepts_embedding_hnsw"}
