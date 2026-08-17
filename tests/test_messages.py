"""Durable conversation history.

The property under test throughout: PostgreSQL is the single owner of what was
said. A conversation must be reconstructable from `messages` alone, without
ADK, without an in-memory session, and after a process restart.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MESSAGE_RETENTION_DAYS, Message, Session, Turn, User
from app.services.messages import (
    MAX_CONTENT_CHARS,
    MessageService,
    estimate_tokens,
)


async def _session_for(db_session: AsyncSession) -> tuple[User, Session]:
    user = User(auth_subject=f"messages-{uuid.uuid4()}")
    db_session.add(user)
    await db_session.flush()

    conversation = Session(user_id=user.user_id)
    db_session.add(conversation)
    await db_session.flush()
    return user, conversation


# --------------------------------------------------------------------------
# Persistence and reconstruction
# --------------------------------------------------------------------------


async def test_an_exchange_is_persisted_in_order(db_session: AsyncSession):
    user, conversation = await _session_for(db_session)
    service = MessageService(db_session)

    await service.append_exchange(
        session_id=conversation.session_id,
        user_id=user.user_id,
        user_content="What is the ELBO?",
        assistant_content="It is a lower bound on the log marginal likelihood.",
    )

    transcript = await service.transcript(conversation.session_id)
    assert [m.role for m in transcript] == ["user", "assistant"]
    assert [m.ordinal for m in transcript] == [0, 1]
    assert transcript[0].content == "What is the ELBO?"


async def test_a_conversation_reconstructs_from_messages_alone(
    db_session: AsyncSession,
):
    """No ADK, no in-memory session — just the table."""
    user, conversation = await _session_for(db_session)
    service = MessageService(db_session)

    exchanges = [
        ("What is the ELBO?", "A lower bound on the log marginal likelihood."),
        ("Why does it contain a KL term?", "Because the bound is a difference."),
        ("Show me the derivation.", "Start from Jensen's inequality."),
    ]
    for question, answer in exchanges:
        await service.append_exchange(
            session_id=conversation.session_id,
            user_id=user.user_id,
            user_content=question,
            assistant_content=answer,
        )

    transcript = await service.transcript(conversation.session_id)

    assert len(transcript) == 6
    assert [m.ordinal for m in transcript] == [0, 1, 2, 3, 4, 5]
    rebuilt = [(transcript[i].content, transcript[i + 1].content) for i in range(0, 6, 2)]
    assert rebuilt == exchanges


async def test_history_survives_a_new_session_object(db_session: AsyncSession):
    """Nothing is held in process memory.

    A fresh ORM identity map reading the same rows is the closest a
    transaction-scoped test can get to an instance restart; the retention and
    transcript tests cover the rest.
    """
    user, conversation = await _session_for(db_session)
    await MessageService(db_session).append_exchange(
        session_id=conversation.session_id,
        user_id=user.user_id,
        user_content="remembered?",
        assistant_content="yes",
    )
    db_session.expunge_all()

    transcript = await MessageService(db_session).transcript(conversation.session_id)
    assert [m.content for m in transcript] == ["remembered?", "yes"]


async def test_ordinals_are_unique_within_a_session(db_session: AsyncSession):
    """The constraint that makes a retried write idempotent rather than doubled."""
    user, conversation = await _session_for(db_session)
    db_session.add(
        Message(
            session_id=conversation.session_id,
            user_id=user.user_id,
            ordinal=0,
            role="user",
            content="first",
        )
    )
    await db_session.flush()

    with pytest.raises(DBAPIError):
        async with db_session.begin_nested():
            db_session.add(
                Message(
                    session_id=conversation.session_id,
                    user_id=user.user_id,
                    ordinal=0,
                    role="user",
                    content="collision",
                )
            )
            await db_session.flush()


async def test_two_sessions_keep_separate_histories(db_session: AsyncSession):
    user, first = await _session_for(db_session)
    second = Session(user_id=user.user_id)
    db_session.add(second)
    await db_session.flush()

    service = MessageService(db_session)
    await service.append(
        session_id=first.session_id, user_id=user.user_id, role="user", content="one"
    )
    await service.append(
        session_id=second.session_id, user_id=user.user_id, role="user", content="two"
    )

    assert [m.content for m in await service.transcript(first.session_id)] == ["one"]
    assert [m.content for m in await service.transcript(second.session_id)] == ["two"]


async def test_a_message_can_carry_its_turn_for_provenance(db_session: AsyncSession):
    user, conversation = await _session_for(db_session)
    turn = Turn(session_id=conversation.session_id, user_id=user.user_id, ordinal=0)
    db_session.add(turn)
    await db_session.flush()

    message = await MessageService(db_session).append(
        session_id=conversation.session_id,
        user_id=user.user_id,
        role="assistant",
        content="grounded answer",
        turn_id=turn.turn_id,
    )
    assert message.turn_id == turn.turn_id


# --------------------------------------------------------------------------
# Immutability
# --------------------------------------------------------------------------


async def test_message_content_cannot_be_rewritten(db_session: AsyncSession):
    """A transcript that can be edited is not a transcript."""
    user, conversation = await _session_for(db_session)
    message = await MessageService(db_session).append(
        session_id=conversation.session_id,
        user_id=user.user_id,
        role="user",
        content="what I actually said",
    )

    with pytest.raises(DBAPIError, match="immutable"):
        async with db_session.begin_nested():
            await db_session.execute(
                text("UPDATE messages SET content = :c WHERE message_id = :id"),
                {"c": "what I wish I had said", "id": message.message_id},
            )


async def test_messages_can_be_deleted(db_session: AsyncSession):
    """Deliberately not blocked: retention and user-cascade both need it."""
    user, conversation = await _session_for(db_session)
    message = await MessageService(db_session).append(
        session_id=conversation.session_id,
        user_id=user.user_id,
        role="user",
        content="ephemeral",
    )

    await db_session.execute(
        text("DELETE FROM messages WHERE message_id = :id"), {"id": message.message_id}
    )
    remaining = await db_session.scalar(
        select(func.count())
        .select_from(Message)
        .where(Message.session_id == conversation.session_id)
    )
    assert remaining == 0


@pytest.mark.parametrize("role", ["user", "assistant", "summary"])
async def test_valid_roles_are_accepted(db_session: AsyncSession, role: str):
    user, conversation = await _session_for(db_session)
    message = await MessageService(db_session).append(
        session_id=conversation.session_id,
        user_id=user.user_id,
        role=role,
        content="content",
    )
    assert message.role == role


async def test_tool_output_is_not_a_valid_role(db_session: AsyncSession):
    """Tool results are working memory, not conversation (decision recorded)."""
    user, conversation = await _session_for(db_session)

    with pytest.raises(DBAPIError):
        async with db_session.begin_nested():
            db_session.add(
                Message(
                    session_id=conversation.session_id,
                    user_id=user.user_id,
                    ordinal=0,
                    role="tool",
                    content="retrieved chunks",
                )
            )
            await db_session.flush()


# --------------------------------------------------------------------------
# Context window
# --------------------------------------------------------------------------


async def test_history_is_returned_oldest_first(db_session: AsyncSession):
    user, conversation = await _session_for(db_session)
    service = MessageService(db_session)
    for index in range(3):
        await service.append(
            session_id=conversation.session_id,
            user_id=user.user_id,
            role="user",
            content=f"message {index}",
        )

    history = await service.history_for_context(conversation.session_id)
    assert [h.content for h in history] == ["message 0", "message 1", "message 2"]


async def test_the_budget_keeps_the_most_recent_history(db_session: AsyncSession):
    """The end of a conversation is the part that matters."""
    user, conversation = await _session_for(db_session)
    service = MessageService(db_session)
    for index in range(20):
        await service.append(
            session_id=conversation.session_id,
            user_id=user.user_id,
            role="user",
            content=f"message {index} " + "padding " * 40,
        )

    history = await service.history_for_context(
        conversation.session_id, token_budget=200
    )

    assert history, "a budget must never return nothing"
    assert "message 19" in history[-1].content
    assert len(history) < 20


async def test_an_over_budget_message_is_still_returned(db_session: AsyncSession):
    """Returning nothing would leave the agent with no context at all."""
    user, conversation = await _session_for(db_session)
    service = MessageService(db_session)
    await service.append(
        session_id=conversation.session_id,
        user_id=user.user_id,
        role="user",
        content="enormous " * 500,
    )

    history = await service.history_for_context(conversation.session_id, token_budget=10)
    assert len(history) == 1


async def test_oversized_content_is_refused_with_a_clear_error(
    db_session: AsyncSession,
):
    """A sentence naming the limit beats an IntegrityError from mid-turn."""
    user, conversation = await _session_for(db_session)

    with pytest.raises(ValueError, match="the limit is"):
        await MessageService(db_session).append(
            session_id=conversation.session_id,
            user_id=user.user_id,
            role="assistant",
            content="x" * (MAX_CONTENT_CHARS + 1),
        )


async def test_empty_content_is_refused(db_session: AsyncSession):
    user, conversation = await _session_for(db_session)

    with pytest.raises(ValueError, match="must have content"):
        await MessageService(db_session).append(
            session_id=conversation.session_id,
            user_id=user.user_id,
            role="user",
            content="   ",
        )


async def test_summaries_are_returned_like_any_other_message(
    db_session: AsyncSession,
):
    """Older history is summarised, never truncated (ARCHITECTURE 9.2 step 6)."""
    user, conversation = await _session_for(db_session)
    service = MessageService(db_session)

    await service.append(
        session_id=conversation.session_id,
        user_id=user.user_id,
        role="summary",
        content="Earlier: the reader struggled with the ELBO.",
    )
    await service.append(
        session_id=conversation.session_id,
        user_id=user.user_id,
        role="user",
        content="go on",
    )

    history = await service.history_for_context(conversation.session_id)
    assert [h.role for h in history] == ["summary", "user"]


async def test_empty_history_is_empty_not_an_error(db_session: AsyncSession):
    _, conversation = await _session_for(db_session)
    assert await MessageService(db_session).history_for_context(
        conversation.session_id
    ) == []


def test_token_estimate_is_proportional_to_length():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 400) == 100


# --------------------------------------------------------------------------
# Retention
# --------------------------------------------------------------------------


async def test_messages_past_the_retention_window_are_removed(
    db_session: AsyncSession,
):
    user, conversation = await _session_for(db_session)
    service = MessageService(db_session)

    recent = await service.append(
        session_id=conversation.session_id,
        user_id=user.user_id,
        role="user",
        content="recent",
    )
    old = await service.append(
        session_id=conversation.session_id,
        user_id=user.user_id,
        role="user",
        content="ancient",
    )
    # Backdate past the window. UPDATE is blocked by the immutability trigger,
    # so the row is replaced rather than edited.
    await db_session.execute(
        text("DELETE FROM messages WHERE message_id = :id"), {"id": old.message_id}
    )
    db_session.add(
        Message(
            session_id=conversation.session_id,
            user_id=user.user_id,
            ordinal=99,
            role="user",
            content="ancient",
            created_at=datetime.now(UTC) - timedelta(days=MESSAGE_RETENTION_DAYS + 1),
        )
    )
    await db_session.flush()

    removed = await service.prune_expired()

    assert removed >= 1
    surviving = [m.content for m in await service.transcript(conversation.session_id)]
    assert surviving == ["recent"]
    assert recent.message_id is not None


async def test_retention_leaves_messages_inside_the_window(db_session: AsyncSession):
    user, conversation = await _session_for(db_session)
    service = MessageService(db_session)
    db_session.add(
        Message(
            session_id=conversation.session_id,
            user_id=user.user_id,
            ordinal=0,
            role="user",
            content="just inside",
            created_at=datetime.now(UTC) - timedelta(days=MESSAGE_RETENTION_DAYS - 1),
        )
    )
    await db_session.flush()

    await service.prune_expired()

    assert len(await service.transcript(conversation.session_id)) == 1


async def test_deleting_a_user_removes_their_messages(db_session: AsyncSession):
    """Cascade must still work — which is why DELETE is not blocked."""
    user, conversation = await _session_for(db_session)
    await MessageService(db_session).append(
        session_id=conversation.session_id,
        user_id=user.user_id,
        role="user",
        content="private",
    )

    await db_session.execute(
        text("DELETE FROM users WHERE user_id = :id"), {"id": user.user_id}
    )

    remaining = await db_session.scalar(
        select(func.count())
        .select_from(Message)
        .where(Message.user_id == user.user_id)
    )
    assert remaining == 0
