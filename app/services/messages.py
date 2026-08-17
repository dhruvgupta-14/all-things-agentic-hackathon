"""Durable conversation history.

PostgreSQL owns the transcript; ADK does not. Each turn hydrates a throwaway
in-memory ADK session from these rows, runs, and hands back a response we
persist ourselves — so history survives an instance being reclaimed, and
swapping the agent framework would not lose a single message.

Three things live here and nowhere else:

* appending a turn's user and assistant messages, in one ordered write
* reading back the window that fits the context budget, summaries included
* sweeping away anything past the retention window
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MESSAGE_RETENTION_DAYS, Message

logger = logging.getLogger(__name__)

# Characters per token, matching the chunker's estimate. A real tokenizer is
# not worth a dependency for context budgeting.
CHARS_PER_TOKEN = 4

# The schema's hard limit on `messages.content`. Checked here so an oversized
# message fails with a sentence naming the limit, rather than as a raw
# IntegrityError surfacing from the middle of a turn.
MAX_CONTENT_CHARS = 32000

# How much of the conversation to hand the agent. The rest is left behind for
# summarisation rather than silently cut (ARCHITECTURE 9.2 step 6).
DEFAULT_HISTORY_TOKEN_BUDGET = 3000

# Retention runs on append rather than on a schedule: no new infrastructure,
# and it happens exactly when the data that needs sweeping is being created.
# Throttled per process so it costs one query an hour, not one per turn.
_PRUNE_INTERVAL_SECONDS = 3600
_last_prune_at: float = 0.0


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


@dataclass(slots=True)
class HistoryMessage:
    """One message, in the shape the agent runtime wants."""

    role: str
    content: str
    token_count: int


class MessageService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _next_ordinal(self, session_id: uuid.UUID) -> int:
        highest = await self._session.scalar(
            select(func.max(Message.ordinal)).where(Message.session_id == session_id)
        )
        return 0 if highest is None else highest + 1

    async def append(
        self,
        *,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        role: str,
        content: str,
        turn_id: uuid.UUID | None = None,
    ) -> Message:
        """Append one message to a session's history.

        Raises ValueError when the content is empty or over the schema limit.
        A transcript is not truncated silently — losing part of what was said
        would make the record a summary pretending to be a transcript.
        """
        if not content.strip():
            raise ValueError("a message must have content")
        if len(content) > MAX_CONTENT_CHARS:
            raise ValueError(
                f"message content is {len(content)} characters; "
                f"the limit is {MAX_CONTENT_CHARS}"
            )

        message = Message(
            session_id=session_id,
            user_id=user_id,
            turn_id=turn_id,
            ordinal=await self._next_ordinal(session_id),
            role=role,
            content=content,
            token_count=min(estimate_tokens(content), 32767),
        )
        self._session.add(message)
        await self._session.flush()
        return message

    async def append_exchange(
        self,
        *,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        user_content: str,
        assistant_content: str,
        turn_id: uuid.UUID | None = None,
    ) -> tuple[Message, Message]:
        """Persist a turn's two messages in order.

        Written together so a session's history can never contain a question
        without its answer, which is what makes the transcript replayable.
        """
        asked = await self.append(
            session_id=session_id,
            user_id=user_id,
            role="user",
            content=user_content,
            turn_id=turn_id,
        )
        answered = await self.append(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            content=assistant_content,
            turn_id=turn_id,
        )
        await self.prune_expired_if_due()
        return asked, answered

    async def history_for_context(
        self,
        session_id: uuid.UUID,
        *,
        token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET,
    ) -> list[HistoryMessage]:
        """The most recent history that fits the budget, oldest first.

        Walks backwards from the newest message and stops at the budget, then
        restores chronological order — so the agent always sees the end of the
        conversation, which is the part that matters, and never a window that
        starts mid-exchange going forwards.
        """
        rows = list(
            (
                await self._session.scalars(
                    select(Message)
                    .where(Message.session_id == session_id)
                    .order_by(Message.ordinal.desc())
                )
            ).all()
        )

        selected: list[Message] = []
        spent = 0
        for row in rows:
            cost = row.token_count or estimate_tokens(row.content)
            if selected and spent + cost > token_budget:
                break
            selected.append(row)
            spent += cost

        selected.reverse()
        return [
            HistoryMessage(
                role=row.role,
                content=row.content,
                token_count=row.token_count or estimate_tokens(row.content),
            )
            for row in selected
        ]

    async def transcript(self, session_id: uuid.UUID) -> list[Message]:
        """The whole conversation in order. For replay and inspection."""
        return list(
            (
                await self._session.scalars(
                    select(Message)
                    .where(Message.session_id == session_id)
                    .order_by(Message.ordinal)
                )
            ).all()
        )

    async def prune_expired(self) -> int:
        """Delete history past the retention window. Returns rows removed."""
        cutoff = datetime.now(UTC) - timedelta(days=MESSAGE_RETENTION_DAYS)
        result = await self._session.execute(
            delete(Message).where(Message.created_at < cutoff)
        )
        removed = result.rowcount or 0
        if removed:
            logger.info(
                "pruned expired messages",
                extra={"removed": removed, "retention_days": MESSAGE_RETENTION_DAYS},
            )
        return removed

    async def prune_expired_if_due(self) -> int:
        """Run the sweep at most once an hour per process."""
        global _last_prune_at
        now = time.monotonic()
        if now - _last_prune_at < _PRUNE_INTERVAL_SECONDS:
            return 0
        _last_prune_at = now
        return await self.prune_expired()
