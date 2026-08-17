"""TurnPipeline — the deterministic wrapper around the agent (ARCHITECTURE 9.2).

The agent decides what to search for and how to phrase the answer. Everything
that has to be *correct* rather than plausible happens here: identity, session
ownership, retrieval scope, citation verification, and every database write.

Steps implemented in this slice, numbered as in §9.2:

    1-2  identity and session ownership          (the router's dependencies)
    5    build scope from session state          ⛨
    6    assemble context from `messages`
    7    agent loop, tools receive injected scope
    8    compose
    9    verify citations
    11   persist: turn + turn_retrievals + messages, one transaction
    12   stream

Steps 3-4, 10 and 13 (quiz routing, memory prefetch, callback gate, learning
signals) arrive with the tools that need them; a turn currently records
`memory_read = False` because nothing has read memory yet, which is honest
rather than aspirational.

Citations are verified *before* the first token is streamed. That costs
latency, and buys a stream that is never retracted: a marker the reader sees
has already been matched to a passage.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.runner import AgentUnavailable, run_turn
from app.agent.tools import ToolScopeViolation, TurnToolContext
from app.db.models import Paper, Session, Turn, TurnRetrieval
from app.schemas.sse import (
    CitationPayload,
    CitationsEvent,
    DoneEvent,
    ErrorEvent,
    MemoryUsedEvent,
    StateEvent,
    TokenEvent,
)
from app.services import citations as citation_verifier
from app.services.messages import MessageService
from app.services.retrieval import (
    RetrievalScopeViolation,
    RetrievalService,
    authorized_paper_scope,
)

logger = logging.getLogger(__name__)

# Tokens are replayed in slices so the client renders progressively. Purely
# cosmetic — the text is already final and verified.
STREAM_CHUNK_CHARS = 24


class TurnFailed(Exception):
    """Carries a typed code for `turns.error_code` and the SSE error event."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class TurnResult:
    turn_id: uuid.UUID
    text: str
    grounding_status: str
    latency_ms: int


class TurnPipeline:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._messages = MessageService(session)

    async def _scope_for(
        self, conversation: Session, user_id: uuid.UUID
    ) -> tuple[list[uuid.UUID], Paper | None]:
        """Step 5 ⛨ — scope comes from session state, re-verified against grants.

        Session state names the paper; `user_paper_access` decides whether it
        may be read. A revoked grant empties the scope here, at read time.
        """
        if conversation.active_paper_id is None:
            return [], None

        scope = await authorized_paper_scope(
            self._session, user_id, [conversation.active_paper_id]
        )
        paper = (
            await self._session.get(Paper, conversation.active_paper_id)
            if scope
            else None
        )
        return scope, paper

    async def _next_ordinal(self, session_id: uuid.UUID) -> int:
        highest = await self._session.scalar(
            select(func.max(Turn.ordinal)).where(Turn.session_id == session_id)
        )
        return 0 if highest is None else highest + 1

    async def run(
        self, conversation: Session, user_id: uuid.UUID, user_message: str
    ) -> AsyncIterator[str]:
        """Run one turn, yielding encoded SSE frames."""
        started = time.perf_counter()

        yield StateEvent(phase="started", activity=conversation.activity).encode()

        try:
            scope, paper = await self._scope_for(conversation, user_id)

            context = TurnToolContext(
                session=self._session,
                user_id=user_id,
                paper_scope=scope,
                retrieval=RetrievalService(self._session),
            )

            # Step 6 — context assembly. History is the durable transcript,
            # not anything ADK remembers.
            history = await self._messages.history_for_context(
                conversation.session_id
            )

            yield StateEvent(
                phase="retrieving", activity=conversation.activity
            ).encode()

            # Steps 7-8 — the agent loop and composition.
            try:
                outcome = await run_turn(
                    context=context,
                    history=history,
                    user_message=user_message,
                    paper_title=paper.title if paper else None,
                    session_key=str(conversation.session_id),
                )
            except ToolScopeViolation as exc:
                raise TurnFailed("scope_violation", str(exc)) from exc
            except RetrievalScopeViolation as exc:
                raise TurnFailed("scope_violation", str(exc)) from exc
            except AgentUnavailable as exc:
                raise TurnFailed("agent_unavailable", str(exc)) from exc

            yield StateEvent(
                phase="verifying",
                activity=conversation.activity,
                tools_called=outcome.tools_called,
            ).encode()

            # Step 9 — the deterministic gate. Nothing streamed before this.
            verified = citation_verifier.verify(outcome.text, context.retrieved)
            if not verified.text:
                raise TurnFailed("empty_response", "The agent produced no answer.")

            latency_ms = int((time.perf_counter() - started) * 1000)

            # Step 11 — persist, in one transaction.
            turn = await self._persist(
                conversation=conversation,
                user_id=user_id,
                paper_id=paper.paper_id if paper else None,
                user_message=user_message,
                verified=verified,
                context=context,
                outcome=outcome,
                latency_ms=latency_ms,
            )

            # Step 12 — stream the verified answer.
            for index in range(0, len(verified.text), STREAM_CHUNK_CHARS):
                yield TokenEvent(
                    text=verified.text[index : index + STREAM_CHUNK_CHARS]
                ).encode()

            yield CitationsEvent(
                citations=[
                    CitationPayload(
                        marker=citation.marker,
                        chunk_id=str(citation.chunk.chunk_id),
                        paper_id=str(citation.chunk.paper_id),
                        section_path=citation.chunk.section_path,
                        page_start=citation.chunk.page_start,
                        page_end=citation.chunk.page_end,
                        similarity=citation.chunk.similarity,
                    )
                    for citation in verified.citations
                ]
            ).encode()

            # Empty until `retrieve_learner_memory` exists. Emitted anyway so
            # the client has one shape to render, not two.
            yield MemoryUsedEvent(memory=[]).encode()

            yield StateEvent(
                phase="persisted",
                activity=conversation.activity,
                tools_called=outcome.tools_called,
            ).encode()

            yield DoneEvent(
                turn_id=str(turn.turn_id),
                grounding_status=verified.grounding_status,
                latency_ms=latency_ms,
            ).encode()

        except TurnFailed as exc:
            await self._session.rollback()
            logger.warning("turn failed", extra={"code": exc.code})
            yield ErrorEvent(code=exc.code, message=str(exc)).encode()
        except Exception as exc:
            await self._session.rollback()
            logger.exception("turn crashed")
            yield ErrorEvent(
                code="internal_error", message="The turn could not be completed."
            ).encode()
            raise exc from None

    async def _persist(
        self,
        *,
        conversation: Session,
        user_id: uuid.UUID,
        paper_id: uuid.UUID | None,
        user_message: str,
        verified: citation_verifier.VerificationResult,
        context: TurnToolContext,
        outcome,
        latency_ms: int,
    ) -> Turn:
        """Step 11 — the turn, its retrieval set, and the two messages."""
        turn = Turn(
            session_id=conversation.session_id,
            user_id=user_id,
            paper_id=paper_id,
            ordinal=await self._next_ordinal(conversation.session_id),
            agent_action="answer",
            memory_read=False,
            grounding_status=verified.grounding_status,
            tools_called=outcome.tools_called or None,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            latency_ms=latency_ms,
        )
        self._session.add(turn)
        await self._session.flush()

        cited = verified.cited_chunk_ids
        marker_by_chunk = {
            citation.chunk.chunk_id: citation.marker for citation in verified.citations
        }
        for rank, chunk in enumerate(context.retrieved, start=1):
            was_cited = chunk.chunk_id in cited
            self._session.add(
                TurnRetrieval(
                    turn_id=turn.turn_id,
                    chunk_id=chunk.chunk_id,
                    rank=rank,
                    similarity=chunk.similarity,
                    retrieval_query=(context.queries[0] if context.queries else None),
                    was_cited=was_cited,
                    # The CHECK constraint requires these to agree.
                    citation_marker=marker_by_chunk.get(chunk.chunk_id)
                    if was_cited
                    else None,
                )
            )

        await self._messages.append_exchange(
            session_id=conversation.session_id,
            user_id=user_id,
            user_content=user_message,
            assistant_content=verified.text,
            turn_id=turn.turn_id,
        )

        conversation.turn_count = turn.ordinal + 1
        conversation.last_activity_at = func.now()

        await self._session.commit()
        return turn
