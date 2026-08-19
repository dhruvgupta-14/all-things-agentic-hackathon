"""TurnPipeline — the deterministic wrapper around the agent (ARCHITECTURE 9.2).

The agent decides what to search for and how to phrase the answer. Everything
that has to be *correct* rather than plausible happens here: identity, session
ownership, retrieval scope, citation verification, and every database write.

Steps implemented in this slice, numbered as in §9.2:

    1-2  identity and session ownership          (the router's dependencies)
    3    deterministic route on QUIZ_PENDING     — no classification call
    4    memory prefetch, unconditional          ⛨
    5    build scope from session state          ⛨
    6    assemble context from `messages`
    7    agent loop, tools receive injected scope
    8    compose
    9    verify citations
    10   callback gate, with suppression recorded
    11   persist: turn + turn_retrievals + messages + signals, one transaction
    12   stream
    13   learning signals, with the deterministic backstop

`turns.memory_read` is derived from what the prefetch and the memory tools
actually returned, never asserted — which is what the CHECK constraint on
`callback_concept_id` relies on.

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
from app.db.models import Paper, Session, Turn, TurnRetrieval, User
from app.schemas.sse import (
    CitationPayload,
    CitationsEvent,
    DoneEvent,
    ErrorEvent,
    MemoryRecordPayload,
    MemoryUsedEvent,
    StateEvent,
    TokenEvent,
)
from app.services import citations as citation_verifier
from app.services.callbacks import CallbackDecision, CallbackService
from app.services.feedback import FeedbackService, depth_instruction
from app.services.memory import MemoryRecord, MemoryService
from app.services.messages import MessageService
from app.services.quizzes import (
    NEXT_REVISIT_PREREQUISITE,
    GradingFailed,
    QuizService,
)
from app.services.retrieval import (
    RetrievalScopeViolation,
    RetrievalService,
    authorized_paper_scope,
)
from app.services.signals import SignalRejected, SignalService
from app.services.timing import TurnTimings

logger = logging.getLogger(__name__)

# Tokens are replayed in slices so the client renders progressively. Purely
# cosmetic — the text is already final and verified.
STREAM_CHUNK_CHARS = 24


def _memory_summary(records: list[MemoryRecord]) -> str | None:
    """Prefetched memory, as compact lines for the instruction.

    Scores and a style, never conversation text — the agent is given a claim
    with provenance behind it, not a transcript to paraphrase.
    """
    if not records:
        return None

    lines = []
    for record in records:
        score = (
            "no score yet"
            if record.understanding_score is None
            else f"understanding {record.understanding_score:.2f}"
        )
        confidence = (
            ""
            if record.score_confidence is None
            else f", confidence {record.score_confidence:.2f}"
        )
        style = (
            f", explaining it {record.effective_style}ly has worked before"
            if record.effective_style
            else ""
        )
        related = (
            "; related: "
            + ", ".join(
                f"{item.name} ({item.relationship_type})" for item in record.related[:3]
            )
            if record.related
            else ""
        )
        lines.append(f"- {record.canonical_name}: {score}{confidence}{style}{related}")

    return "\n".join(lines)


def _grading_response(result) -> str:
    """The reply to a graded answer — composed here, not by the model.

    The three-way next action is fully determined by the grade and the concept
    graph (ARCHITECTURE 11), so there is nothing here for a model to decide and
    a second call would only add latency and a chance to contradict the grade
    that was just recorded. What the reader gets is what was actually written
    to `quiz_attempts`.
    """
    if result.grade is None:
        return (
            "I could not grade that reliably, so I have not recorded a result "
            "— I would rather tell you than guess. Shall we keep going?"
        )

    missing = ""
    if result.missing_elements:
        missing = "\n\nStill missing: " + "; ".join(result.missing_elements) + "."

    if result.grade == "correct":
        return f"That is right.{missing}\n\nLet's move on."

    opening = (
        "Close — part of that is right."
        if result.grade == "partial"
        else "Not quite."
    )

    if result.next_action == NEXT_REVISIT_PREREQUISITE and result.prerequisite_name:
        return (
            f"{opening}{missing}\n\nBefore we go further, I think "
            f"{result.prerequisite_name} is what is getting in the way — it "
            f"underpins this and you have found it difficult before. Shall we "
            f"go back to it?"
        )

    return (
        f"{opening}{missing}\n\nLet me explain {result.concept_name} a different "
        f"way and we can try again."
    )


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
        timings = TurnTimings()

        yield StateEvent(phase="started", activity=conversation.activity).encode()

        try:
            # Step 3 — deterministic route. A pending quiz means the next
            # message is a quiz answer, full stop. Asking a model "is this a
            # quiz answer?" moments after asking the question would be a wasted
            # call and a source of nondeterminism at the most measured point in
            # the system.
            if conversation.activity == "QUIZ_PENDING":
                async for frame in self._grade_pending_quiz(
                    conversation, user_id, user_message, started
                ):
                    yield frame
                return

            with timings.span("scope"):
                scope, paper = await self._scope_for(conversation, user_id)

            context = TurnToolContext(
                session=self._session,
                    user_id=user_id,
                paper_scope=scope,
                retrieval=RetrievalService(self._session),
                memory=MemoryService(self._session),
                signals=SignalService(self._session),
                quizzes=QuizService(self._session),
                    conversation=conversation,
                session_id=conversation.session_id,
            )

            # Step 6 — context assembly. History is the durable transcript,
            # not anything ADK remembers.
            with timings.span("history"):
                history = await self._messages.history_for_context(
                    conversation.session_id
                )

            # Step 4 — memory prefetch, unconditional. An agent that *chooses*
            # whether to consult memory will sometimes not, and the
            # differentiator disappears on exactly the turns nobody is
            # watching. The tool remains, for follow-up lookups.
            yield StateEvent(
                phase="consulting_memory", activity=conversation.activity
            ).encode()

            with timings.span("memory_prefetch"):
                prefetched = await context.memory.prefetch(user_id, user_message)
                context.remember(prefetched)

            # Step 10 — the callback gate. Every path out of it records a
            # reason, so a callback that did not happen is never silent.
            with timings.span("callback_gate"):
                reader = await self._session.get(User, user_id)
                callback = await CallbackService(self._session).decide(
                    user=reader,
                    active_paper_id=paper.paper_id if paper else None,
                    prefetched=prefetched,
                )
            if callback.fired and callback.prior_paper_id is not None:
                # Scope widens to exactly two papers, and only after the grant
                # on the prior one was re-verified inside the gate.
                context.paper_scope = [*scope, callback.prior_paper_id]

            yield StateEvent(
                phase="retrieving", activity=conversation.activity
            ).encode()

            # Steps 7-8 — the agent loop and composition.
            context.timings = timings
            try:
                with timings.span("agent_loop"):
                    outcome = await run_turn(
                        context=context,
                        history=history,
                        user_message=user_message,
                        paper_title=paper.title if paper else None,
                        session_key=str(conversation.session_id),
                        memory_summary=_memory_summary(prefetched),
                        callback_hint=callback.hint(),
                        depth_hint=depth_instruction(
                            reader.preferences if reader else None
                        ),
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
            with timings.span("verify_citations"):
                verified = citation_verifier.verify(outcome.text, context.retrieved)
            if not verified.text:
                raise TurnFailed("empty_response", "The agent produced no answer.")

            latency_ms = int((time.perf_counter() - started) * 1000)

            # Step 11 — persist, in one transaction.
            with timings.span("persist"):
                turn = await self._persist(
                    conversation=conversation,
                    user_id=user_id,
                    paper_id=paper.paper_id if paper else None,
                    user_message=user_message,
                    verified=verified,
                    context=context,
                    outcome=outcome,
                    latency_ms=latency_ms,
                    callback=callback,
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

            # What memory actually informed this turn — the prefetch plus
            # anything the agent looked up itself. Derived from the same list
            # that sets `turns.memory_read`, so the event and the row cannot
            # disagree about whether memory was read.
            yield MemoryUsedEvent(
                memory=[
                    MemoryRecordPayload(
                        concept_id=str(record.concept_id),
                        name=record.canonical_name,
                        understanding_score=record.understanding_score,
                        score_confidence=record.score_confidence,
                        effective_style=record.effective_style,
                    )
                    for record in context.memory_seen
                ]
            ).encode()

            yield StateEvent(
                phase="persisted",
                activity=conversation.activity,
                tools_called=outcome.tools_called,
            ).encode()

            timings.log(str(turn.turn_id))

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

    async def _grade_pending_quiz(
        self,
        conversation: Session,
        user_id: uuid.UUID,
        answer_text: str,
        started: float,
    ) -> AsyncIterator[str]:
        """Steps 5-10 — grade, record, transition, respond.

        No agent loop and no retrieval: the question was already grounded when
        it was asked, and re-searching would only add latency and a chance to
        contradict the rubric.
        """
        quizzes = QuizService(self._session)
        signals = SignalService(self._session)

        yield StateEvent(phase="composing", activity=conversation.activity).encode()

        try:
            result = await quizzes.grade_pending(
                conversation=conversation, user_id=user_id, answer_text=answer_text
            )
        except GradingFailed as exc:
            await self._session.commit()  # the state transition still stands
            raise TurnFailed("grading_failed", str(exc)) from exc

        text = _grading_response(result)
        latency_ms = int((time.perf_counter() - started) * 1000)

        turn = Turn(
            session_id=conversation.session_id,
            user_id=user_id,
            paper_id=conversation.active_paper_id,
            ordinal=await self._next_ordinal(conversation.session_id),
            agent_action=f"quiz_{result.next_action}",
            memory_read=False,
            grounding_status="n/a",
            tools_called=["grade_quiz_answer"],
            latency_ms=latency_ms,
        )
        self._session.add(turn)
        await self._session.flush()

        # Step 8 — a graded attempt becomes a weighted learning signal, at the
        # highest weight class. A failed grading writes nothing: we did not
        # learn anything about the reader, only about the grader.
        if result.signal_type is not None:
            await signals.record(
                user_id=user_id,
                concept_name=result.concept_name,
                signal_type=result.signal_type,
                session_id=conversation.session_id,
                turn_id=turn.turn_id,
                paper_id=conversation.active_paper_id,
                note=f"Quiz answer graded {result.grade}.",
                quiz_attempt_id=result.attempt_id,
            )

        await self._messages.append_exchange(
            session_id=conversation.session_id,
            user_id=user_id,
            user_content=answer_text,
            assistant_content=text,
            turn_id=turn.turn_id,
        )
        conversation.turn_count = turn.ordinal + 1
        conversation.last_activity_at = func.now()
        await self._session.commit()

        for index in range(0, len(text), STREAM_CHUNK_CHARS):
            yield TokenEvent(text=text[index : index + STREAM_CHUNK_CHARS]).encode()

        # No citations: the grading turn makes no claim about the paper. The
        # empty event keeps the client rendering one shape rather than two.
        yield CitationsEvent(citations=[]).encode()
        yield MemoryUsedEvent(memory=[]).encode()
        yield StateEvent(
            phase="persisted", activity=conversation.activity
        ).encode()
        yield DoneEvent(
            turn_id=str(turn.turn_id),
            grounding_status="n/a",
            latency_ms=latency_ms,
        ).encode()

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
        callback: CallbackDecision,
    ) -> Turn:
        """Step 11 — the turn, its retrieval set, and the two messages."""
        turn = Turn(
            session_id=conversation.session_id,
            user_id=user_id,
            paper_id=paper_id,
            ordinal=await self._next_ordinal(conversation.session_id),
            agent_action="callback" if callback.fired else "answer",
            # The gate's decision, not a reading of the prose. That is what
            # makes the pair (`callback_concept_id`, `callback_suppressed_reason`)
            # a measurement: every turn records one or the other, always.
            callback_concept_id=callback.concept_id,
            callback_suppressed_reason=callback.suppressed_reason,
            # Set only when the backend actually told the agent which style to
            # use; otherwise nothing here knows what style the answer took.
            explanation_style=callback.effective_style if callback.fired else None,
            # Derived from what memory actually returned, never asserted. The
            # CHECK constraint on `callback_concept_id` leans on this being
            # true only when a read really happened.
            memory_read=context.memory_read,
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

        # Step 13 — the buffered signals, now that they have a turn to point at.
        for pending in context.pending_signals:
            await context.signals.commit(
                pending,
                user_id=user_id,
                session_id=conversation.session_id,
                turn_id=turn.turn_id,
            )

        # Feedback that moved a standing preference composed *this* turn,
        # so this is where it becomes verifiable rather than asserted.
        await FeedbackService(self._session).apply_pending(user_id, turn.turn_id)

        await self._backstop_signal(context, user_id, turn)

        conversation.turn_count = turn.ordinal + 1
        conversation.last_activity_at = func.now()

        await self._session.commit()
        return turn

    async def _backstop_signal(
        self, context: TurnToolContext, user_id: uuid.UUID, turn: Turn
    ) -> None:
        """Step 13's deterministic backstop (ARCHITECTURE 14.2).

        A conceptual turn where the agent recorded nothing still writes a
        `reinforcement` observation against the concept the turn was about.
        The agent forgetting must not silently stop memory accumulating.

        A backstop row carries zero evidentiary weight, so it cannot move a
        score or manufacture confidence — what it does is move
        `last_reinforced_at`, which resets the decay clock. That is the honest
        claim: this concept came up again, and nothing was observed about how
        well it landed.
        """
        if context.pending_signals or context.signals is None:
            return
        if not context.memory_seen:
            return

        # The concept the turn was actually about: the strongest memory hit,
        # which is the first the prefetch returned.
        concept = context.memory_seen[0]
        try:
            await context.signals.record(
                user_id=user_id,
                concept_name=concept.canonical_name,
                signal_type="reinforcement",
                session_id=turn.session_id,
                turn_id=turn.turn_id,
                paper_id=turn.paper_id,
                note="Revisited without an explicit signal.",
            )
        except SignalRejected as exc:
            # Never fail a turn over bookkeeping the reader did not ask for.
            logger.warning("backstop signal rejected: %s", exc)
