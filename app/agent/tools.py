"""The agent's tools, and the boundary they cannot reach across.

`retrieve_paper_context` (ARCHITECTURE 14.2) is the only tool in this slice.
Its contract splits cleanly:

    the agent decides   what to search for, which section role, how many
    the backend decides whose data, which papers, the relevance floor, dedup,
                        and the post-retrieval assertion

That split is enforced structurally rather than by instruction. The tool is a
closure over a per-turn context, so `user_id` and `paper_scope` are **not
parameters** — there is no argument through which the model could name another
reader's papers, successful prompt injection or not (ARCHITECTURE 13.1).

`before_tool_callback` is the audited checkpoint on top of that: it records
every call and refuses any attempt to smuggle scope in through arguments.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SECTION_ROLE, Concept, Session, User
from app.services.memory import MemoryRecord, MemoryService
from app.services.quizzes import QuizService, QuizUnavailable
from app.services.retrieval import RetrievalService, RetrievedChunk
from app.services.signals import PendingSignal, SignalRejected, SignalService
from app.services.timing import TurnTimings

logger = logging.getLogger(__name__)

# Arguments a tool may never accept from the model. Present as a tripwire: if
# one ever appears, the tool signature has drifted into something unsafe.
FORBIDDEN_TOOL_ARGS = frozenset(
    {"user_id", "paper_id", "paper_scope", "session_id", "turn_id", "sql", "path"}
)

MAX_QUERY_CHARS = 500
MAX_TOP_K = 10
# Matches `concepts.canonical_name` (ARCHITECTURE 4.9): a name the model
# invents cannot be longer than one the schema can hold.
MAX_CONCEPT_NAME_CHARS = 200
MAX_CONCEPT_DEPTH = 2


class ToolScopeViolation(Exception):
    """The model tried to supply scope. Fails the turn closed."""


@dataclass
class TurnToolContext:
    """Everything the tools need, none of it reachable by the model."""

    session: AsyncSession
    user_id: uuid.UUID
    paper_scope: list[uuid.UUID]
    retrieval: RetrievalService
    memory: MemoryService | None = None
    signals: SignalService | None = None
    quizzes: QuizService | None = None
    conversation: Session | None = None
    # Set by the pipeline so tool time is attributed inside the agent loop
    # rather than disappearing into it.
    timings: TurnTimings | None = None
    session_id: uuid.UUID | None = None
    # The turn row does not exist until step 11, so observations written during
    # the loop carry a turn id the pipeline backfills. Null provenance is
    # permitted by the schema and is better than a wrong id.
    turn_id: uuid.UUID | None = None
    # Filled as the agent works; the pipeline reads these afterwards.
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    # Signals the agent asked for, validated and priced but not yet written.
    # Empty means the backstop (ARCHITECTURE 14.2) has to fire: the agent
    # forgetting must not silently stop memory accumulating.
    pending_signals: list[PendingSignal] = field(default_factory=list)
    # Set when `generate_quiz` actually put a question to the reader, so
    # the pipeline can record the activity transition it caused.
    quiz_asked: uuid.UUID | None = None
    # Memory the turn actually used, whether from the unconditional prefetch
    # or a tool lookup. `turns.memory_read` and the `memory_used` SSE event are
    # both derived from this, so neither can claim a read that did not happen.
    memory_seen: list[MemoryRecord] = field(default_factory=list)

    @property
    def memory_read(self) -> bool:
        return bool(self.memory_seen)

    def remember(self, records: Sequence[MemoryRecord]) -> None:
        """Record what memory returned, without duplicating concepts."""
        known = {record.concept_id for record in self.memory_seen}
        for record in records:
            if record.concept_id not in known:
                known.add(record.concept_id)
                self.memory_seen.append(record)

    def marker_for(self, chunk_id: uuid.UUID) -> str | None:
        for index, chunk in enumerate(self.retrieved, start=1):
            if chunk.chunk_id == chunk_id:
                return f"[{index}]"
        return None


def build_retrieve_paper_context(context: TurnToolContext):
    """Create the tool bound to one turn's authorization scope."""

    async def retrieve_paper_context(
        query: str,
        section_role: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Search the reader's open paper for passages relevant to a query.

        Args:
            query: What to look for, in natural language. 1-500 characters.
            section_role: Optionally restrict to one part of the paper, such as
                'method' or 'results'. Omit to search the whole paper.
            top_k: How many passages to return, 1-10.

        Returns:
            Passages, each with the citation marker to use when referring to it.
        """
        context.tools_called.append("retrieve_paper_context")
        began = time.perf_counter()

        cleaned = (query or "").strip()[:MAX_QUERY_CHARS]
        if not cleaned:
            return {"passages": [], "note": "Empty query; nothing was searched."}

        role = section_role if section_role in SECTION_ROLE else None
        limit = max(1, min(int(top_k or 5), MAX_TOP_K))

        results = await context.retrieval.retrieve(
            cleaned,
            paper_scope=context.paper_scope,
            top_k=limit,
            section_role=role,
        )
        context.queries.append(cleaned)

        # Markers are positional over everything retrieved this turn, so a
        # second search continues the numbering rather than restarting it and
        # silently repointing [1] at different text.
        passages = []
        for chunk in results:
            if not any(seen.chunk_id == chunk.chunk_id for seen in context.retrieved):
                context.retrieved.append(chunk)
            passages.append(
                {
                    "marker": context.marker_for(chunk.chunk_id),
                    "text": chunk.content,
                    "section": chunk.section_path,
                    "section_role": chunk.section_role,
                    "pages": chunk.citation_locator,
                }
            )

        if context.timings is not None:
            elapsed = (time.perf_counter() - began) * 1000
            embed_ms = getattr(context.retrieval, "last_embed_ms", 0.0)
            context.timings.record("tool:retrieve:embed", embed_ms)
            context.timings.record("tool:retrieve:ann+sql", elapsed - embed_ms)

        if not passages:
            return {
                "passages": [],
                "note": (
                    "Nothing in this paper matched. Tell the reader the paper "
                    "does not appear to cover this rather than answering from "
                    "general knowledge."
                ),
            }
        return {"passages": passages}

    return retrieve_paper_context


def build_retrieve_learner_memory(context: TurnToolContext):
    """What this reader has struggled with before (ARCHITECTURE 14.2).

    User-scoped by construction: `user_id` is closed over, exactly as in
    `retrieve_paper_context`, so there is no cross-user path to reach for.

    The backend also prefetches memory unconditionally on conceptual turns.
    This tool is for the agent's *follow-up* lookups — "what else do they know
    that touches this?" — and the prefetch is what guarantees memory is never
    silently skipped.
    """

    async def retrieve_learner_memory(
        query: str | None = None,
        concept_name: str | None = None,
        include_related: bool = True,
        only_weak: bool = False,
    ) -> dict[str, Any]:
        """Look up what this reader already knows, and how well.

        Args:
            query: A topic to search their learned concepts for.
            concept_name: One specific concept, if you already know its name.
            include_related: Include neighbouring concepts from the graph.
            only_weak: Return only concepts they are known to find difficult.

        Returns:
            Compact records: a score, a confidence, the explanation style that
            has worked before, and related concepts. Never past conversation.
        """
        context.tools_called.append("retrieve_learner_memory")
        if context.memory is None:
            return {"concepts": [], "note": "Learner memory is unavailable."}

        records = await context.memory.lookup(
            context.user_id,
            query=(query or "").strip()[:MAX_QUERY_CHARS] or None,
            concept_name=(concept_name or "").strip()[:MAX_CONCEPT_NAME_CHARS] or None,
            include_related=bool(include_related),
            only_weak=bool(only_weak),
        )
        context.remember(records)

        if not records:
            return {
                "concepts": [],
                "note": (
                    "Nothing recorded for this reader yet. Do not imply you "
                    "remember them; this is the first time you are meeting "
                    "this material with them."
                ),
            }
        return {"concepts": [record.for_model() for record in records]}

    return retrieve_learner_memory


def build_get_concept_context(context: TurnToolContext):
    """One concept in full: evidence, provenance and neighbours.

    `retrieve_learner_memory` answers "what do they know"; this answers "how do
    we know that". Comparisons and callbacks need the evidence behind a score,
    not just the number — it is what lets the agent say "this took a couple of
    passes last time" and have it be true.
    """

    async def get_concept_context(
        concept_name: str,
        depth: int = 1,
    ) -> dict[str, Any]:
        """Get the full picture of one concept for this reader.

        Args:
            concept_name: The concept to look up.
            depth: How far to walk the concept graph, 1 or 2.

        Returns:
            The concept's understanding, the papers it came from, its typed
            relationships, and the evidence behind its score.
        """
        context.tools_called.append("get_concept_context")
        if context.memory is None:
            return {"concept": None, "note": "Learner memory is unavailable."}

        cleaned = (concept_name or "").strip()[:MAX_CONCEPT_NAME_CHARS]
        if not cleaned:
            return {"concept": None, "note": "No concept named."}

        concept = await context.memory.by_name(context.user_id, cleaned)
        if concept is None:
            return {
                "concept": None,
                "note": f"Nothing recorded for {cleaned!r} for this reader.",
            }

        depth = max(1, min(int(depth or 1), MAX_CONCEPT_DEPTH))
        records = await context.memory.lookup(
            context.user_id, concept_name=cleaned, include_related=False
        )
        if not records:
            return {"concept": None, "note": f"Nothing recorded for {cleaned!r}."}

        record = records[0]
        edges = await context.memory.neighbours(
            context.user_id, [record.concept_id], depth=depth
        )
        record.related = edges.get(record.concept_id, [])
        context.remember([record])

        # ARCHITECTURE 12 step 7: memory pointing at a paper is not
        # authorization to read it. The grant is re-checked here.
        papers = await context.memory.visible_source_papers(
            context.user_id, record.source_paper_ids
        )
        observations = await context.memory.evidence_for(record.concept_id)

        payload = record.for_model()
        payload["source_papers"] = [
            {"title": title or "untitled"} for _, title in papers
        ]
        payload["evidence"] = [
            {
                "signal": observation.signal_type,
                "style_in_play": observation.style_in_play,
                "note": observation.note,
                "observed_at": observation.observed_at.date().isoformat()
                if observation.observed_at
                else None,
            }
            for observation in observations
        ]
        return {"concept": payload}

    return get_concept_context


def build_record_learning_signal(context: TurnToolContext):
    """The only model-reachable write into learner memory.

    `user_id`, `session_id` and `turn_id` are all injected — the model cannot
    name a different reader, conversation or turn. It cannot set a score
    either: it reports what happened, and the backend decides what that is
    worth (ARCHITECTURE 14.2).

    A malformed signal is rejected and logged, and the turn continues. The
    reader asked a question and is owed an answer regardless of what the agent
    got wrong about bookkeeping.
    """

    async def record_learning_signal(
        concept_name: str,
        signal_type: str,
        style_in_play: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Record something you noticed about how this reader is doing.

        Call this when the reader shows confusion, says something clicked, or
        uses a concept correctly — not for every message.

        Args:
            concept_name: The concept the signal is about.
            signal_type: One of explicit_confusion, implicit_confusion,
                explicit_understanding, applied_correctly, user_stated_known,
                user_stated_unknown.
            style_in_play: The explanation style being used, if relevant: one
                of formal, intuitive, numerical, analogical, visual_verbal,
                code, contrastive.
            note: A short human-readable line of evidence, under 500 chars.

        Returns:
            Whether it was recorded, and the concept's updated standing.
        """
        context.tools_called.append("record_learning_signal")
        if context.signals is None:
            return {"recorded": False, "note": "Learner memory is unavailable."}

        try:
            pending = await context.signals.prepare(
                user_id=context.user_id,
                concept_name=concept_name,
                signal_type=signal_type,
                paper_id=context.paper_scope[0] if context.paper_scope else None,
                style_in_play=style_in_play,
                note=note,
            )
        except SignalRejected as exc:
            logger.warning("learning signal rejected: %s", exc)
            return {"recorded": False, "note": str(exc)}

        # Buffered, not written: the turn row it belongs to does not exist
        # yet, and `observations` is append-only, so the provenance link could
        # never be added afterwards. The pipeline commits these at step 11.
        context.pending_signals.append(pending)

        score = pending.projected.raw_score
        return {
            "recorded": True,
            "concept": pending.concept_name,
            "understanding_score": None if score is None else round(score, 2),
            "score_confidence": round(pending.projected.confidence, 2),
            "resolved_earlier_struggle": pending.resolves_observation_id is not None,
        }

    return record_learning_signal


def build_generate_quiz(context: TurnToolContext):
    """Ask the reader a grounded comprehension question (ARCHITECTURE 14.2).

    The backend decides whether a check is *allowed* — frequency, appetite, and
    which passages ground it. The agent decides whether one is *useful* now.

    **The rubric is never returned.** If the agent held it, it could leak the
    expected answer into the question it asks, and the check would measure
    nothing.
    """

    async def generate_quiz(
        concept_name: str,
        difficulty: str = "medium",
    ) -> dict[str, Any]:
        """Check the reader's understanding with one grounded question.

        Only useful after you have explained something and want to know whether
        it landed. Do not use it to open a conversation.

        Args:
            concept_name: The concept to check.
            difficulty: One of easy, medium, hard.

        Returns:
            The question to put to the reader, or why a check is not allowed
            right now.
        """
        context.tools_called.append("generate_quiz")
        if context.quizzes is None or context.conversation is None:
            return {"asked": False, "note": "Checks are unavailable in this session."}

        cleaned = (concept_name or "").strip()[:MAX_CONCEPT_NAME_CHARS]
        if not cleaned:
            return {"asked": False, "note": "No concept named."}

        user = await context.session.get(User, context.user_id)
        concept = (
            await context.memory.by_name(context.user_id, cleaned)
            if context.memory
            else None
        )

        decision = await context.quizzes.gate(
            user=user, conversation=context.conversation, concept=concept
        )
        if not decision.allowed:
            # A refusal is not a failure: the agent is told plainly so it can
            # carry on explaining rather than pretending it asked something.
            return {"asked": False, "note": f"A check is not allowed now ({decision.reason})."}

        if concept is None:
            if context.signals is None:
                return {"asked": False, "note": "That concept is not on record."}
            # First time this concept has come up: create it the same way any
            # other conversational concept is created, then check it.
            pending = await context.signals.prepare(
                user_id=context.user_id,
                concept_name=cleaned,
                signal_type="reinforcement",
                paper_id=context.paper_scope[0] if context.paper_scope else None,
            )
            concept = await context.session.get(Concept, pending.concept_id)

        if not context.paper_scope:
            return {"asked": False, "note": "No paper is open to ground a question in."}

        passages = [(chunk.chunk_id, chunk.content) for chunk in context.retrieved]
        try:
            quiz = await context.quizzes.create(
                user_id=context.user_id,
                conversation=context.conversation,
                concept=concept,
                paper_id=context.paper_scope[0],
                passages=passages,
                difficulty=difficulty,
            )
        except QuizUnavailable as exc:
            logger.warning("quiz not created: %s", exc)
            return {
                "asked": False,
                "note": (
                    "A check needs passages from this paper first — search it, "
                    "then try again."
                ),
            }

        context.quiz_asked = quiz.quiz_id
        # The question, and nothing else. No rubric, no expected answer.
        return {"asked": True, "question": quiz.question}

    return generate_quiz


def build_scope_guard(context: TurnToolContext):
    """The `before_tool_callback` audit checkpoint (ARCHITECTURE 13.1).

    Scope is already unreachable — it is closed over, not passed — so this
    exists to record what the agent chose and to fail loudly if a tool ever
    grows an argument that could carry authorization.
    """

    def before_tool(tool, args: dict[str, Any], tool_context) -> dict | None:
        smuggled = FORBIDDEN_TOOL_ARGS.intersection(args or {})
        if smuggled:
            logger.error(
                "SECURITY: model supplied scope-bearing tool arguments",
                extra={"tool": getattr(tool, "name", "?"), "args": sorted(smuggled)},
            )
            raise ToolScopeViolation(
                f"{sorted(smuggled)} may not be supplied by the model"
            )

        logger.info(
            "tool call",
            extra={
                "tool": getattr(tool, "name", "?"),
                "user_id": str(context.user_id),
                "papers_in_scope": len(context.paper_scope),
            },
        )
        # None means "proceed with the real tool".
        return None

    return before_tool
