"""The cross-paper callback gate (ARCHITECTURE 12, and 9.2 step 10).

The demo's decisive moment, and the one most likely to be probed: a reader asks
about a concept in the paper they have open, and the system connects it to
something they struggled with in a *different* paper, weeks ago — with a
clickable citation into that earlier paper.

Every step of the decision is `[D]`. The model is told a connection is
available and what style has worked; it is never asked whether to make one,
because "should I bring up their old struggle" is a rate-limiting and
authorization question, not a language one.

Two properties this file exists to keep:

**Memory pointing at a paper is not authorization to read it.** A concept
remembers `source_paper_ids` forever, and a grant can be revoked at any time.
Step 7 re-verifies the grant against `user_paper_access` before the scope
widens, every single turn. This is the checkpoint ARCHITECTURE 12 says to
scrutinise, and it is the reason scope widens to exactly two papers rather than
"search all my papers".

**Suppression is a feature and is measured.** Every path out of this gate
returns a reason, and the pipeline writes it to `turns.callback_suppressed_reason`.
A callback that silently did not happen is indistinguishable from a gate that
is broken, which is why there is no silent path.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Concept, Paper, Turn, User
from app.services.learner_state import (
    CALLBACK_MIN_TURN_GAP,
    decay_factor,
    is_callback_candidate,
)
from app.services.memory import MemoryRecord
from app.services.retrieval import authorized_paper_scope

logger = logging.getLogger(__name__)

# Reasons fit `callback_suppressed_reason VARCHAR(64)`. Short slugs rather than
# sentences: these are counted, not read.
SUPPRESSED_PERSONALIZATION_OFF = "personalization_off"
SUPPRESSED_PROACTIVITY_OFF = "proactivity_off"
SUPPRESSED_NO_MEMORY = "no_memory"
SUPPRESSED_NO_CANDIDATE = "no_weak_candidate"
SUPPRESSED_NO_PRIOR_PAPER = "no_prior_paper"
SUPPRESSED_RATE_LIMITED = "rate_limited"
SUPPRESSED_GRANT_REVOKED = "grant_revoked"

# `users.preferences.proactivity` (ARCHITECTURE 4.1 — "proactivity tolerance").
# A multiplier on the turn gap rather than a separate gap per level, so there
# is one number to reason about and the levels stay ordered by construction.
PROACTIVITY_GAP_MULTIPLIER: dict[str, float | None] = {
    "off": None,  # never
    "low": 2.0,
    "normal": 1.0,
    "high": 0.5,
}
DEFAULT_PROACTIVITY = "normal"


@dataclass(slots=True)
class CallbackDecision:
    """What the gate decided, and why. Never silent either way."""

    concept_id: uuid.UUID | None = None
    concept_name: str | None = None
    relationship_type: str | None = None
    prior_paper_id: uuid.UUID | None = None
    prior_paper_title: str | None = None
    effective_style: str | None = None
    understanding_score: float | None = None
    suppressed_reason: str | None = None

    @property
    def fired(self) -> bool:
        return self.concept_id is not None

    def hint(self) -> str | None:
        """The one paragraph the agent is given about this.

        Deliberately an instruction about *how*, not a script. There is no
        canned sentence anywhere in this system — the same machinery runs for
        any user, any paper pair, any concept.
        """
        if not self.fired:
            return None

        style = (
            f" Lead with a {self.effective_style} explanation; that is what "
            f"worked for them last time."
            if self.effective_style
            else ""
        )
        return (
            f"This reader worked on {self.concept_name!r} before, in "
            f"{self.prior_paper_title or 'another paper'}, and found it hard. "
            f"It is {self.relationship_type or 'related'} to what they are "
            f"asking about now. Both papers are in scope for "
            f"`retrieve_paper_context`: search the earlier one too, connect the "
            f"two explicitly, and cite the earlier paper for what you say about "
            f"it.{style} Do not announce that you remember — just make the "
            f"connection."
        )


class CallbackService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def decide(
        self,
        *,
        user: User,
        active_paper_id: uuid.UUID | None,
        prefetched: list[MemoryRecord],
    ) -> CallbackDecision:
        """Steps 4-7: weakness filter, ranking, rate limit, scope re-check."""
        preferences = user.preferences or {}

        if preferences.get("personalization_enabled", True) is False:
            return CallbackDecision(suppressed_reason=SUPPRESSED_PERSONALIZATION_OFF)

        proactivity = preferences.get("proactivity", DEFAULT_PROACTIVITY)
        multiplier = PROACTIVITY_GAP_MULTIPLIER.get(proactivity, 1.0)
        if multiplier is None:
            return CallbackDecision(suppressed_reason=SUPPRESSED_PROACTIVITY_OFF)

        if not prefetched:
            return CallbackDecision(suppressed_reason=SUPPRESSED_NO_MEMORY)

        # Step 4-5 — the weakness filter and ranking, over the 1-hop
        # neighbourhood of whatever the reader is actually asking about.
        candidates = await self._rank_candidates(user.user_id, prefetched)
        if not candidates:
            return CallbackDecision(suppressed_reason=SUPPRESSED_NO_CANDIDATE)

        # Step 6 — rate limit. Checked before the authorization work below
        # because it is the cheaper question and rejects more often.
        minimum_gap = max(1, round(CALLBACK_MIN_TURN_GAP * multiplier))
        if not await self._gap_satisfied(user.user_id, minimum_gap):
            return CallbackDecision(suppressed_reason=SUPPRESSED_RATE_LIMITED)

        # Step 7 ⛨ — scope expansion. The critical checkpoint.
        for concept, relationship_type, score in candidates:
            prior = await self._authorized_prior_paper(
                user.user_id, concept, active_paper_id
            )
            if prior is None:
                continue

            paper_id, title = prior
            logger.info(
                "cross-paper callback permitted",
                extra={
                    "concept_id": str(concept.concept_id),
                    "prior_paper_id": str(paper_id),
                    "relationship": relationship_type,
                },
            )
            return CallbackDecision(
                concept_id=concept.concept_id,
                concept_name=concept.canonical_name,
                relationship_type=relationship_type,
                prior_paper_id=paper_id,
                prior_paper_title=title,
                effective_style=concept.effective_style,
                understanding_score=score,
            )

        # Every candidate was weak, confident and connected — and none of them
        # resolved to a paper this reader may still read. That is the grant
        # doing its job, and it is recorded rather than passed over.
        return CallbackDecision(suppressed_reason=SUPPRESSED_GRANT_REVOKED)

    async def _rank_candidates(
        self, user_id: uuid.UUID, prefetched: list[MemoryRecord]
    ) -> list[tuple[Concept, str, float]]:
        """Weak-and-confident neighbours of the concepts in play, best first."""
        # Only the strongest hit counts as "what they asked about"; telling
        # someone they once struggled with the very thing they just asked about
        # is noise, not memory.
        #
        # Excluding the *whole* prefetch instead would be wrong, and quietly
        # so: a concept related enough to be worth calling back to is usually
        # similar enough to the question that the ANN returns it too, so the
        # broad rule suppresses exactly the callbacks that should fire.
        primary = prefetched[0].concept_id

        # Best edge to each neighbour: a concept reachable by two relationships
        # is one candidate, ranked by its strongest connection.
        best_edge: dict[uuid.UUID, tuple[str, float]] = {}
        for record in prefetched:
            for related in record.related:
                if related.concept_id == primary:
                    continue
                existing = best_edge.get(related.concept_id)
                if existing is None or related.confidence > existing[1]:
                    best_edge[related.concept_id] = (
                        related.relationship_type,
                        related.confidence,
                    )

        if not best_edge:
            return []

        concepts = (
            await self._session.scalars(
                select(Concept).where(
                    Concept.user_id == user_id,
                    Concept.concept_id.in_(list(best_edge)),
                    Concept.merged_into_id.is_(None),
                )
            )
        ).all()

        now = datetime.now(UTC)
        ranked: list[tuple[float, uuid.UUID, Concept, str, float]] = []
        for concept in concepts:
            score = concept.user_override_score
            if score is None and concept.understanding_score is not None:
                score = concept.understanding_score * decay_factor(
                    concept.last_reinforced_at, now
                )

            if not is_callback_candidate(score, concept.score_confidence):
                continue

            relationship_type, confidence = best_edge[concept.concept_id]
            # Confident edge to a weak concept ranks highest. The concept_id
            # breaks ties so the same evidence always picks the same candidate
            # — a callback that varied run to run would be untestable.
            rank = confidence * (1.0 - (score or 0.0))
            ranked.append((rank, concept.concept_id, concept, relationship_type, score))

        ranked.sort(key=lambda row: (-row[0], row[1].bytes))
        return [(concept, kind, score) for _, _, concept, kind, score in ranked]

    async def _gap_satisfied(self, user_id: uuid.UUID, minimum_gap: int) -> bool:
        """Enough turns since the last callback, anywhere for this reader.

        Scoped to the user rather than the session on purpose: someone who
        opens a new conversation has not forgotten being told about a concept
        thirty seconds ago.
        """
        last_callback_at = await self._session.scalar(
            select(func.max(Turn.created_at)).where(
                Turn.user_id == user_id,
                Turn.callback_concept_id.isnot(None),
            )
        )
        if last_callback_at is None:
            return True

        turns_since = await self._session.scalar(
            select(func.count())
            .select_from(Turn)
            .where(Turn.user_id == user_id, Turn.created_at > last_callback_at)
        )
        return (turns_since or 0) >= minimum_gap

    async def _authorized_prior_paper(
        self,
        user_id: uuid.UUID,
        concept: Concept,
        active_paper_id: uuid.UUID | None,
    ) -> tuple[uuid.UUID, str | None] | None:
        """Step 7 ⛨ — a *different* paper, and one the grant still allows.

        Returns None when the concept has no other source paper, or when the
        reader may no longer read the one it names.
        """
        sources = [
            paper_id
            for paper_id in (concept.source_paper_ids or [])
            if paper_id != active_paper_id
        ]
        if not sources:
            return None

        authorized = await authorized_paper_scope(self._session, user_id, sources)
        if not authorized:
            return None

        paper_id = authorized[0]
        title = await self._session.scalar(
            select(Paper.title).where(Paper.paper_id == paper_id)
        )
        return paper_id, title
