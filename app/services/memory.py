"""MemoryService — reading the learner model (ARCHITECTURE 9.2 step 4, 10, 12).

Everything here is `[D]`. The agent decides *whether* to look and whether to
say what came back; scope, ranking and record shape are decided here.

Two rules shape the return type and are worth stating plainly:

**Compact records, never transcripts.** A memory record is a handful of scalars
and a name. Feeding old conversation text back into the context window would
make memory a retrieval problem with no grounding guarantee behind it — the
opposite of the citation discipline the rest of the system runs on. What the
agent gets is a *claim with provenance*, not a quotation.

**User scope is structural.** Every query filters on `user_id` taken from the
verified principal. There is no argument, on any method here, through which a
caller could name a different reader.

Scores are decayed at read time (ARCHITECTURE 17) rather than read raw from the
cache, so a concept untouched for a month reads as stale here even though the
row has not been rewritten.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import Float, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Concept, ConceptRelationship, Observation, Paper, UserPaperAccess
from app.services.embeddings import Embedder, get_embedder
from app.services.learner_state import (
    decay_factor,
    is_callback_candidate,
)

logger = logging.getLogger(__name__)

# How many concepts an unconditional prefetch may pull. Small on purpose: this
# runs on every conceptual turn and lands in the context window, where it
# competes with the passages the answer is actually grounded in.
PREFETCH_LIMIT = 5

# Depth cap on graph traversal (ARCHITECTURE 14.2, `get_concept_context`).
MAX_DEPTH = 2

# A concept below the relevance floor is not "the thing being asked about".
# The floor is taken from the embedder rather than written down here: cosine
# scores are not comparable between models (HANDOFF 7.4), so a literal would be
# right for one vector space and meaningless in the other.


@dataclass(slots=True)
class RelatedConcept:
    concept_id: uuid.UUID
    name: str
    relationship_type: str
    confidence: float


@dataclass(slots=True)
class MemoryRecord:
    """One concept, as the agent is allowed to see it."""

    concept_id: uuid.UUID
    canonical_name: str
    understanding_score: float | None
    score_confidence: float | None
    effective_style: str | None
    last_reinforced_at: datetime | None
    evidence_count: int
    evidence_note: str | None = None
    related: list[RelatedConcept] = field(default_factory=list)
    source_paper_ids: list[uuid.UUID] = field(default_factory=list)

    @property
    def is_weak(self) -> bool:
        return is_callback_candidate(self.understanding_score, self.score_confidence)

    def for_model(self) -> dict:
        """The shape the agent sees. Scalars and names — no transcript."""
        return {
            "concept": self.canonical_name,
            "understanding_score": (
                None
                if self.understanding_score is None
                else round(self.understanding_score, 2)
            ),
            "score_confidence": (
                None
                if self.score_confidence is None
                else round(self.score_confidence, 2)
            ),
            "effective_style": self.effective_style,
            "last_seen": (
                self.last_reinforced_at.date().isoformat()
                if self.last_reinforced_at
                else None
            ),
            "evidence_count": self.evidence_count,
            "evidence_note": self.evidence_note,
            "related": [
                {
                    "concept": related.name,
                    "relationship": related.relationship_type,
                    "confidence": round(related.confidence, 2),
                }
                for related in self.related
            ],
        }


class MemoryService:
    def __init__(
        self, session: AsyncSession, *, embedder: Embedder | None = None
    ) -> None:
        self._session = session
        self._embedder = embedder or get_embedder()

    # -- record construction ------------------------------------------------

    def _to_record(self, concept: Concept, now: datetime) -> MemoryRecord:
        """Apply decay and the explicit override, in that order of authority.

        `user_override_score` is a correction the reader made about themselves.
        It outranks inference and is not decayed — decay models our uncertainty
        about inferred knowledge, and there is nothing inferred about it.
        """
        if concept.user_override_score is not None:
            score = concept.user_override_score
        elif concept.understanding_score is None:
            score = None
        else:
            score = concept.understanding_score * decay_factor(
                concept.last_reinforced_at, now
            )

        return MemoryRecord(
            concept_id=concept.concept_id,
            canonical_name=concept.canonical_name,
            understanding_score=score,
            score_confidence=concept.score_confidence,
            effective_style=concept.effective_style,
            last_reinforced_at=concept.last_reinforced_at,
            evidence_count=concept.evidence_count or 0,
            source_paper_ids=list(concept.source_paper_ids or []),
        )

    # -- recall -------------------------------------------------------------

    async def nearest(
        self, user_id: uuid.UUID, query: str, *, limit: int = PREFETCH_LIMIT
    ) -> list[Concept]:
        """ANN over this user's concepts. Empty query returns nothing."""
        if not query.strip():
            return []

        vector = self._embedder.embed_query(query)
        distance = Concept.embedding.cosine_distance(vector)
        similarity = (1 - distance).cast(Float).label("similarity")

        rows = (
            await self._session.execute(
                select(Concept, similarity)
                .where(
                    Concept.user_id == user_id,
                    Concept.merged_into_id.is_(None),
                    Concept.embedding.isnot(None),
                )
                .order_by(distance)
                .limit(limit)
            )
        ).all()

        floor = self._embedder.default_min_similarity
        return [concept for concept, score in rows if score >= floor]

    async def by_name(self, user_id: uuid.UUID, name: str) -> Concept | None:
        """Exact name or alias first, then the nearest vector match.

        The zero-cost path is the same one canonicalization uses, so the agent
        naming a concept the way the reader named it never costs an embedding.
        """
        cleaned = (name or "").strip()
        if not cleaned:
            return None

        from app.ingestion.concepts import normalize_name

        normalized = normalize_name(cleaned)
        exact = await self._session.scalar(
            select(Concept)
            .where(
                Concept.user_id == user_id,
                Concept.merged_into_id.is_(None),
                or_(
                    Concept.normalized_name == normalized,
                    Concept.aliases.overlap([cleaned]),
                ),
            )
            .limit(1)
        )
        if exact is not None:
            return exact

        nearest = await self.nearest(user_id, cleaned, limit=1)
        return nearest[0] if nearest else None

    async def neighbours(
        self,
        user_id: uuid.UUID,
        concept_ids: Sequence[uuid.UUID],
        *,
        depth: int = 1,
    ) -> dict[uuid.UUID, list[RelatedConcept]]:
        """1-hop (or 2-hop) traversal, both directions.

        Symmetric relationship types are stored once with a canonical
        orientation (ARCHITECTURE 4.10), so traversing only `source` would
        silently lose half the graph.
        """
        if not concept_ids:
            return {}

        depth = max(1, min(depth, MAX_DEPTH))
        frontier = list(concept_ids)
        seen: set[uuid.UUID] = set(concept_ids)
        edges: dict[uuid.UUID, list[RelatedConcept]] = {}

        for _ in range(depth):
            if not frontier:
                break

            rows = (
                await self._session.scalars(
                    select(ConceptRelationship).where(
                        ConceptRelationship.user_id == user_id,
                        or_(
                            ConceptRelationship.source_concept_id.in_(frontier),
                            ConceptRelationship.target_concept_id.in_(frontier),
                        ),
                    )
                )
            ).all()

            endpoints = {
                endpoint
                for edge in rows
                for endpoint in (edge.source_concept_id, edge.target_concept_id)
            }
            names = await self._names_for(user_id, endpoints)

            next_frontier: list[uuid.UUID] = []
            for edge in rows:
                for near, far in (
                    (edge.source_concept_id, edge.target_concept_id),
                    (edge.target_concept_id, edge.source_concept_id),
                ):
                    if near not in frontier or far not in names:
                        continue
                    edges.setdefault(near, []).append(
                        RelatedConcept(
                            concept_id=far,
                            name=names[far],
                            relationship_type=edge.relationship_type,
                            confidence=edge.confidence,
                        )
                    )
                    if far not in seen:
                        seen.add(far)
                        next_frontier.append(far)

            frontier = next_frontier

        return edges

    async def _names_for(
        self, user_id: uuid.UUID, concept_ids: set[uuid.UUID]
    ) -> dict[uuid.UUID, str]:
        if not concept_ids:
            return {}
        rows = (
            await self._session.execute(
                select(Concept.concept_id, Concept.canonical_name).where(
                    Concept.user_id == user_id,
                    Concept.concept_id.in_(list(concept_ids)),
                    Concept.merged_into_id.is_(None),
                )
            )
        ).all()
        return {concept_id: name for concept_id, name in rows}

    # -- the two entry points ----------------------------------------------

    async def prefetch(
        self, user_id: uuid.UUID, query: str, *, limit: int = PREFETCH_LIMIT
    ) -> list[MemoryRecord]:
        """Step 4 — unconditional on conceptual turns.

        Unconditional is the point: an agent that *chooses* whether to consult
        memory will sometimes not, and the differentiator disappears on exactly
        the turns nobody is watching. The tool exists for follow-up lookups;
        this guarantees memory is never silently skipped.
        """
        concepts = await self.nearest(user_id, query, limit=limit)
        if not concepts:
            return []

        now = datetime.now(UTC)
        records = [self._to_record(concept, now) for concept in concepts]

        edges = await self.neighbours(
            user_id, [record.concept_id for record in records], depth=1
        )
        for record in records:
            record.related = edges.get(record.concept_id, [])

        return records

    async def lookup(
        self,
        user_id: uuid.UUID,
        *,
        query: str | None = None,
        concept_name: str | None = None,
        include_related: bool = True,
        only_weak: bool = False,
        limit: int = PREFETCH_LIMIT,
    ) -> list[MemoryRecord]:
        """What `retrieve_learner_memory` calls (ARCHITECTURE 14.2)."""
        concepts: list[Concept] = []

        if concept_name:
            found = await self.by_name(user_id, concept_name)
            if found is not None:
                concepts = [found]
        elif query:
            concepts = await self.nearest(user_id, query, limit=limit)
        else:
            # Neither given: the weakest concepts we are confident about. The
            # index predicate carries the confidence floor.
            concepts = list(
                (
                    await self._session.scalars(
                        select(Concept)
                        .where(
                            Concept.user_id == user_id,
                            Concept.merged_into_id.is_(None),
                            Concept.score_confidence >= 0.3,
                        )
                        .order_by(Concept.understanding_score)
                        .limit(limit)
                    )
                ).all()
            )

        if not concepts:
            return []

        now = datetime.now(UTC)
        records = [self._to_record(concept, now) for concept in concepts]

        if only_weak:
            records = [record for record in records if record.is_weak]
            if not records:
                return []

        if include_related:
            edges = await self.neighbours(
                user_id, [record.concept_id for record in records], depth=1
            )
            for record in records:
                record.related = edges.get(record.concept_id, [])

        await self._attach_evidence_notes(records)
        return records

    async def _attach_evidence_notes(self, records: Sequence[MemoryRecord]) -> None:
        """The most recent human-readable evidence line for each concept.

        One note, not a history: this is what lets the agent say "this took a
        couple of passes last time" truthfully, without replaying the
        conversation it came from.
        """
        for record in records:
            note = await self._session.scalar(
                select(Observation.note)
                .where(
                    Observation.concept_id == record.concept_id,
                    Observation.note.isnot(None),
                )
                .order_by(Observation.observed_at.desc())
                .limit(1)
            )
            record.evidence_note = note

    async def evidence_for(
        self, concept_id: uuid.UUID, *, limit: int = 5
    ) -> list[Observation]:
        """The evidence trail behind one concept, most recent first."""
        return list(
            (
                await self._session.scalars(
                    select(Observation)
                    .where(Observation.concept_id == concept_id)
                    .order_by(Observation.observed_at.desc())
                    .limit(limit)
                )
            ).all()
        )

    async def visible_source_papers(
        self, user_id: uuid.UUID, paper_ids: Sequence[uuid.UUID]
    ) -> list[tuple[uuid.UUID, str | None]]:
        """Source papers filtered through `user_paper_access`.

        A concept remembers which paper introduced it, and that memory outlives
        the grant. Memory pointing at a paper is not authorization to read it
        (ARCHITECTURE 12 step 7), so the grant is re-checked here every time.
        """
        if not paper_ids:
            return []

        rows = (
            await self._session.execute(
                select(Paper.paper_id, Paper.title)
                .join(UserPaperAccess, UserPaperAccess.paper_id == Paper.paper_id)
                .where(
                    UserPaperAccess.user_id == user_id,
                    UserPaperAccess.revoked_at.is_(None),
                    Paper.paper_id.in_(list(paper_ids)),
                )
            )
        ).all()
        return [(paper_id, title) for paper_id, title in rows]
