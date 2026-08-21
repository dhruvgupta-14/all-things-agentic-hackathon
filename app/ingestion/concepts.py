"""Phase 6b — fold a paper's concept candidates into one reader's graph.

Concepts are user-scoped, so this runs per user even when the paper's analysis
is shared. It implements the governing pattern from ARCHITECTURE 2.2:

    [D] deterministic recall -> [M] model adjudication -> [D] deterministic commit

Recall is exact-match, then alias containment, then vector ANN. Only the
ambiguous middle band is worth a model call, and the commit is always ours.

Idempotency matters more here than anywhere else in ingestion: a retry must
not create a second "attention mechanism" for the same reader. Every write is
an upsert against the partial unique index on (user_id, normalized_name), and
`source_paper_ids` is a set union rather than an append.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.db.models import Concept, ConceptRelationship
from app.services import adjudication
from app.services.adjudication import Adjudicator, ConceptPair
from app.services.analysis import ConceptCandidate
from app.services.embeddings import Embedder

logger = logging.getLogger(__name__)

# The ANN floor from §16.3 step 3. Below it, a candidate is a new concept at
# zero model cost. Above it, the model decides — there is deliberately no
# "similar enough to merge without asking" band.
#
# Measured with gemini-embedding-001, which is why:
#     variational inference / variational autoencoder = 0.9263   MUST NOT merge
#     evidence lower bound  / ELBO                    = 0.8595   MUST merge
#
# The pair that must stay separate scores *higher* than the pair that must
# merge. No threshold separates them, so any auto-merge band silently collapses
# adjacent concepts into one — which is the failure that destroys the concept
# graph, and the one §16.3 exists to prevent. Recall comes from embeddings;
# precision comes from the model; the two are not interchangeable.
AUTO_DISTINCT_BELOW = 0.72

# A merge is destructive and awkward to unpick, so it needs more certainty
# than an edge does. Spike S-4 returned 0.95-1.00 on every correct "same"
# verdict, so 0.85 is comfortably clear of the observed noise floor.
MIN_MERGE_CONFIDENCE = 0.85
MIN_EDGE_CONFIDENCE = 0.70


def normalize_name(name: str) -> str:
    """The exact-match key. Case, punctuation and spacing are not identity."""
    lowered = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    return re.sub(r"\s+", " ", lowered).strip()


@dataclass(slots=True)
class CanonicalizationResult:
    created: list[uuid.UUID] = field(default_factory=list)
    matched: list[uuid.UUID] = field(default_factory=list)
    relationships_created: int = 0

    @property
    def total(self) -> int:
        return len(self.created) + len(self.matched)


class ConceptService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        embedder: Embedder,
        adjudicator: Adjudicator | None = None,
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._adjudicator = adjudicator or adjudication.get_adjudicator()

    # -- recall ------------------------------------------------------------

    async def _exact_match(self, user_id: uuid.UUID, candidate: ConceptCandidate):
        """Zero-cost path: normalized name, or one of the stored aliases.

        Served entirely by the partial unique index and the GIN index on
        aliases, so the common case never touches a model or a vector.
        """
        normalized = normalize_name(candidate.name)
        names = {normalized, *(normalize_name(a) for a in candidate.aliases)}

        return await self._session.scalar(
            select(Concept)
            .where(
                Concept.user_id == user_id,
                Concept.merged_into_id.is_(None),
                or_(
                    Concept.normalized_name.in_(list(names)),
                    Concept.aliases.overlap([candidate.name, *candidate.aliases]),
                ),
            )
            .limit(1)
        )

    async def _nearest(
        self, user_id: uuid.UUID, vector: list[float]
    ) -> tuple[Concept | None, float]:
        distance = Concept.embedding.cosine_distance(vector)
        row = (
            await self._session.execute(
                select(Concept, (1 - distance).label("similarity"))
                .where(
                    Concept.user_id == user_id,
                    Concept.merged_into_id.is_(None),
                    Concept.embedding.isnot(None),
                )
                .order_by(distance)
                .limit(1)
            )
        ).first()

        if row is None:
            return None, 0.0
        concept, similarity = row
        return concept, float(similarity)

    # -- commit ------------------------------------------------------------

    async def _upsert(
        self,
        user_id: uuid.UUID,
        candidate: ConceptCandidate,
        paper_id: uuid.UUID,
        vector: list[float],
    ) -> tuple[uuid.UUID, bool]:
        """Insert the concept. Returns (concept_id, created).

        The exact-match path above already handles the everyday "we have seen
        this concept" case, so ON CONFLICT here is a race guard rather than
        the main road: a concurrent job inserting the same name loses, and we
        fold this paper into the winner's row instead of failing the job.
        """
        normalized = normalize_name(candidate.name)[:200]

        statement = (
            insert(Concept)
            .values(
                user_id=user_id,
                canonical_name=candidate.name[:200],
                normalized_name=normalized,
                aliases=candidate.aliases,
                description=candidate.description,
                embedding=vector,
                source_paper_ids=[paper_id],
            )
            .on_conflict_do_nothing(
                index_elements=[Concept.user_id, Concept.normalized_name],
                index_where=Concept.merged_into_id.is_(None),
            )
            .returning(Concept.concept_id)
        )

        created_id = await self._session.scalar(statement)
        if created_id is not None:
            return created_id, True

        winner = await self._session.scalar(
            select(Concept).where(
                Concept.user_id == user_id,
                Concept.normalized_name == normalized,
                Concept.merged_into_id.is_(None),
            )
        )
        if winner is None:  # pragma: no cover - the index guarantees one exists
            raise RuntimeError(f"concept {normalized!r} vanished mid-upsert")

        if paper_id not in (winner.source_paper_ids or []):
            winner.source_paper_ids = [*(winner.source_paper_ids or []), paper_id]
            await self._session.flush()
        return winner.concept_id, False

    async def _link(
        self,
        user_id: uuid.UUID,
        source_id: uuid.UUID,
        target_id: uuid.UUID,
        confidence: float,
    ) -> bool:
        """Create a prerequisite edge, ignoring one that already exists."""
        return await self._link_typed(
            user_id, source_id, target_id, "prerequisite_of", confidence
        )

    async def _link_typed(
        self,
        user_id: uuid.UUID,
        source_id: uuid.UUID,
        target_id: uuid.UUID,
        relationship_type: str,
        confidence: float,
    ) -> bool:
        """Create a typed edge, ignoring one that already exists."""
        if source_id == target_id:
            return False

        statement = (
            insert(ConceptRelationship)
            .values(
                user_id=user_id,
                source_concept_id=source_id,
                target_concept_id=target_id,
                relationship_type=relationship_type,
                confidence=confidence,
                discovery_method="model",
            )
            .on_conflict_do_nothing(
                index_elements=[
                    ConceptRelationship.user_id,
                    ConceptRelationship.source_concept_id,
                    ConceptRelationship.target_concept_id,
                    ConceptRelationship.relationship_type,
                ]
            )
            .returning(ConceptRelationship.relationship_id)
        )
        return (await self._session.execute(statement)).first() is not None

    # -- entry point -------------------------------------------------------

    async def canonicalize(
        self,
        user_id: uuid.UUID,
        paper_id: uuid.UUID,
        candidates: list[ConceptCandidate],
    ) -> CanonicalizationResult:
        result = CanonicalizationResult()
        if not candidates:
            return result

        vectors = self._embedder.embed_batch([c.name for c in candidates])
        # Candidate name -> the concept id it resolved to, for prerequisites.
        resolved: dict[str, uuid.UUID] = {}
        # (candidate name, existing concept id, verdict) from the ambiguous
        # band, committed once both endpoints have ids.
        proposed_edges: list[tuple[str, uuid.UUID, object]] = []

        # --- pass 1: deterministic recall (§16.3 steps 1-3) ----------------
        # Exact match settles the common case at zero model cost. What is left
        # above the ANN floor is the ambiguous band, gathered here so the whole
        # paper costs one adjudication call rather than one per concept.
        matched_exactly: dict[int, Concept] = {}
        ambiguous: list[tuple[int, Concept]] = []

        for index, (candidate, vector) in enumerate(
            zip(candidates, vectors, strict=True)
        ):
            existing = await self._exact_match(user_id, candidate)
            if existing is not None:
                matched_exactly[index] = existing
                continue

            nearest, similarity = await self._nearest(user_id, vector)
            if nearest is not None and similarity >= AUTO_DISTINCT_BELOW:
                ambiguous.append((index, nearest))

        # --- pass 2: model adjudication (§16.3 step 4), one call -----------
        # Embeddings put "variational inference" and "variational autoencoder"
        # close together; cosine similarity cannot tell "same" from "adjacent",
        # and this is the only place that distinction gets made.
        verdicts: dict[int, object] = {}
        if ambiguous:
            pairs = [
                ConceptPair(
                    candidate_name=candidates[index].name,
                    existing_name=nearest.canonical_name,
                    candidate_description=candidates[index].description,
                    existing_description=nearest.description,
                )
                for index, nearest in ambiguous
            ]
            try:
                judged = await run_in_threadpool(
                    self._adjudicator.adjudicate_batch, pairs
                )
                verdicts = {
                    index: verdict
                    for (index, _), verdict in zip(ambiguous, judged, strict=True)
                }
            except Exception as exc:
                # Falling back to "distinct" duplicates a concept at worst.
                # Failing the ingest would lose a searchable paper over an
                # enrichment step, which is the worse trade.
                logger.warning("adjudication unavailable, treating as distinct: %s", exc)

        # --- pass 3: deterministic commit ----------------------------------
        for index, (candidate, vector) in enumerate(
            zip(candidates, vectors, strict=True)
        ):
            existing = matched_exactly.get(index)
            verdict = verdicts.get(index)

            if existing is None and verdict is not None:
                nearest = dict(ambiguous)[index]
                if (
                    verdict.verdict == "same"
                    and verdict.confidence >= MIN_MERGE_CONFIDENCE
                ):
                    existing = nearest
                elif (
                    verdict.verdict == "related"
                    and verdict.confidence >= MIN_EDGE_CONFIDENCE
                ):
                    # Stays a separate concept, but the connection is worth
                    # keeping — resolved into an edge once both sides have ids.
                    proposed_edges.append((candidate.name, nearest.concept_id, verdict))

            if existing is not None:
                if paper_id not in (existing.source_paper_ids or []):
                    existing.source_paper_ids = [
                        *(existing.source_paper_ids or []),
                        paper_id,
                    ]
                # Widen the alias set with anything new this paper called it.
                merged_aliases = list(existing.aliases or [])
                for alias in candidate.aliases:
                    if alias not in merged_aliases:
                        merged_aliases.append(alias)
                existing.aliases = merged_aliases
                await self._session.flush()

                resolved[candidate.name] = existing.concept_id
                result.matched.append(existing.concept_id)
                continue

            concept_id, created = await self._upsert(user_id, candidate, paper_id, vector)
            resolved[candidate.name] = concept_id
            (result.created if created else result.matched).append(concept_id)

        # Adjudicated "related" verdicts become typed edges.
        for name, existing_id, verdict in proposed_edges:
            source_id = resolved.get(name)
            edge_type = verdict.typed_relationship()
            if source_id is None or edge_type is None:
                continue
            if await self._link_typed(
                user_id, source_id, existing_id, edge_type, verdict.confidence
            ):
                result.relationships_created += 1

        # Prerequisite edges are created only once every concept has an id.
        for candidate in candidates:
            target = resolved.get(candidate.name)
            if target is None:
                continue
            for prerequisite in candidate.prerequisites:
                source = resolved.get(prerequisite)
                if source is None:
                    continue
                if await self._link(user_id, source, target, confidence=0.7):
                    result.relationships_created += 1

        await self._session.flush()
        logger.info(
            "canonicalized concepts",
            extra={
                "user_id": str(user_id),
                "paper_id": str(paper_id),
                "created": len(result.created),
                "matched": len(result.matched),
            },
        )
        return result
