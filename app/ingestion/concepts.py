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

from app.db.models import Concept, ConceptRelationship
from app.services.analysis import ConceptCandidate
from app.services.embeddings import Embedder

logger = logging.getLogger(__name__)

# Above this cosine similarity two names are the same concept without asking.
AUTO_MERGE_ABOVE = 0.92
# Below this they are unrelated. Between the two is the adjudication band.
AUTO_DISTINCT_BELOW = 0.72


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
    def __init__(self, session: AsyncSession, *, embedder: Embedder) -> None:
        self._session = session
        self._embedder = embedder

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
        if source_id == target_id:
            return False

        statement = (
            insert(ConceptRelationship)
            .values(
                user_id=user_id,
                source_concept_id=source_id,
                target_concept_id=target_id,
                relationship_type="prerequisite_of",
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

        for candidate, vector in zip(candidates, vectors, strict=True):
            existing = await self._exact_match(user_id, candidate)

            if existing is None:
                nearest, similarity = await self._nearest(user_id, vector)
                if nearest is not None and similarity >= AUTO_MERGE_ABOVE:
                    existing = nearest
                elif nearest is not None and similarity >= AUTO_DISTINCT_BELOW:
                    # The adjudication band. Until the agent can be asked,
                    # treat it as distinct: a wrong merge is destructive and
                    # awkward to unpick, while a duplicate is merely untidy
                    # and can be merged later via `merged_into_id`.
                    logger.debug(
                        "ambiguous concept match left distinct",
                        extra={"name": candidate.name, "similarity": similarity},
                    )

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

        # Relationships are created only once every concept has an id.
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
