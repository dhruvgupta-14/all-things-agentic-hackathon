"""Filtered ANN retrieval over chunk embeddings.

This is what the agent's `retrieve_paper_context` tool calls. Everything here
is deterministic: the model supplies a query string and nothing else. Scope is
constructed by application code from session state and verified grants, and is
passed in — never taken from a tool argument (ARCHITECTURE 9.1).

The scope filter is a `WHERE` clause evaluated *inside* the vector search
rather than a post-filter applied to its output. With a separate vector store
that distinction is what silently degrades recall; in pgvector it is free.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import Float, or_, select
from sqlalchemy import true as sa_true
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Chunk, Paper, Section, UserPaperAccess
from app.services.embeddings import Embedder, get_embedder

logger = logging.getLogger(__name__)


class RetrievalScopeViolation(Exception):
    """A chunk outside the authorized scope came back from the database.

    This is a defect, not a permission denial: the filter is applied in SQL,
    so reaching this means the query itself is wrong. The turn fails closed.
    """


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    paper_id: uuid.UUID
    content: str
    similarity: float
    rank: int
    page_start: int
    page_end: int
    section_path: str
    section_heading: str | None
    section_role: str

    @property
    def citation_locator(self) -> str:
        """The human-actionable half of a citation: 'section 3.2, p.5'."""
        pages = (
            f"p.{self.page_start}"
            if self.page_start == self.page_end
            else f"pp.{self.page_start}-{self.page_end}"
        )
        return f"{self.section_path}, {pages}"


class RetrievalService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        self._session = session
        self._embedder = embedder or get_embedder()

    async def retrieve(
        self,
        query: str,
        *,
        paper_scope: Sequence[uuid.UUID],
        top_k: int | None = None,
        min_similarity: float | None = None,
        section_role: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the most relevant chunks within `paper_scope`.

        `paper_scope` is the enumerated set of papers this user may read,
        already checked against `user_paper_access`. An empty scope returns
        nothing rather than falling back to an unfiltered search.

        `section_role` narrows to one part of the paper. It is the one filter
        the agent chooses (ARCHITECTURE 14.2) — it constrains *what* is
        searched, never *whose* data, so it is safe in the model's hands.
        """
        settings = get_settings()
        top_k = top_k or settings.retrieval_top_k
        # Precedence: explicit argument, then an operator override in config,
        # then the floor that belongs to this embedder's vector space. Scores
        # are not comparable between models, so the embedder's own default is
        # the only one guaranteed to be meaningful.
        if min_similarity is not None:
            floor = min_similarity
        elif settings.retrieval_min_similarity is not None:
            floor = settings.retrieval_min_similarity
        else:
            floor = self._embedder.default_min_similarity

        if not paper_scope or not query.strip():
            return []

        scope = list(paper_scope)
        query_vector = self._embedder.embed_query(query)

        # Cosine distance in [0, 2]; similarity is its complement.
        distance = Chunk.embedding.cosine_distance(query_vector)
        similarity = (1 - distance).cast(Float).label("similarity")

        # The predicates on is_indexable and embedding match the partial HNSW
        # index exactly, so the planner can use it.
        #
        # The join to papers enforces the vector-space guard: cosine distance
        # between embeddings from two different models is a meaningless number
        # that still sorts, so mixing them returns confident nonsense rather
        # than an error. Scope construction filters for this too; this is the
        # belt that holds if a caller builds a scope carelessly.
        statement = (
            select(Chunk, Section, similarity)
            .join(Section, Section.section_id == Chunk.section_id)
            .join(Paper, Paper.paper_id == Chunk.paper_id)
            .where(
                Chunk.paper_id.in_(scope),
                Chunk.is_indexable.is_(True),
                Chunk.embedding.isnot(None),
                Paper.embedding_model == self._embedder.model_name,
            )
            .order_by(distance)
            .where(
                # A no-op when unset, so the query shape is unchanged for the
                # common case and still matches the partial HNSW index.
                Section.section_role == section_role
                if section_role
                else sa_true()
            )
            # Over-fetch so the relevance floor and dedup have something to
            # cut into, rather than returning fewer than top_k after filtering.
            .limit(top_k * 3)
        )

        rows = (await self._session.execute(statement)).all()

        authorized = set(scope)
        results: list[RetrievedChunk] = []
        seen_content: set[str] = set()

        for chunk, section, score in rows:
            # Checkpoint 4: assert what came back is what we asked for.
            if chunk.paper_id not in authorized:
                logger.error(
                    "SECURITY: retrieval returned an out-of-scope chunk",
                    extra={
                        "chunk_id": str(chunk.chunk_id),
                        "paper_id": str(chunk.paper_id),
                    },
                )
                raise RetrievalScopeViolation(
                    f"chunk {chunk.chunk_id} is outside the authorized scope"
                )

            if score is None or float(score) < floor:
                continue

            # Identical text across papers adds nothing to the context window.
            if chunk.content_hash in seen_content:
                continue
            seen_content.add(chunk.content_hash)

            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    content=chunk.content,
                    similarity=round(float(score), 6),
                    rank=len(results) + 1,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_path=section.section_path,
                    section_heading=section.heading,
                    section_role=section.section_role,
                )
            )

            if len(results) >= top_k:
                break

        logger.info(
            "retrieval complete",
            extra={
                "papers": len(scope),
                "candidates": len(rows),
                "returned": len(results),
            },
        )
        return results


async def authorized_paper_scope(
    session: AsyncSession,
    user_id: uuid.UUID,
    paper_ids: Sequence[uuid.UUID] | None = None,
    *,
    embedding_model: str | None = None,
) -> list[uuid.UUID]:
    """The papers this user may currently read, optionally narrowed.

    Scope construction lives here rather than at the call site so there is one
    place where a revoked grant takes effect.

    `embedding_model` additionally excludes papers embedded with a different
    model. Those papers are readable — the exclusion is about vector-space
    compatibility, not authorization — so a caller that wants to tell the user
    "this needs re-indexing" should ask `stale_paper_scope` rather than
    inferring it from the gap.
    """
    statement = select(UserPaperAccess.paper_id).where(
        UserPaperAccess.user_id == user_id,
        UserPaperAccess.revoked_at.is_(None),
    )
    if paper_ids is not None:
        if not paper_ids:
            return []
        statement = statement.where(UserPaperAccess.paper_id.in_(list(paper_ids)))

    if embedding_model is not None:
        statement = statement.join(
            Paper, Paper.paper_id == UserPaperAccess.paper_id
        ).where(Paper.embedding_model == embedding_model)

    return list((await session.scalars(statement)).all())


async def stale_paper_scope(
    session: AsyncSession, user_id: uuid.UUID, *, embedding_model: str
) -> list[uuid.UUID]:
    """This user's papers that were embedded with some other model.

    A NULL `embedding_model` counts as stale: it predates the field being
    recorded, so what produced those vectors is unknown, and unknown is not
    the same as compatible.
    """
    statement = (
        select(UserPaperAccess.paper_id)
        .join(Paper, Paper.paper_id == UserPaperAccess.paper_id)
        .where(
            UserPaperAccess.user_id == user_id,
            UserPaperAccess.revoked_at.is_(None),
            or_(
                Paper.embedding_model.is_(None),
                Paper.embedding_model != embedding_model,
            ),
        )
    )
    return list((await session.scalars(statement)).all())
