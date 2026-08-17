"""Embedding-model versioning: staleness, exclusion, and re-indexing.

Mixing vector spaces is the failure mode these guard against. It produces no
error — cosine distance between two models' embeddings is a number, and it
sorts — so nothing here can be left to a runtime exception.
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Paper, User, UserPaperAccess
from app.ingestion.pipeline import ingest_paper
from app.services.embeddings import HashingEmbedder
from app.services.retrieval import (
    RetrievalService,
    authorized_paper_scope,
    stale_paper_scope,
)
from app.services.storage import LocalStorage
from scripts.reindex import find_stale
from tests.conftest import build_pdf

PAGES = [
    "Attention Mechanisms\nAbstract\nScaled dot product attention aids translation.\n"
    "1 Introduction\nRecurrent networks process tokens sequentially in order here.",
    "2 Method\nScaled dot product attention weights every token pair directly.\n"
    "3 Results\nTranslation quality improves over the recurrent baseline here.",
]


class OtherModelEmbedder(HashingEmbedder):
    """A second embedder whose vectors are incompatible by declaration."""

    @property
    def model_name(self) -> str:
        return "some-other-model-v9"


@pytest.fixture
def storage(storage_dir) -> LocalStorage:
    return LocalStorage(storage_dir)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


async def _ingest(session, storage, embedder, user=None) -> Paper:
    content_hash = uuid.uuid4().hex + uuid.uuid4().hex[:32]
    paper = Paper(
        content_hash=content_hash,
        storage_uri=storage.put(build_pdf(PAGES), content_hash=content_hash),
        processing_status="queued",
    )
    session.add(paper)
    await session.flush()
    await ingest_paper(session, paper.paper_id, storage=storage, embedder=embedder)
    if user is not None:
        session.add(UserPaperAccess(user_id=user.user_id, paper_id=paper.paper_id))
        await session.flush()
    return paper


async def _user(session) -> User:
    user = User(auth_subject=f"reindex-{uuid.uuid4()}")
    session.add(user)
    await session.flush()
    return user


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


async def test_paper_records_the_model_that_embedded_it(
    db_session: AsyncSession, storage, embedder
):
    paper = await _ingest(db_session, storage, embedder)
    assert paper.embedding_model == "local-hashing-v1"


async def test_find_stale_reports_papers_from_another_model(
    db_session: AsyncSession, storage, embedder
):
    current = await _ingest(db_session, storage, embedder)
    other = await _ingest(db_session, storage, OtherModelEmbedder())

    stale = await find_stale(db_session, "local-hashing-v1")
    stale_ids = {p.paper_id for p in stale}

    assert other.paper_id in stale_ids
    assert current.paper_id not in stale_ids


async def test_unrecorded_model_counts_as_stale(
    db_session: AsyncSession, storage, embedder
):
    """NULL means we do not know what produced those vectors, not that they fit."""
    paper = await _ingest(db_session, storage, embedder)
    paper.embedding_model = None
    await db_session.flush()

    stale = await find_stale(db_session, "local-hashing-v1")
    assert paper.paper_id in {p.paper_id for p in stale}


async def test_failed_papers_are_not_stale(db_session: AsyncSession, storage, embedder):
    """A paper with no vectors cannot have them in the wrong space."""
    paper = await _ingest(db_session, storage, embedder)
    paper.processing_status = "failed"
    paper.embedding_model = None
    await db_session.flush()

    assert paper.paper_id not in {
        p.paper_id for p in await find_stale(db_session, "local-hashing-v1")
    }


# --------------------------------------------------------------------------
# Exclusion
# --------------------------------------------------------------------------


async def test_retrieval_excludes_papers_from_another_model(
    db_session: AsyncSession, storage, embedder
):
    """The guard that matters: no results beats plausible nonsense."""
    stale = await _ingest(db_session, storage, OtherModelEmbedder())
    service = RetrievalService(db_session, embedder=embedder)

    results = await service.retrieve(
        "scaled dot product attention",
        paper_scope=[stale.paper_id],
        min_similarity=0.0,
    )
    assert results == []


async def test_retrieval_still_serves_matching_papers_alongside_stale_ones(
    db_session: AsyncSession, storage, embedder
):
    current = await _ingest(db_session, storage, embedder)
    stale = await _ingest(db_session, storage, OtherModelEmbedder())
    service = RetrievalService(db_session, embedder=embedder)

    results = await service.retrieve(
        "scaled dot product attention",
        paper_scope=[current.paper_id, stale.paper_id],
        min_similarity=0.0,
    )

    assert results
    assert {r.paper_id for r in results} == {current.paper_id}


async def test_scope_construction_can_filter_by_model(
    db_session: AsyncSession, storage, embedder
):
    user = await _user(db_session)
    current = await _ingest(db_session, storage, embedder, user)
    stale = await _ingest(db_session, storage, OtherModelEmbedder(), user)

    compatible = await authorized_paper_scope(
        db_session, user.user_id, embedding_model="local-hashing-v1"
    )
    assert compatible == [current.paper_id]

    # Without the filter the paper is still authorized — this is a vector-space
    # exclusion, not an authorization one.
    everything = await authorized_paper_scope(db_session, user.user_id)
    assert set(everything) == {current.paper_id, stale.paper_id}


async def test_stale_scope_names_what_needs_reindexing(
    db_session: AsyncSession, storage, embedder
):
    user = await _user(db_session)
    await _ingest(db_session, storage, embedder, user)
    stale = await _ingest(db_session, storage, OtherModelEmbedder(), user)

    assert await stale_paper_scope(
        db_session, user.user_id, embedding_model="local-hashing-v1"
    ) == [stale.paper_id]


# --------------------------------------------------------------------------
# Visibility
# --------------------------------------------------------------------------


async def test_api_flags_a_paper_that_needs_reindexing(
    client: AsyncClient, db_session: AsyncSession, dev_auth, storage, embedder
):
    await client.get("/api/me")
    user = await db_session.scalar(select(User).where(User.auth_subject == dev_auth))
    current = await _ingest(db_session, storage, embedder, user)
    stale = await _ingest(db_session, storage, OtherModelEmbedder(), user)

    by_id = {p["paper_id"]: p for p in (await client.get("/api/papers")).json()}

    assert by_id[str(current.paper_id)]["needs_reindex"] is False
    assert by_id[str(stale.paper_id)]["needs_reindex"] is True
    assert by_id[str(stale.paper_id)]["embedding_model"] == "some-other-model-v9"


# --------------------------------------------------------------------------
# Re-indexing
# --------------------------------------------------------------------------


async def test_reindex_moves_a_paper_into_the_active_vector_space(
    db_session: AsyncSession, storage, embedder
):
    stale = await _ingest(db_session, storage, OtherModelEmbedder())
    assert stale.embedding_model == "some-other-model-v9"

    await ingest_paper(db_session, stale.paper_id, storage=storage, embedder=embedder)

    assert stale.embedding_model == "local-hashing-v1"
    assert stale.processing_status == "ready"

    results = await RetrievalService(db_session, embedder=embedder).retrieve(
        "scaled dot product attention",
        paper_scope=[stale.paper_id],
        min_similarity=0.0,
    )
    assert results, "the paper should be searchable again after re-indexing"


async def test_reindexing_is_idempotent(db_session: AsyncSession, storage, embedder):
    """Running it twice must not duplicate chunks or leave anything stale."""
    paper = await _ingest(db_session, storage, OtherModelEmbedder())

    first = await ingest_paper(
        db_session, paper.paper_id, storage=storage, embedder=embedder
    )
    second = await ingest_paper(
        db_session, paper.paper_id, storage=storage, embedder=embedder
    )

    assert (first.section_count, first.chunk_count) == (
        second.section_count,
        second.chunk_count,
    )
    # Scoped to this paper: a global check would also see whatever a developer
    # has ingested into their local database with a different embedder.
    stale_ids = {p.paper_id for p in await find_stale(db_session, "local-hashing-v1")}
    assert paper.paper_id not in stale_ids


async def test_reindex_does_not_recanonicalize_concepts(
    db_session: AsyncSession, storage, embedder
):
    """Re-embedding is a vector operation; it is not about any reader."""
    from sqlalchemy import func as sa_func

    from app.db.models import Concept

    user = await _user(db_session)
    paper = await _ingest(db_session, storage, OtherModelEmbedder(), user)

    # Re-index without a user_id, as the command does.
    await ingest_paper(db_session, paper.paper_id, storage=storage, embedder=embedder)

    concepts = await db_session.scalar(
        select(sa_func.count()).select_from(Concept).where(Concept.user_id == user.user_id)
    )
    assert concepts == 0
