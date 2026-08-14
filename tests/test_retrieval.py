"""Embeddings, phase 5, and filtered retrieval."""

import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Paper, User, UserPaperAccess
from app.ingestion.pipeline import ingest_paper
from app.services.embeddings import HashingEmbedder
from app.services.retrieval import (
    RetrievalScopeViolation,
    RetrievalService,
    authorized_paper_scope,
)
from app.services.storage import LocalStorage
from tests.conftest import build_pdf

ATTENTION_PAPER = [
    "Attention Mechanisms\nAbstract\nWe study attention for machine translation.\n"
    "1 Introduction\nRecurrent networks process tokens sequentially in order.",
    "2 Method\nScaled dot product attention weights every token pair directly.\n"
    "3 Results\nTranslation quality improves over the recurrent baseline.",
]

PROTEIN_PAPER = [
    "Protein Folding Structures\nAbstract\nWe predict tertiary protein structure.\n"
    "1 Introduction\nAmino acid chains fold into stable three dimensional shapes.",
    "2 Method\nA diffusion process refines candidate backbone coordinates.\n"
    "3 Results\nPredicted structures match crystallography measurements closely.",
]


@pytest.fixture
def storage(storage_dir) -> LocalStorage:
    return LocalStorage(storage_dir)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


async def _ingest(session, storage, embedder, pages) -> Paper:
    content_hash = uuid.uuid4().hex + uuid.uuid4().hex[:32]
    paper = Paper(
        content_hash=content_hash,
        storage_uri=storage.put(build_pdf(pages), content_hash=content_hash),
        processing_status="queued",
    )
    session.add(paper)
    await session.flush()
    await ingest_paper(session, paper.paper_id, storage=storage, embedder=embedder)
    return paper


async def _user_with(session, subject: str, papers: list[Paper]) -> User:
    user = User(auth_subject=f"{subject}-{uuid.uuid4()}")
    session.add(user)
    await session.flush()
    for paper in papers:
        session.add(UserPaperAccess(user_id=user.user_id, paper_id=paper.paper_id))
    await session.flush()
    return user


# --------------------------------------------------------------------------
# Embedder
# --------------------------------------------------------------------------


def test_embeddings_are_deterministic(embedder):
    assert embedder.embed_query("attention") == embedder.embed_query("attention")


def test_embeddings_are_unit_vectors(embedder):
    for text in ["attention mechanisms", "", "a"]:
        vector = embedder.embed_query(text)
        assert len(vector) == 768
        assert sum(value * value for value in vector) == pytest.approx(1.0, abs=1e-9)


def test_empty_text_does_not_produce_a_zero_vector(embedder):
    """A zero vector makes cosine distance undefined rather than merely far."""
    assert any(value != 0.0 for value in embedder.embed_query(""))


def test_shared_vocabulary_scores_closer_than_unrelated_text(embedder):
    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    query = embedder.embed_query("attention weights every token")
    related = embedder.embed_query("attention weights token pairs directly")
    unrelated = embedder.embed_query("amino acid chains fold into shapes")

    assert cosine(query, related) > cosine(query, unrelated)


def test_batch_matches_single(embedder):
    texts = ["alpha beta", "gamma delta"]
    assert embedder.embed_batch(texts) == [embedder.embed_query(t) for t in texts]


# --------------------------------------------------------------------------
# Phase 5
# --------------------------------------------------------------------------


async def test_ingestion_now_reaches_ready(
    db_session: AsyncSession, storage, embedder
):
    paper = await _ingest(db_session, storage, embedder, ATTENTION_PAPER)

    assert paper.processing_status == "ready"
    assert paper.processing_phase is None
    assert paper.embedding_model == "local-hashing-v1"


async def test_indexable_chunks_are_embedded(
    db_session: AsyncSession, storage, embedder
):
    paper = await _ingest(db_session, storage, embedder, ATTENTION_PAPER)

    missing = await db_session.scalar(
        select(func.count())
        .select_from(Chunk)
        .where(
            Chunk.paper_id == paper.paper_id,
            Chunk.is_indexable.is_(True),
            Chunk.embedding.is_(None),
        )
    )
    assert missing == 0


async def test_partial_document_stays_partially_ready(
    db_session: AsyncSession, storage, embedder
):
    """Unreadable pages must not be papered over by a `ready` status."""
    paper = await _ingest(
        db_session,
        storage,
        embedder,
        ATTENTION_PAPER + ["", ""],
    )

    assert paper.processing_status == "partially_ready"
    assert paper.unreadable_pages == [3, 4]


async def test_embedding_failure_is_transient_not_permanent(
    db_session: AsyncSession, storage
):
    """A embedding outage must not write the paper off as failed."""

    class BrokenEmbedder(HashingEmbedder):
        def embed_batch(self, texts):
            raise RuntimeError("vertex is down")

    from app.ingestion.pipeline import TransientIngestionError

    content_hash = uuid.uuid4().hex + uuid.uuid4().hex[:32]
    paper = Paper(
        content_hash=content_hash,
        storage_uri=storage.put(build_pdf(ATTENTION_PAPER), content_hash=content_hash),
        processing_status="queued",
    )
    db_session.add(paper)
    await db_session.flush()

    with pytest.raises(TransientIngestionError):
        await ingest_paper(
            db_session, paper.paper_id, storage=storage, embedder=BrokenEmbedder()
        )

    assert paper.processing_status != "failed"
    assert paper.error_code is None


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


async def test_retrieval_finds_the_relevant_passage(
    db_session: AsyncSession, storage, embedder
):
    paper = await _ingest(db_session, storage, embedder, ATTENTION_PAPER)
    service = RetrievalService(db_session, embedder=embedder)

    results = await service.retrieve(
        "scaled dot product attention weights token pairs",
        paper_scope=[paper.paper_id],
        min_similarity=0.0,
    )

    assert results
    assert "attention" in results[0].content.lower()
    assert results[0].rank == 1
    assert results[0].paper_id == paper.paper_id


async def test_results_are_ranked_by_descending_similarity(
    db_session: AsyncSession, storage, embedder
):
    paper = await _ingest(db_session, storage, embedder, ATTENTION_PAPER)
    service = RetrievalService(db_session, embedder=embedder)

    results = await service.retrieve(
        "attention", paper_scope=[paper.paper_id], min_similarity=0.0
    )

    scores = [r.similarity for r in results]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


async def test_scope_excludes_papers_the_user_cannot_read(
    db_session: AsyncSession, storage, embedder
):
    """The isolation guarantee: another user's paper is not retrievable."""
    mine = await _ingest(db_session, storage, embedder, ATTENTION_PAPER)
    theirs = await _ingest(db_session, storage, embedder, PROTEIN_PAPER)
    service = RetrievalService(db_session, embedder=embedder)

    results = await service.retrieve(
        "amino acid chains fold into stable shapes",
        paper_scope=[mine.paper_id],
        min_similarity=0.0,
    )

    assert results, "the query should still match something in the authorized paper"
    assert {r.paper_id for r in results} == {mine.paper_id}
    assert theirs.paper_id not in {r.paper_id for r in results}


async def test_empty_scope_returns_nothing_rather_than_everything(
    db_session: AsyncSession, storage, embedder
):
    """Failing open here would leak every paper in the database."""
    await _ingest(db_session, storage, embedder, ATTENTION_PAPER)
    service = RetrievalService(db_session, embedder=embedder)

    assert await service.retrieve("attention", paper_scope=[]) == []


async def test_blank_query_returns_nothing(
    db_session: AsyncSession, storage, embedder
):
    paper = await _ingest(db_session, storage, embedder, ATTENTION_PAPER)
    service = RetrievalService(db_session, embedder=embedder)

    assert await service.retrieve("   ", paper_scope=[paper.paper_id]) == []


async def test_relevance_floor_suppresses_weak_matches(
    db_session: AsyncSession, storage, embedder
):
    """Returning nothing beats grounding an answer in a weak match."""
    paper = await _ingest(db_session, storage, embedder, ATTENTION_PAPER)
    service = RetrievalService(db_session, embedder=embedder)

    assert (
        await service.retrieve(
            "unrelated zebra husbandry techniques",
            paper_scope=[paper.paper_id],
            min_similarity=0.99,
        )
        == []
    )


async def test_top_k_is_respected(db_session: AsyncSession, storage, embedder):
    paper = await _ingest(db_session, storage, embedder, ATTENTION_PAPER)
    service = RetrievalService(db_session, embedder=embedder)

    results = await service.retrieve(
        "attention", paper_scope=[paper.paper_id], top_k=2, min_similarity=0.0
    )
    assert len(results) <= 2


async def test_reference_chunks_are_never_retrieved(
    db_session: AsyncSession, storage, embedder
):
    pages = ATTENTION_PAPER + [
        "References\n[1] Someone. A prior paper on attention weights. 2015.\n"
        "[2] Another. More attention research about tokens. 2016."
    ]
    paper = await _ingest(db_session, storage, embedder, pages)
    service = RetrievalService(db_session, embedder=embedder)

    results = await service.retrieve(
        "attention", paper_scope=[paper.paper_id], top_k=50, min_similarity=0.0
    )

    assert all(r.section_role != "references" for r in results)


async def test_results_carry_a_usable_citation_locator(
    db_session: AsyncSession, storage, embedder
):
    paper = await _ingest(db_session, storage, embedder, ATTENTION_PAPER)
    service = RetrievalService(db_session, embedder=embedder)

    results = await service.retrieve(
        "attention", paper_scope=[paper.paper_id], min_similarity=0.0
    )

    top = results[0]
    assert top.page_start >= 1
    assert top.section_path
    assert "p." in top.citation_locator


async def test_scope_violation_fails_closed(
    db_session: AsyncSession, storage, embedder, monkeypatch
):
    """If the SQL filter ever regressed, the assertion must stop the turn."""
    paper = await _ingest(db_session, storage, embedder, ATTENTION_PAPER)
    service = RetrievalService(db_session, embedder=embedder)

    real_execute = db_session.execute
    other_id = uuid.uuid4()

    async def leaky_execute(statement, *args, **kwargs):
        result = await real_execute(statement, *args, **kwargs)
        rows = result.all()
        if rows and hasattr(rows[0][0], "paper_id"):
            rows[0][0].paper_id = other_id  # simulate a broken filter

            class _Fake:
                def all(self_inner):
                    return rows

            return _Fake()
        return result

    monkeypatch.setattr(db_session, "execute", leaky_execute)

    with pytest.raises(RetrievalScopeViolation):
        await service.retrieve(
            "attention", paper_scope=[paper.paper_id], min_similarity=0.0
        )


# --------------------------------------------------------------------------
# Scope construction
# --------------------------------------------------------------------------


async def test_authorized_scope_lists_only_granted_papers(
    db_session: AsyncSession, storage, embedder
):
    mine = await _ingest(db_session, storage, embedder, ATTENTION_PAPER)
    theirs = await _ingest(db_session, storage, embedder, PROTEIN_PAPER)
    user = await _user_with(db_session, "reader", [mine])

    scope = await authorized_paper_scope(db_session, user.user_id)
    assert scope == [mine.paper_id]
    assert theirs.paper_id not in scope


async def test_revoked_grant_leaves_the_scope(
    db_session: AsyncSession, storage, embedder
):
    from datetime import UTC, datetime

    paper = await _ingest(db_session, storage, embedder, ATTENTION_PAPER)
    user = await _user_with(db_session, "reader", [paper])

    grant = await db_session.scalar(
        select(UserPaperAccess).where(UserPaperAccess.user_id == user.user_id)
    )
    grant.revoked_at = datetime.now(UTC)
    await db_session.flush()

    assert await authorized_paper_scope(db_session, user.user_id) == []


async def test_requested_papers_are_intersected_with_grants(
    db_session: AsyncSession, storage, embedder
):
    """Asking for a paper you were never granted narrows to nothing."""
    mine = await _ingest(db_session, storage, embedder, ATTENTION_PAPER)
    theirs = await _ingest(db_session, storage, embedder, PROTEIN_PAPER)
    user = await _user_with(db_session, "reader", [mine])

    assert await authorized_paper_scope(db_session, user.user_id, [theirs.paper_id]) == []
    assert await authorized_paper_scope(db_session, user.user_id, [mine.paper_id]) == [
        mine.paper_id
    ]
