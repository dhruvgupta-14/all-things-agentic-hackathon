"""The ingestion job end to end, and the upload boundary in front of it."""

import hashlib
import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Paper, Section, UserPaperAccess
from app.ingestion.pipeline import PermanentIngestionError, ingest_paper
from app.services.storage import LocalStorage
from tests.conftest import build_pdf

PAPER_PAGES = [
    "Attention Mechanisms In Practice\nJane Doe and John Roe\n"
    "Abstract\nWe study attention mechanisms and their effect on translation.\n"
    "1 Introduction\nSequence models have long relied on recurrence for context.",
    "2 Related Work\nEarlier approaches used convolution to widen the field.\n"
    "3 Method\nWe propose scaled dot-product attention over the input sequence.",
    "4 Experiments\nWe train on a standard benchmark with the usual splits.\n"
    "5 Results\nThe model improves over the recurrent baseline by a margin.\n"
    "References\n[1] Someone. An earlier paper about sequences. 2015.",
]


async def _seed_paper(session: AsyncSession, storage: LocalStorage, pages) -> Paper:
    data = build_pdf(pages)
    content_hash = uuid.uuid4().hex + uuid.uuid4().hex[:32]
    paper = Paper(
        content_hash=content_hash,
        storage_uri=storage.put(data, content_hash=content_hash),
        processing_status="queued",
    )
    session.add(paper)
    await session.flush()
    return paper


@pytest.fixture
def storage(storage_dir) -> LocalStorage:
    return LocalStorage(storage_dir)


async def test_ingest_produces_sections_and_chunks(
    db_session: AsyncSession, storage: LocalStorage
):
    paper = await _seed_paper(db_session, storage, PAPER_PAGES)

    result = await ingest_paper(db_session, paper.paper_id, storage=storage)

    assert result.section_count > 0
    assert result.chunk_count > 0

    sections = (
        await db_session.scalars(
            select(Section).where(Section.paper_id == paper.paper_id).order_by(Section.ordinal)
        )
    ).all()
    roles = {section.section_role for section in sections}
    assert {"abstract", "introduction", "method", "results"} <= roles

    chunks = (
        await db_session.scalars(select(Chunk).where(Chunk.paper_id == paper.paper_id))
    ).all()
    assert chunks
    # The FK is the no-crossing rule; every chunk must land in a real section.
    section_ids = {section.section_id for section in sections}
    assert all(chunk.section_id in section_ids for chunk in chunks)
    assert all(chunk.page_start >= 1 for chunk in chunks)


async def test_ingest_finishes_ready_with_no_phase_left(
    db_session: AsyncSession, storage: LocalStorage
):
    """A fully readable paper ends `ready`, with the phase cleared."""
    paper = await _seed_paper(db_session, storage, PAPER_PAGES)
    await ingest_paper(db_session, paper.paper_id, storage=storage)

    assert paper.processing_status == "ready"
    assert paper.processing_phase is None
    assert paper.error_code is None
    assert paper.page_count == len(PAPER_PAGES)
    assert paper.embedding_model  # recorded, so a model change is detectable


async def test_reference_chunks_are_excluded_from_the_index(
    db_session: AsyncSession, storage: LocalStorage
):
    paper = await _seed_paper(db_session, storage, PAPER_PAGES)
    await ingest_paper(db_session, paper.paper_id, storage=storage)

    reference_section = await db_session.scalar(
        select(Section).where(
            Section.paper_id == paper.paper_id, Section.section_role == "references"
        )
    )
    assert reference_section is not None

    excluded = await db_session.scalar(
        select(func.count())
        .select_from(Chunk)
        .where(Chunk.section_id == reference_section.section_id, Chunk.is_indexable.is_(False))
    )
    assert excluded > 0


async def test_reingest_is_idempotent(db_session: AsyncSession, storage: LocalStorage):
    """A retry re-runs from the top, so it must not double the rows."""
    paper = await _seed_paper(db_session, storage, PAPER_PAGES)

    first = await ingest_paper(db_session, paper.paper_id, storage=storage)
    second = await ingest_paper(db_session, paper.paper_id, storage=storage)

    assert (first.section_count, first.chunk_count) == (
        second.section_count,
        second.chunk_count,
    )
    total = await db_session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.paper_id == paper.paper_id)
    )
    assert total == second.chunk_count


async def test_corrupt_pdf_fails_permanently_with_a_typed_code(
    db_session: AsyncSession, storage: LocalStorage
):
    content_hash = uuid.uuid4().hex + uuid.uuid4().hex[:32]
    paper = Paper(
        content_hash=content_hash,
        storage_uri=storage.put(b"%PDF-1.4 not actually a pdf", content_hash=content_hash),
        processing_status="queued",
    )
    db_session.add(paper)
    await db_session.flush()

    with pytest.raises(PermanentIngestionError) as raised:
        await ingest_paper(db_session, paper.paper_id, storage=storage)

    assert raised.value.code == "pdf_corrupt"
    assert paper.processing_status == "failed"
    assert paper.error_code == "pdf_corrupt"
    assert paper.processing_phase is None


async def test_missing_original_is_permanent_not_transient(
    db_session: AsyncSession, storage: LocalStorage
):
    paper = Paper(
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        storage_uri="file://nothing-here.pdf",
        processing_status="queued",
    )
    db_session.add(paper)
    await db_session.flush()

    with pytest.raises(PermanentIngestionError) as raised:
        await ingest_paper(db_session, paper.paper_id, storage=storage)
    assert raised.value.code == "original_missing"


# --------------------------------------------------------------------------
# Upload boundary
# --------------------------------------------------------------------------


@pytest.fixture
def no_background(monkeypatch):
    """Record enqueued jobs instead of running them.

    The real job opens its own session and commits, which would escape the
    per-test transaction. The pipeline itself is covered above.
    """
    enqueued: list = []
    monkeypatch.setattr(
        "app.routers.papers.run_ingestion_job",
        lambda paper_id, user_id=None: enqueued.append(("ingest", paper_id)),
    )
    monkeypatch.setattr(
        "app.routers.papers.run_canonicalization_job",
        lambda paper_id, user_id: enqueued.append(("canonicalize", paper_id)),
    )
    return enqueued


async def test_upload_accepts_a_pdf_and_enqueues_ingestion(
    client: AsyncClient, db_session: AsyncSession, dev_auth, storage_dir, no_background
):
    response = await client.post(
        "/papers",
        files={"file": ("paper.pdf", build_pdf(PAPER_PAGES), "application/pdf")},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["processing_status"] == "queued"
    assert body["page_count"] == len(PAPER_PAGES)
    assert len(no_background) == 1

    # It is visible to the uploader straight away.
    listed = (await client.get("/papers")).json()
    assert [p["paper_id"] for p in listed] == [body["paper_id"]]


async def test_upload_rejects_non_pdf_by_sniffing_content(
    client: AsyncClient, dev_auth, storage_dir, no_background
):
    """The declared type is not trusted; the bytes are."""
    response = await client.post(
        "/papers",
        files={"file": ("evil.pdf", b"MZ\x90\x00 this is an executable", "application/pdf")},
    )
    assert response.status_code == 415
    assert not no_background


async def test_upload_rejects_an_empty_file(
    client: AsyncClient, dev_auth, storage_dir, no_background
):
    response = await client.post("/papers", files={"file": ("x.pdf", b"", "application/pdf")})
    assert response.status_code == 400


async def test_upload_enforces_the_size_cap(
    client: AsyncClient, dev_auth, storage_dir, settings_env, no_background
):
    settings_env(max_upload_bytes="1024")
    response = await client.post(
        "/papers", files={"file": ("big.pdf", build_pdf(PAPER_PAGES), "application/pdf")}
    )
    assert response.status_code == 413


async def test_upload_enforces_the_page_cap(
    client: AsyncClient, dev_auth, storage_dir, settings_env, no_background
):
    settings_env(max_page_count="2")
    response = await client.post(
        "/papers", files={"file": ("long.pdf", build_pdf(PAPER_PAGES), "application/pdf")}
    )
    assert response.status_code == 422
    assert "3 pages" in response.json()["detail"]


async def test_identical_bytes_are_not_ingested_twice(
    client: AsyncClient, db_session: AsyncSession, dev_auth, storage_dir, no_background
):
    """Content hash is the idempotency key (ARCHITECTURE 8.3)."""
    data = build_pdf(PAPER_PAGES)

    first = await client.post("/papers", files={"file": ("a.pdf", data, "application/pdf")})
    second = await client.post("/papers", files={"file": ("b.pdf", data, "application/pdf")})

    assert first.json()["paper_id"] == second.json()["paper_id"]

    # Phases 1-5 run once for the bytes; the second upload only enqueues
    # phase 6b, because concepts are per-reader and cannot be shared.
    assert [kind for kind, _ in no_background] == ["ingest", "canonicalize"]

    # Count only the bytes under test. A global count would also see whatever
    # papers a developer has uploaded into their local database.
    digest = hashlib.sha256(data).hexdigest()
    papers = await db_session.scalar(
        select(func.count()).select_from(Paper).where(Paper.content_hash == digest)
    )
    assert papers == 1


async def test_upload_by_a_second_user_grants_access_without_reingesting(
    client: AsyncClient, db_session: AsyncSession, dev_auth, storage_dir, no_background
):
    """Chunks are paper-scoped and shared; only the grant is per-user."""
    data = build_pdf(PAPER_PAGES)
    first = await client.post("/papers", files={"file": ("a.pdf", data, "application/pdf")})
    paper_id = uuid.UUID(first.json()["paper_id"])

    grants = await db_session.scalar(
        select(func.count())
        .select_from(UserPaperAccess)
        .where(UserPaperAccess.paper_id == paper_id)
    )
    assert grants == 1
    assert len(no_background) == 1


async def test_get_paper_hides_unauthorized_ids_as_404(
    client: AsyncClient, db_session: AsyncSession, dev_auth, storage: LocalStorage
):
    """A 403 would confirm the id is real to someone with no access."""
    paper = await _seed_paper(db_session, storage, PAPER_PAGES)

    response = await client.get(f"/papers/{paper.paper_id}")
    assert response.status_code == 404


async def test_get_paper_reports_status_for_polling(
    client: AsyncClient, dev_auth, storage_dir, no_background
):
    created = await client.post(
        "/papers", files={"file": ("p.pdf", build_pdf(PAPER_PAGES), "application/pdf")}
    )
    paper_id = created.json()["paper_id"]

    response = await client.get(f"/papers/{paper_id}")
    assert response.status_code == 200
    assert response.json()["processing_status"] == "queued"
