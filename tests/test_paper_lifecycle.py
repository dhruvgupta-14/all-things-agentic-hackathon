"""Recovering a failed paper, and removing one from a library.

Both were reported from a real deployment, and both are the same shape: an
action the reader would obviously try, which silently did nothing.

**Re-uploading a failed paper.** Dedupe matched on content hash, granted
access, enqueued canonicalization and returned the same `failed` row. A paper
that failed for a transient reason — a Vertex outage, an enqueue refused by a
missing IAM role — was unrecoverable by the one action anybody would think of,
and the reader had no way to tell a permanent failure from a temporary one.

**Removing a paper.** There was no route at all.
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Paper, UserPaperAccess
from tests.conftest import build_pdf
from tests.fakes import InMemoryStorage

PAGES = [
    "Mergers And Acquisitions\nAbstract\nWe study acquisition effects on exports.",
    "2 Method\nWe compare acquiring firms against a matched control group.",
]


@pytest.fixture
def storage(storage_backend) -> InMemoryStorage:
    return storage_backend


@pytest.fixture
def dispatched(monkeypatch):
    """Record what would have been enqueued, without enqueuing it."""
    jobs: list[tuple[str, uuid.UUID]] = []

    async def fake_dispatch(job, paper_id, user_id):
        jobs.append((job, paper_id))
        return "tasks/1"

    monkeypatch.setattr("app.routers.papers.dispatch", fake_dispatch)
    return jobs


async def _upload(client: AsyncClient, data: bytes, name: str = "paper.pdf"):
    return await client.post(
        "/api/papers", files={"file": (name, data, "application/pdf")}
    )


# --------------------------------------------------------------------------
# Re-uploading a failed paper
# --------------------------------------------------------------------------


async def test_re_uploading_a_failed_paper_ingests_it_again(
    client: AsyncClient, db_session: AsyncSession, signed_in, storage, dispatched
):
    data = build_pdf(PAGES)
    first = await _upload(client, data)
    paper_id = uuid.UUID(first.json()["paper_id"])

    # However it failed — this one stands in for a refused enqueue.
    paper = await db_session.get(Paper, paper_id)
    paper.processing_status = "failed"
    paper.error_code = "enqueue_failed"
    await db_session.flush()
    dispatched.clear()

    second = await _upload(client, data, name="paper-again.pdf")

    assert second.status_code == 202
    assert second.json()["processing_status"] == "queued"
    assert dispatched == [("ingest", paper_id)], "it must re-run ingestion, not phase 6b"

    await db_session.refresh(paper)
    assert paper.processing_status == "queued"
    # The old reason must go with the old outcome, or the UI reports a failure
    # for a paper that is currently processing.
    assert paper.error_code is None
    assert paper.processing_phase is None


async def test_a_healthy_paper_is_not_re_ingested(
    client: AsyncClient, db_session: AsyncSession, signed_in, storage, dispatched
):
    """The narrowness is the point. Re-uploading a `ready` paper must still
    skip phases 1-5 — they are paper-scoped and shared, and re-running them
    would re-embed a whole corpus to no purpose."""
    data = build_pdf(PAGES)
    first = await _upload(client, data)
    paper_id = uuid.UUID(first.json()["paper_id"])

    paper = await db_session.get(Paper, paper_id)
    paper.processing_status = "ready"
    await db_session.flush()
    dispatched.clear()

    await _upload(client, data)

    assert dispatched == [("canonicalize", paper_id)]


async def test_re_uploading_a_failed_paper_does_not_duplicate_the_row(
    client: AsyncClient, db_session: AsyncSession, signed_in, storage, dispatched
):
    """Content hash is unique; a second row would violate it, and a second
    *paper* would split the reader's library in two."""
    data = build_pdf(PAGES)
    await _upload(client, data)

    paper = await db_session.scalar(
        select(Paper).where(Paper.original_filename == "paper.pdf")
    )
    paper.processing_status = "failed"
    await db_session.flush()

    await _upload(client, data)

    count = await db_session.scalar(
        select(func.count()).select_from(Paper).where(Paper.content_hash == paper.content_hash)
    )
    assert count == 1


# --------------------------------------------------------------------------
# Removing a paper
# --------------------------------------------------------------------------


async def test_removing_a_paper_takes_it_out_of_the_library(
    client: AsyncClient, signed_in, storage, dispatched
):
    created = await _upload(client, build_pdf(PAGES))
    paper_id = created.json()["paper_id"]

    response = await client.delete(f"/api/papers/{paper_id}")

    assert response.status_code == 204
    assert await (await client.get("/api/papers")).aread() == b"[]"
    # And it is gone from the read route too, not merely hidden from the list.
    assert (await client.get(f"/api/papers/{paper_id}")).status_code == 404


async def test_removal_revokes_rather_than_deletes(
    client: AsyncClient, db_session: AsyncSession, signed_in, storage, dispatched
):
    """Papers are shared by content hash. Deleting the row because one reader
    tidied up would empty another reader's library — possibly mid-session."""
    created = await _upload(client, build_pdf(PAGES))
    paper_id = uuid.UUID(created.json()["paper_id"])

    await client.delete(f"/api/papers/{paper_id}")

    assert await db_session.get(Paper, paper_id) is not None
    grant = await db_session.scalar(
        select(UserPaperAccess).where(UserPaperAccess.paper_id == paper_id)
    )
    assert grant is not None and grant.revoked_at is not None


async def test_re_uploading_a_removed_paper_brings_it_back(
    client: AsyncClient, signed_in, storage, dispatched
):
    """Removal is reversible, which is what makes it safe to offer without
    ceremony."""
    data = build_pdf(PAGES)
    created = await _upload(client, data)
    await client.delete(f"/api/papers/{created.json()['paper_id']}")

    again = await _upload(client, data)

    assert again.status_code == 202
    listed = (await client.get("/api/papers")).json()
    assert [p["paper_id"] for p in listed] == [created.json()["paper_id"]]


async def test_removing_someone_elses_paper_is_a_404(
    client: AsyncClient, db_session: AsyncSession, signed_in, storage
):
    """Possessing an id grants nothing, and a 403 would confirm the id is real."""
    content_hash = uuid.uuid4().hex + uuid.uuid4().hex[:32]
    stranger_paper = Paper(
        content_hash=content_hash,
        storage_uri=storage.put(build_pdf(PAGES), content_hash=content_hash),
        processing_status="ready",
    )
    db_session.add(stranger_paper)
    await db_session.flush()

    response = await client.delete(f"/api/papers/{stranger_paper.paper_id}")

    assert response.status_code == 404


async def test_removing_twice_is_a_404_not_an_error(
    client: AsyncClient, signed_in, storage, dispatched
):
    created = await _upload(client, build_pdf(PAGES))
    paper_id = created.json()["paper_id"]

    assert (await client.delete(f"/api/papers/{paper_id}")).status_code == 204
    assert (await client.delete(f"/api/papers/{paper_id}")).status_code == 404
