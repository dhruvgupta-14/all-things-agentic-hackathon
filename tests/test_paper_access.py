"""The authorization boundary: possessing a paper_id grants nothing.

Only a live row in `user_paper_access` makes a paper visible
(ARCHITECTURE 4.2, 9.1).
"""

import uuid
from datetime import UTC, datetime

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Paper, User, UserPaperAccess


async def _make_paper(session: AsyncSession, title: str) -> Paper:
    paper = Paper(
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        storage_uri=f"gs://test-bucket/{uuid.uuid4()}.pdf",
        title=title,
        processing_status="ready",
    )
    session.add(paper)
    await session.flush()
    return paper


async def _current_user(session: AsyncSession, subject: str) -> User:
    user = await session.scalar(select(User).where(User.auth_subject == subject))
    assert user is not None, "the bypass should have provisioned this user"
    return user


async def test_granted_paper_is_listed(
    client: AsyncClient, db_session: AsyncSession, dev_auth: str
):
    await client.get("/me")  # provision the caller
    user = await _current_user(db_session, dev_auth)
    paper = await _make_paper(db_session, "Attention Is All You Need")

    db_session.add(
        UserPaperAccess(
            user_id=user.user_id, paper_id=paper.paper_id, nickname="transformers"
        )
    )
    await db_session.flush()

    body = (await client.get("/papers")).json()
    assert [p["title"] for p in body] == ["Attention Is All You Need"]
    assert body[0]["nickname"] == "transformers"
    assert body[0]["processing_status"] == "ready"


async def test_ungranted_paper_is_invisible(
    client: AsyncClient, db_session: AsyncSession, dev_auth: str
):
    """The paper exists and is ready; no grant means it does not appear."""
    await client.get("/me")
    await _make_paper(db_session, "Someone Else's Paper")

    assert (await client.get("/papers")).json() == []


async def test_revoked_grant_hides_the_paper_immediately(
    client: AsyncClient, db_session: AsyncSession, dev_auth: str
):
    await client.get("/me")
    user = await _current_user(db_session, dev_auth)
    paper = await _make_paper(db_session, "Revoked Paper")

    grant = UserPaperAccess(user_id=user.user_id, paper_id=paper.paper_id)
    db_session.add(grant)
    await db_session.flush()
    assert len((await client.get("/papers")).json()) == 1

    grant.revoked_at = datetime.now(UTC)
    await db_session.flush()

    # Read-time authorization, so revocation needs no cache invalidation.
    assert (await client.get("/papers")).json() == []


async def test_another_users_grant_does_not_leak(
    client: AsyncClient, db_session: AsyncSession, dev_auth: str
):
    await client.get("/me")
    paper = await _make_paper(db_session, "Private To Someone Else")

    other = User(auth_subject=f"other-{uuid.uuid4()}")
    db_session.add(other)
    await db_session.flush()
    db_session.add(UserPaperAccess(user_id=other.user_id, paper_id=paper.paper_id))
    await db_session.flush()

    assert (await client.get("/papers")).json() == []
