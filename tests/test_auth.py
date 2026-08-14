"""The authentication boundary.

Identity must come from a verified token (or the explicitly gated local
bypass) and nowhere else — ARCHITECTURE section 9.1.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User

PROTECTED_ROUTES = ["/me", "/papers"]


async def test_health_needs_no_token(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


@pytest.mark.parametrize("route", PROTECTED_ROUTES)
async def test_missing_token_is_rejected(client: AsyncClient, route: str, settings_env):
    settings_env(auth_dev_bypass_subject=None)
    response = await client.get(route)
    assert response.status_code == 401
    # Without the challenge header a browser client cannot know how to retry.
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("route", PROTECTED_ROUTES)
async def test_unconfigured_firebase_is_503_not_401(
    client: AsyncClient, route: str, settings_env
):
    """A deployment fault must not masquerade as a credential problem.

    Returning 401 here would send clients into a pointless re-login loop
    against a server that cannot verify anything.
    """
    settings_env(auth_dev_bypass_subject=None, firebase_project_id=None)
    response = await client.get(route, headers={"Authorization": "Bearer whatever"})
    assert response.status_code == 503


async def test_bypass_refused_outside_local(client: AsyncClient, settings_env):
    """A bypass subject set in a deployed environment fails closed."""
    settings_env(app_env="production", auth_dev_bypass_subject="someone")
    response = await client.get("/me")
    assert response.status_code == 503
    assert response.json()["detail"] == "Authentication is misconfigured on this deployment."


async def test_bypass_authenticates_and_provisions_user(
    client: AsyncClient, db_session: AsyncSession, dev_auth: str
):
    response = await client.get("/me")
    assert response.status_code == 200

    body = response.json()
    assert body["auth_subject"] == dev_auth
    assert body["email"] == f"{dev_auth}@local.invalid"
    assert body["user_id"]

    stored = await db_session.scalar(
        select(User).where(User.auth_subject == dev_auth)
    )
    assert stored is not None
    assert str(stored.user_id) == body["user_id"]


async def test_repeat_login_reuses_the_same_user(
    client: AsyncClient, db_session: AsyncSession, dev_auth: str
):
    """Provisioning is once-per-subject, not once-per-request."""
    first = await client.get("/me")
    second = await client.get("/me")

    assert first.json()["user_id"] == second.json()["user_id"]

    count = await db_session.scalar(
        select(func.count()).select_from(User).where(User.auth_subject == dev_auth)
    )
    assert count == 1


async def test_papers_is_empty_for_a_new_user(client: AsyncClient, dev_auth: str):
    response = await client.get("/papers")
    assert response.status_code == 200
    assert response.json() == []
