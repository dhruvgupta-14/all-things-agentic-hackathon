"""The Cloud Tasks push route: who may call it, and what the status code means.

Two independent things are being guarded here.

**Reachability.** The service is public — it serves the SPA and the browser API
from the same origin — so Cloud Run's own IAM cannot protect this route. If the
OIDC check is wrong, `/internal/ingest` is an unauthenticated endpoint that
re-ingests any paper id it is handed.

**The retry contract** (ARCHITECTURE 8.2). Cloud Tasks reads the status code
and nothing else. 503 means try again; 200 means stop. Getting it backwards is
invisible in every log: a corrupt PDF quietly retried five times, or a Vertex
timeout quietly abandoned.
"""

import uuid
from unittest import mock

import pytest
from httpx import AsyncClient

from app.auth.oidc import (
    OidcConfigurationError,
    require_cloud_tasks,
    verify_push_token,
)
from app.main import app

PATH = "/internal/ingest"
SERVICE_ACCOUNT = "paper-companion@proj.iam.gserviceaccount.com"
BASE_URL = "https://companion-abc.a.run.app"

PUSH_ENV = {
    "SERVICE_ACCOUNT_EMAIL": SERVICE_ACCOUNT,
    "SERVICE_BASE_URL": BASE_URL,
}


def body(job="ingest", user_id=None):
    return {
        "job": job,
        "paper_id": str(uuid.uuid4()),
        "user_id": str(user_id or uuid.uuid4()),
    }


@pytest.fixture
def authorized():
    """Treat the caller as a verified push, so the tests below are about the
    route rather than about token verification (covered separately)."""
    app.dependency_overrides[require_cloud_tasks] = lambda: SERVICE_ACCOUNT
    yield
    app.dependency_overrides.pop(require_cloud_tasks, None)


# --------------------------------------------------------------------------
# Reachability
# --------------------------------------------------------------------------


async def test_an_unauthenticated_push_is_refused(client: AsyncClient, signed_in):
    """`signed_in` is active, which is the point: the local bypass authenticates
    the *browser* API. It must not open this route, or every local run would be
    exercising a version of the endpoint that does not exist in production."""
    response = await client.post(PATH, json=body())

    assert response.status_code == 401


async def test_a_firebase_user_token_does_not_open_the_internal_route(
    client: AsyncClient, signed_in, settings_env
):
    """A signed-in user presenting their own perfectly valid ID token is still
    not Cloud Tasks."""
    settings_env(**PUSH_ENV)

    with mock.patch(
        "app.auth.oidc.verify_push_token", side_effect=PermissionError("not valid")
    ):
        response = await client.post(
            PATH, json=body(), headers={"Authorization": "Bearer a-firebase-id-token"}
        )

    assert response.status_code == 401


def test_the_audience_comes_from_configuration_not_from_the_request(settings_env):
    """`Host` is client-controlled. Verifying a token against an audience the
    caller chose is not a check at all."""
    from app.config import get_settings

    settings_env(**PUSH_ENV)

    with mock.patch("google.oauth2.id_token.verify_oauth2_token") as verify:
        verify.return_value = {"email": SERVICE_ACCOUNT, "email_verified": True}
        verify_push_token("token", get_settings())

    assert verify.call_args.kwargs["audience"] == BASE_URL


def test_a_valid_google_token_from_another_identity_is_refused(settings_env):
    """Every service account in the project can mint a correctly signed token
    for this audience. Only ours may run ingestion."""
    from app.config import get_settings

    settings_env(**PUSH_ENV)

    with mock.patch("google.oauth2.id_token.verify_oauth2_token") as verify:
        verify.return_value = {
            "email": "someone-else@proj.iam.gserviceaccount.com",
            "email_verified": True,
        }
        with pytest.raises(PermissionError):
            verify_push_token("token", get_settings())


def test_an_unverified_email_claim_is_refused(settings_env):
    from app.config import get_settings

    settings_env(**PUSH_ENV)

    with mock.patch("google.oauth2.id_token.verify_oauth2_token") as verify:
        verify.return_value = {"email": SERVICE_ACCOUNT, "email_verified": False}
        with pytest.raises(PermissionError):
            verify_push_token("token", get_settings())


def test_missing_configuration_never_degrades_into_no_check():
    """Missing configuration must not become "no audience to verify against".

    Settings makes this unreachable in a running process — both fields are
    required, and an http SERVICE_BASE_URL is refused at startup. The guard
    stays because this function takes its settings as an argument, and the one
    thing it must never do is treat absent configuration as permission.
    """
    from types import SimpleNamespace

    unconfigured = SimpleNamespace(service_base_url=None, service_account_email=None)

    with pytest.raises(OidcConfigurationError):
        verify_push_token("token", unconfigured)


def test_a_non_https_service_url_is_refused_at_startup():
    """It is the OIDC audience. An http value would fail verification on every
    push, which surfaces as papers that never leave `queued`."""
    from pydantic import ValidationError

    from app.config import Settings

    with pytest.raises(ValidationError, match="https"):
        Settings(SERVICE_BASE_URL="http://not-secure.example")


# --------------------------------------------------------------------------
# The retry contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        # Retried with backoff: a Vertex timeout deserves another attempt.
        ("retry", 503),
        # Not retried. The paper row already carries the typed reason, and
        # four more attempts at a corrupt PDF would bury it.
        ("failed", 200),
        ("ok", 200),
    ],
)
async def test_the_status_code_carries_the_retry_decision(
    client: AsyncClient, authorized, outcome, expected
):
    async def job(paper_id, user_id):
        return outcome

    with mock.patch.dict("app.routers.internal.JOBS", {"ingest": job}):
        response = await client.post(PATH, json=body())

    assert response.status_code == expected
    assert response.json()["status"] == outcome


async def test_an_unknown_job_name_is_rejected_before_it_reaches_a_lookup(
    client: AsyncClient, authorized
):
    """A `KeyError` here would surface as a 500, which Cloud Tasks reads as
    transient and retries five times over."""
    response = await client.post(PATH, json=body(job="rm-rf"))

    assert response.status_code == 422


async def test_a_canonicalize_push_without_a_reader_is_not_retried(
    client: AsyncClient, authorized
):
    """Phase 6b canonicalizes into one reader's graph; there is nothing to do
    without one. Permanent, because a retry re-reads the same payload."""
    payload = body(job="canonicalize")
    payload["user_id"] = None

    response = await client.post(PATH, json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == "failed"


async def test_the_job_receives_the_ids_from_the_signed_payload(
    client: AsyncClient, authorized
):
    """`user_id` is trustworthy here in a way it never is on a browser route:
    the body was written by `dispatch` at upload time and signed by the queue,
    so it is the uploader's id rather than one the caller chose."""
    seen = {}

    async def job(paper_id, user_id):
        seen["paper_id"] = paper_id
        seen["user_id"] = user_id
        return "ok"

    payload = body()
    with mock.patch.dict("app.routers.internal.JOBS", {"ingest": job}):
        await client.post(PATH, json=payload)

    assert seen == {
        "paper_id": uuid.UUID(payload["paper_id"]),
        "user_id": uuid.UUID(payload["user_id"]),
    }
