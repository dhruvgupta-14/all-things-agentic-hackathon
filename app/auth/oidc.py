"""Verifying that a request really came from our Cloud Tasks queue.

`/internal/ingest` cannot be protected by Cloud Run's own IAM. The service is
public — it serves the SPA and the browser API from the same origin — so
"internal" has to be enforced in the application. Without this module the route
is an unauthenticated endpoint that ingests any paper id it is handed.

Cloud Tasks signs each push with an OIDC token minted as a service account we
name. Three things are then checked, and all three matter:

  * the signature and issuer, so the token is Google's and not forged;
  * the audience, pinned to this deployment's own URL, so a token minted for
    some other service cannot be replayed here;
  * the service account email, so a token from any other identity in the
    project — including a developer's own — is refused.

The audience comes from configuration rather than from the request. Deriving it
from `Host` or `X-Forwarded-Host` would let a caller choose the value it is
checked against, which is not a check at all.
"""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Google's clock and this deployment's may disagree by a few seconds. The same
# tolerance the Firebase path uses, and for the same reason — see
# app/auth/firebase.py, where a host 41 seconds behind rejected every token as
# "used too early". It relaxes only the not-before check.
CLOCK_SKEW_TOLERANCE_SECONDS = 60

_bearer = HTTPBearer(auto_error=False)

_REFUSED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authorized.",
    headers={"WWW-Authenticate": "Bearer"},
)


class OidcConfigurationError(RuntimeError):
    """The deployment cannot verify pushes at all. Not the caller's fault."""


def verify_push_token(raw_token: str, settings: Settings) -> str:
    """Return the verified caller's service-account email, or raise.

    Raises:
        OidcConfigurationError: nothing to verify against (503).
        PermissionError: the token is absent, invalid, or another identity.
    """
    if not (settings.service_base_url and settings.service_account_email):
        raise OidcConfigurationError(
            "SERVICE_BASE_URL and SERVICE_ACCOUNT_EMAIL must both be set before "
            "/internal/ingest can authenticate a push."
        )

    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token as google_id_token

    try:
        claims = google_id_token.verify_oauth2_token(
            raw_token,
            google_requests.Request(),
            audience=settings.service_base_url.rstrip("/"),
            clock_skew_in_seconds=CLOCK_SKEW_TOLERANCE_SECONDS,
        )
    except Exception as exc:
        # Detail to the logs, not to the caller.
        logger.warning("rejected push token: %s", exc)
        raise PermissionError("Token is not valid.") from exc

    email = claims.get("email")
    if email != settings.service_account_email:
        # A valid Google token from the wrong identity. Worth a warning rather
        # than an info line: it is either a misconfigured queue or someone
        # probing the route with credentials they do have.
        logger.warning("push token from unexpected identity: %r", email)
        raise PermissionError("Token is not valid.")

    if not claims.get("email_verified"):
        raise PermissionError("Token is not valid.")

    return email


async def require_cloud_tasks(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """FastAPI dependency guarding the internal push route."""
    if credentials is None or not credentials.credentials:
        raise _REFUSED

    settings = get_settings()
    try:
        # Verification fetches and caches Google's signing certificates, so the
        # first call does network I/O. In a thread: blocking the event loop
        # would stall every in-flight SSE stream on the instance.
        return await run_in_threadpool(
            verify_push_token, credentials.credentials, settings
        )
    except OidcConfigurationError as exc:
        logger.error("push auth misconfigured: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Internal task authentication is not configured.",
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
