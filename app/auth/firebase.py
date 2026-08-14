"""Firebase ID token verification.

This module is the only place the system learns who is calling. Identity never
comes from a request body, a query parameter, or a tool argument — see
ARCHITECTURE section 9.1.
"""

import logging
import threading
from dataclasses import dataclass

import firebase_admin
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials

from app.config import get_settings

logger = logging.getLogger(__name__)

_app: firebase_admin.App | None = None
_lock = threading.Lock()


class AuthConfigurationError(RuntimeError):
    """The deployment cannot verify tokens at all. Not the caller's fault."""


class InvalidTokenError(Exception):
    """The presented token is missing, malformed, expired, or not ours."""


@dataclass(frozen=True, slots=True)
class VerifiedToken:
    """The subset of Firebase claims this system is willing to act on."""

    subject: str
    email: str | None
    display_name: str | None


def _get_app() -> firebase_admin.App:
    """Initialize the Firebase app once, on first use."""
    global _app
    if _app is not None:
        return _app

    with _lock:
        if _app is not None:
            return _app

        settings = get_settings()
        if not settings.firebase_project_id:
            raise AuthConfigurationError(
                "FIREBASE_PROJECT_ID is not set. Without a pinned project, "
                "verification would accept tokens minted by any Firebase project."
            )

        # Application Default Credentials: the metadata server on Cloud Run,
        # GOOGLE_APPLICATION_CREDENTIALS locally. No long-lived key in config.
        try:
            cred = credentials.ApplicationDefault()
        except Exception as exc:  # pragma: no cover - environment dependent
            raise AuthConfigurationError(
                f"No application default credentials available: {exc}"
            ) from exc

        _app = firebase_admin.initialize_app(
            cred, {"projectId": settings.firebase_project_id}
        )
        logger.info(
            "firebase initialized", extra={"project_id": settings.firebase_project_id}
        )
        return _app


def verify_id_token(raw_token: str) -> VerifiedToken:
    """Verify a Firebase ID token and return the claims we trust.

    Raises:
        AuthConfigurationError: the deployment cannot verify tokens (503).
        InvalidTokenError: the token is not acceptable (401).
    """
    app = _get_app()

    try:
        claims = firebase_auth.verify_id_token(raw_token, app=app)
    except firebase_auth.ExpiredIdTokenError as exc:
        raise InvalidTokenError("Token has expired.") from exc
    except firebase_auth.RevokedIdTokenError as exc:
        raise InvalidTokenError("Token has been revoked.") from exc
    except (firebase_auth.InvalidIdTokenError, ValueError) as exc:
        # Deliberately vague to the caller; the detail goes to the logs only.
        logger.warning("rejected id token: %s", exc)
        raise InvalidTokenError("Token is not valid.") from exc

    subject = claims.get("uid") or claims.get("sub")
    if not subject:
        raise InvalidTokenError("Token carries no subject.")

    email = claims.get("email")
    return VerifiedToken(
        subject=subject,
        email=email.lower() if isinstance(email, str) else None,
        display_name=claims.get("name"),
    )
