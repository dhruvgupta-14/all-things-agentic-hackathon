"""The authentication dependency: bearer token in, `Principal` out.

`Principal.user_id` is our own UUID, resolved from the Firebase subject. Every
downstream authorization check keys off it, so nothing below this layer ever
sees a client-supplied identifier.
"""

import logging
import uuid
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.auth.firebase import (
    AuthConfigurationError,
    InvalidTokenError,
    VerifiedToken,
    verify_id_token,
)
from app.config import get_settings
from app.db.base import get_db
from app.db.models import User

logger = logging.getLogger(__name__)

# auto_error=False so a missing header produces our own 401 with a
# WWW-Authenticate challenge rather than FastAPI's bare 403.
_bearer = HTTPBearer(auto_error=False)

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated.",
    headers={"WWW-Authenticate": "Bearer"},
)


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller."""

    user_id: uuid.UUID
    auth_subject: str
    email: str | None


async def _resolve_user(session: AsyncSession, token: VerifiedToken) -> User:
    """Find the user for this Firebase subject, creating them on first login.

    Two round trips in the common case is one too many, but the alternative —
    an upsert that also writes `email` — can collide with the partial unique
    index on `users.email` and turn a routine request into a 500. The
    `auth_subject` lookup is the highest-frequency query in the system and is
    served by a unique index.
    """
    existing = await session.scalar(
        select(User).where(User.auth_subject == token.subject)
    )
    if existing is not None:
        return existing

    stmt = (
        insert(User)
        .values(
            auth_subject=token.subject,
            email=token.email,
            display_name=token.display_name,
        )
        .on_conflict_do_nothing(index_elements=["auth_subject"])
        .returning(User)
    )
    created = await session.scalar(stmt)

    if created is None:
        # A concurrent request won the race; re-read its row.
        created = await session.scalar(
            select(User).where(User.auth_subject == token.subject)
        )
        if created is None:  # pragma: no cover - should be unreachable
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not resolve user account.",
            )
    else:
        await session.commit()
        logger.info("provisioned user", extra={"user_id": str(created.user_id)})

    return created


def _dev_bypass_token() -> VerifiedToken | None:
    """The local-development escape hatch, or None when it is not in play.

    Two independent conditions must hold: an explicitly configured subject,
    and `app_env == "local"`. A subject configured anywhere else raises rather
    than being quietly ignored — silently disregarding a credential-shaped
    setting is how a bypass survives to production unnoticed.
    """
    settings = get_settings()
    subject = settings.auth_dev_bypass_subject
    if not subject:
        return None

    if settings.app_env != "local":
        raise AuthConfigurationError(
            f"AUTH_DEV_BYPASS_SUBJECT is set but APP_ENV is {settings.app_env!r}. "
            "The development auth bypass is only permitted when APP_ENV=local."
        )

    logger.warning(
        "AUTH BYPASS ACTIVE — request authenticated as %r without a token. "
        "This must never be enabled outside local development.",
        subject,
    )
    return VerifiedToken(
        subject=subject,
        email=f"{subject}@local.invalid",
        display_name="Local Development User",
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: AsyncSession = Depends(get_db),
) -> Principal:
    try:
        bypass = _dev_bypass_token()
    except AuthConfigurationError as exc:
        logger.error("auth misconfigured: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is misconfigured on this deployment.",
        ) from exc

    if bypass is not None:
        user = await _resolve_user(session, bypass)
        return Principal(
            user_id=user.user_id,
            auth_subject=user.auth_subject,
            email=user.email,
        )

    if credentials is None or not credentials.credentials:
        raise _UNAUTHENTICATED

    try:
        token = await run_in_threadpool(verify_id_token, credentials.credentials)
    except AuthConfigurationError as exc:
        # A deployment fault, not a bad request. Do not report it as a 401 —
        # that would send the client into a pointless re-login loop.
        logger.error("auth misconfigured: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured on this deployment.",
        ) from exc
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user = await _resolve_user(session, token)
    return Principal(
        user_id=user.user_id,
        auth_subject=user.auth_subject,
        email=user.email,
    )
