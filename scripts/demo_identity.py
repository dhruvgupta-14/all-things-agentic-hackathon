"""Who the demo data belongs to.

It used to be `local-dev-user`, the subject the development auth bypass
authenticated every unauthenticated request as. With the bypass gone that
identity does not exist anywhere: there is no way to become it, and a judge
signing in reaches their own Firebase subject.

So the demo account is a real Firebase account, and its `users.auth_subject` is
the UID Firebase assigns it. Resolving the UID from the email — rather than
pasting it into four scripts — keeps them pointed at the same reader even if
the account is ever recreated.

Needs Firebase Admin over ADC: `gcloud auth application-default login`.
"""

from __future__ import annotations

from functools import lru_cache

DEMO_EMAIL = "judge@research-companion.demo"


@lru_cache(maxsize=4)
def resolve_uid(email: str = DEMO_EMAIL) -> str:
    """The Firebase UID for this account, or raise with something actionable."""
    from firebase_admin import auth as firebase_auth

    from app.auth.firebase import _get_app

    try:
        return firebase_auth.get_user_by_email(email, app=_get_app()).uid
    except firebase_auth.UserNotFoundError as exc:
        raise SystemExit(
            f"No Firebase account for {email!r}.\n"
            "Create it in the Firebase console (Authentication -> Users), or\n"
            "pass --email for an account that does exist."
        ) from exc
