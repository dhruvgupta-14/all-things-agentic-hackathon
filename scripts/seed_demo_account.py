"""Give the published demo account everything the demo needs.

A judge signs in as a real Firebase account and must land in a library that
already has papers, concepts and enough learner memory for the cross-paper
callback to fire. A fresh account has none of that.

This closes that gap:

    PYTHONPATH=. python scripts/seed_demo_account.py
    PYTHONPATH=. python scripts/seed_demo_account.py --email someone@else
    PYTHONPATH=. python scripts/seed_demo_account.py --check   # verify only

The papers themselves are not re-ingested. Phases 1-5 are paper-scoped and
shared by content hash (ARCHITECTURE 8.4), so a second reader of the same PDF
skips to phase 6b: their own concepts, canonicalized against their own memory.
That costs one batched adjudication call per paper, not 113 embeddings.

Requires Firebase Admin access over ADC to resolve the account's UID, which is
what the `users.auth_subject` row has to match. `gcloud auth
application-default login` is enough.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import select

from app.config import get_settings
from app.db.base import async_session_factory
from app.db.models import Concept, ConceptRelationship, Paper, User, UserPaperAccess
from app.ingestion.pipeline import canonicalize_existing_paper
from app.services.callbacks import SUPPRESSED_RATE_LIMITED, CallbackService
from app.services.memory import MemoryService
from app.services.signals import SignalService
from scripts.demo_identity import DEMO_EMAIL as DEFAULT_EMAIL

# Shared with `verify_callback.py` so the two scripts cannot drift into
# seeding different scenarios. The pair is *discovered* rather than named:
# relationship typing is a model judgment, so which edges exist varies per
# canonicalization, and hardcoding one turns ordinary variation into a
# broken demo.
from scripts.verify_callback import choose_callback_pair


def _ok(label: str, detail: str = "") -> None:
    print(f"  [PASS] {label}" + (f" — {detail}" if detail else ""))


def _fail(label: str, detail: str = "") -> None:
    print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))


def resolve_uid(email: str) -> str:
    """The Firebase UID for this account.

    This is what `users.auth_subject` must hold. Seeding against the email
    alone would create a row the login flow never finds — it resolves by
    subject, and a mismatch silently provisions a second, empty user.
    """
    import firebase_admin
    from firebase_admin import auth, credentials

    settings = get_settings()
    project = settings.firebase_project_id or settings.vertex_project
    if not project:
        raise SystemExit(
            "set FIREBASE_PROJECT_ID (or VERTEX_PROJECT) so the account can be "
            "looked up in the right Firebase project"
        )

    app = next(iter(firebase_admin._apps.values()), None) or firebase_admin.initialize_app(
        credentials.ApplicationDefault(), {"projectId": project}
    )
    try:
        return auth.get_user_by_email(email, app=app).uid
    except Exception as exc:
        raise SystemExit(
            f"could not find {email!r} in Firebase ({type(exc).__name__}). "
            f"Create it under Authentication -> Users, and make sure "
            f"`gcloud auth application-default login` has run."
        ) from exc


async def _demo_papers(session) -> list[Paper]:
    """The two ingested demo papers, in the order they were added."""
    papers = list(
        (
            await session.scalars(
                select(Paper)
                .where(Paper.processing_status.in_(["ready", "partially_ready"]))
                .order_by(Paper.created_at)
            )
        ).all()
    )
    return [p for p in papers if (p.original_filename or "").lower().endswith(".pdf")]


async def seed(email: str) -> int:
    uid = resolve_uid(email)
    print(f"\n=== {email} ===")
    print(f"  firebase uid: {uid}")

    async with async_session_factory() as session:
        user = await session.scalar(select(User).where(User.auth_subject == uid))
        if user is None:
            user = User(auth_subject=uid, email=email, is_demo_account=True)
            session.add(user)
            await session.flush()
            _ok("created the account's user row", str(user.user_id))
        else:
            user.is_demo_account = True
            _ok("account already provisioned", str(user.user_id))

        papers = await _demo_papers(session)
        if len(papers) < 2:
            _fail(
                "fewer than two ingested papers",
                "run verify_demo_ingestion.py --ingest first",
            )
            return 1

        # Grant first: `GET /api/papers` reads the library *through*
        # `user_paper_access`, so an ungranted paper is invisible however
        # healthy it looks.
        granted = 0
        for paper in papers:
            existing = await session.scalar(
                select(UserPaperAccess).where(
                    UserPaperAccess.user_id == user.user_id,
                    UserPaperAccess.paper_id == paper.paper_id,
                )
            )
            if existing is None:
                session.add(
                    UserPaperAccess(user_id=user.user_id, paper_id=paper.paper_id)
                )
                granted += 1
            elif existing.revoked_at is not None:
                existing.revoked_at = None
                granted += 1
        await session.flush()
        _ok(f"{len(papers)} paper(s) visible", f"{granted} newly granted")

        # Phase 6b only. The chunks and their embeddings are shared by content
        # hash; concepts are per-reader and have to be built for this one.
        for paper in papers:
            already = await session.scalar(
                select(Concept).where(
                    Concept.user_id == user.user_id,
                    Concept.source_paper_ids.overlap([paper.paper_id]),
                )
            )
            if already is not None:
                _ok(f"concepts already built for {(paper.title or '?')[:40]}")
                continue
            linked = await canonicalize_existing_paper(
                session, paper.paper_id, user.user_id
            )
            await session.commit()
            _ok(f"canonicalized {linked} concepts", (paper.title or "?")[:40])

        await session.commit()

        # Session 1: the struggle and the resolution that closed it, exactly as
        # ARCHITECTURE 10.1 works it through. Without this the callback has
        # nothing to call back to, which is correct rather than broken.
        pair = await choose_callback_pair(session, user)
        if pair is None:
            _fail(
                "this account's graph spans no two papers",
                "nothing to call back to; try --rebuild-concepts",
            )
            return 1

        struggled, asked, relationship = pair
        print(
            f"\n  scenario: struggle with {struggled.canonical_name!r}, "
            f"then ask about {asked.canonical_name!r}"
        )
        print(f"            connected by {relationship}")

        signals = SignalService(session)
        if (struggled.evidence_count or 0) == 0:
            struggle = await signals.record(
                user_id=user.user_id,
                concept_name=struggled.canonical_name,
                signal_type="explicit_confusion",
                style_in_play="formal",
                note="Could not follow the formal derivation.",
            )
            resolution = await signals.record(
                user_id=user.user_id,
                concept_name=struggled.canonical_name,
                signal_type="explicit_understanding",
                style_in_play="numerical",
                note="Worked a concrete example through and it clicked.",
            )
            await session.commit()
            if resolution.resolved_observation_id == struggle.observation_id:
                _ok("seeded the struggle and its resolution")
            else:
                _fail("the resolution did not pair with the struggle")
            state = resolution.state
            print(
                f"         score {state.raw_score:.2f} · "
                f"confidence {state.confidence:.2f} · style {state.effective_style}"
            )
        else:
            _ok(
                f"{struggled.canonical_name} already has evidence",
                f"{struggled.evidence_count} signal(s)",
            )

    return await check(email)


async def check(email: str) -> int:
    """Would a judge signing in right now see a working demo?"""
    uid = resolve_uid(email)
    failures = 0

    async with async_session_factory() as session:
        user = await session.scalar(select(User).where(User.auth_subject == uid))
        print("\n=== what this account sees ===")
        if user is None:
            _fail("the account has no user row", "run without --check to seed it")
            return 1

        visible = list(
            (
                await session.scalars(
                    select(Paper)
                    .join(UserPaperAccess, UserPaperAccess.paper_id == Paper.paper_id)
                    .where(
                        UserPaperAccess.user_id == user.user_id,
                        UserPaperAccess.revoked_at.is_(None),
                    )
                )
            ).all()
        )
        if len(visible) >= 2:
            _ok(f"{len(visible)} papers in the library")
        else:
            _fail(f"only {len(visible)} paper(s) visible")
            failures += 1

        concepts = list(
            (
                await session.scalars(
                    select(Concept).where(
                        Concept.user_id == user.user_id,
                        Concept.merged_into_id.is_(None),
                    )
                )
            ).all()
        )
        _ok(f"{len(concepts)} concepts") if concepts else _fail("no concepts")
        failures += 0 if concepts else 1

        edges = await session.scalar(
            select(ConceptRelationship).where(
                ConceptRelationship.user_id == user.user_id
            )
        )
        _ok("the concept graph has edges") if edges else _fail("no relationships")
        failures += 0 if edges else 1

        pair = await choose_callback_pair(session, user)
        if pair is None:
            _fail("no cross-paper pair in this account's graph")
            return failures + 1
        _, asked, _ = pair

        memory = MemoryService(session)
        prefetched = await memory.prefetch(user.user_id, asked.canonical_name)
        decision = await CallbackService(session).decide(
            user=user,
            active_paper_id=(asked.source_paper_ids or [None])[0],
            prefetched=prefetched,
        )
        if decision.fired:
            _ok("the cross-paper callback fires", decision.concept_name)
            print(f"         prior paper: {decision.prior_paper_title}")
            print(f"         style:       {decision.effective_style}")
            print(
                f'\n  ask this on camera: "Explain {asked.canonical_name}"'
            )
        elif decision.suppressed_reason == SUPPRESSED_RATE_LIMITED:
            print("  [WARN] rate-limited — the scenario is sound, a callback fired recently")
        else:
            _fail("callback suppressed", decision.suppressed_reason)
            failures += 1

    print()
    if failures:
        print(f"{failures} check(s) FAILED")
    else:
        print("this account is ready for the demo")
    return failures


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--check", action="store_true", help="verify only, seed nothing")
    args = parser.parse_args()

    return await (check(args.email) if args.check else seed(args.email))


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
