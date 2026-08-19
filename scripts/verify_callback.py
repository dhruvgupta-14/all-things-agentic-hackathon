"""Set up and check the cross-paper callback (ARCHITECTURE 12).

The callback cannot fire on a fresh database, and that is correct: it needs a
concept the reader has *demonstrably* struggled with. This script plays
session 1 — the struggle and its resolution — so that session 2 has something
to call back to, then checks the gate without spending a model call.

    PYTHONPATH=. python scripts/verify_callback.py            # replay + check
    PYTHONPATH=. python scripts/verify_callback.py --check    # check only
    PYTHONPATH=. python scripts/verify_callback.py --reset    # clear the seed

The default is a *replay*: it clears this concept's observations first, so
running it twice gives the same state rather than piling evidence up. Note
that answering a quiz on this concept legitimately moves the score above the
weakness threshold and stops the callback firing — that is the system working,
not a fault, and re-running this restores the scenario.

The signals it writes are the ones ARCHITECTURE 10.1 works through: confusion
while a concept is explained formally, then understanding after a numerical
re-explanation. That produces score 0.35 and confidence 0.70 by arithmetic, not
by assignment — if those numbers drift, the constants changed.

Nothing here is special-cased for the demo. It records ordinary observations
through the ordinary service; the gate then makes its own decision.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import func, select, text

from app.db.base import async_session_factory
from app.db.models import Concept, Paper, Turn, User
from app.services.callbacks import SUPPRESSED_RATE_LIMITED, CallbackService
from app.services.learner_state import recompute
from app.services.memory import MemoryService
from app.services.signals import SignalService

DEMO_SUBJECT = "local-dev-user"

# The pair the demo rests on. Both are real concepts canonicalized from the
# demo papers, connected by an edge written at ingest.
STRUGGLED_WITH = "Reparameterization trick"          # Auto-Encoding Variational Bayes
ASKED_ABOUT = "Simplified Training Objective"        # Denoising Diffusion


def _ok(label: str, detail: str = "") -> None:
    print(f"  [PASS] {label}" + (f" — {detail}" if detail else ""))


def _fail(label: str, detail: str = "") -> None:
    print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))


async def _demo_user(session) -> User | None:
    return await session.scalar(select(User).where(User.auth_subject == DEMO_SUBJECT))


async def _concept(session, user, name: str) -> Concept | None:
    return await session.scalar(
        select(Concept).where(
            Concept.user_id == user.user_id,
            Concept.canonical_name == name,
            Concept.merged_into_id.is_(None),
        )
    )


async def seed() -> int:
    """Session 1: the struggle, then the resolution that closed it."""
    async with async_session_factory() as session:
        user = await _demo_user(session)
        if user is None:
            print("no demo user — run verify_demo_ingestion.py --ingest first")
            return 2

        concept = await _concept(session, user, STRUGGLED_WITH)
        if concept is None:
            print(f"no concept {STRUGGLED_WITH!r} — has the paper been ingested?")
            return 2

        signals = SignalService(session)
        struggle = await signals.record(
            user_id=user.user_id,
            concept_name=STRUGGLED_WITH,
            signal_type="explicit_confusion",
            style_in_play="formal",
            note="Could not follow why the trick makes the estimator low-variance.",
        )
        resolution = await signals.record(
            user_id=user.user_id,
            concept_name=STRUGGLED_WITH,
            signal_type="explicit_understanding",
            style_in_play="numerical",
            note="Worked a concrete example through and it clicked.",
        )
        await session.commit()

        print(f"\n=== session 1 — {STRUGGLED_WITH} ===")
        _ok("struggle recorded", struggle.observation_id.hex[:8])
        if resolution.resolved_observation_id == struggle.observation_id:
            _ok("the resolution closed that struggle deterministically")
        else:
            _fail("the resolution did not pair with the struggle")

        state = resolution.state
        print(
            f"  score {state.raw_score:.2f} · confidence {state.confidence:.2f} "
            f"· style {state.effective_style}"
        )
        return 0


async def reset() -> int:
    """Remove the seeded observations so the scenario can be replayed."""
    async with async_session_factory() as session:
        user = await _demo_user(session)
        if user is None:
            return 0
        concept = await _concept(session, user, STRUGGLED_WITH)
        if concept is None:
            return 0

        # `observations` is append-only; erasure is the one deliberate door.
        await session.execute(text("SELECT set_config('app.erasure', 'on', true)"))
        await session.execute(
            text("DELETE FROM observations WHERE concept_id = :c"),
            {"c": concept.concept_id},
        )
        await session.execute(text("SELECT set_config('app.erasure', 'off', true)"))
        await recompute(session, concept.concept_id)
        await session.commit()
        print(f"cleared observations for {STRUGGLED_WITH!r}")
        return 0


async def _turns_until_allowed(session, user) -> int:
    """How many more turns before the callback gap is cleared."""
    from app.services.learner_state import CALLBACK_MIN_TURN_GAP

    last = await session.scalar(
        select(func.max(Turn.created_at)).where(
            Turn.user_id == user.user_id, Turn.callback_concept_id.isnot(None)
        )
    )
    if last is None:
        return 0
    since = await session.scalar(
        select(func.count())
        .select_from(Turn)
        .where(Turn.user_id == user.user_id, Turn.created_at > last)
    )
    return max(0, CALLBACK_MIN_TURN_GAP - (since or 0))


async def check() -> int:
    """Would the gate fire, and for the right reason? No model call."""
    failures = 0

    async with async_session_factory() as session:
        user = await _demo_user(session)
        if user is None:
            print("no demo user")
            return 1

        struggled = await _concept(session, user, STRUGGLED_WITH)
        asked = await _concept(session, user, ASKED_ABOUT)

        print("\n=== preconditions ===")
        for label, concept in ((STRUGGLED_WITH, struggled), (ASKED_ABOUT, asked)):
            if concept is None:
                _fail(f"concept {label!r} exists")
                failures += 1
            else:
                _ok(f"concept {label!r} exists")

        if struggled is None or asked is None:
            return failures or 1

        papers = {
            paper_id: title
            for paper_id, title in (
                await session.execute(select(Paper.paper_id, Paper.title))
            ).all()
        }
        prior_titles = [
            papers.get(pid) for pid in (struggled.source_paper_ids or [])
        ]
        active_titles = [papers.get(pid) for pid in (asked.source_paper_ids or [])]
        if set(prior_titles) & set(active_titles):
            _fail("the two concepts share a paper — this is not a cross-paper pair")
            failures += 1
        else:
            _ok("the pair spans two papers", f"{prior_titles} -> {active_titles}")

        if struggled.score_confidence and struggled.score_confidence >= 0.3:
            _ok(
                "the struggle is on record",
                f"score {struggled.understanding_score:.2f} · "
                f"confidence {struggled.score_confidence:.2f} · "
                f"style {struggled.effective_style}",
            )
        else:
            _fail("no confident evidence yet — run without --check to seed it")
            failures += 1

        print("\n=== the gate (ARCHITECTURE 12 steps 4-7) ===")
        active_paper_id = (asked.source_paper_ids or [None])[0]
        memory = MemoryService(session)
        prefetched = await memory.prefetch(user.user_id, ASKED_ABOUT)

        if any(record.concept_id == asked.concept_id for record in prefetched):
            _ok("the question reaches the right concept through memory")
        else:
            _fail("prefetch did not surface the asked-about concept")
            failures += 1

        decision = await CallbackService(session).decide(
            user=user, active_paper_id=active_paper_id, prefetched=prefetched
        )

        if decision.fired:
            _ok("callback permitted", f"{decision.concept_name}")
            print(
                f"         relationship: {decision.relationship_type} · "
                f"style: {decision.effective_style}"
            )
            print(f"         prior paper:  {decision.prior_paper_title}")
            if decision.concept_id == struggled.concept_id:
                _ok("it calls back to the concept they struggled with")
            else:
                _fail(f"it called back to {decision.concept_name!r} instead")
                failures += 1
        elif decision.suppressed_reason == SUPPRESSED_RATE_LIMITED:
            # Not a broken scenario: the gate is spacing callbacks, which is
            # what it is for. Everything else about the setup is sound.
            needed = await _turns_until_allowed(session, user)
            print(
                f"  [WARN] the scenario is ready, but the rate limit is "
                f"holding it back\n"
                f"         a callback fired recently; {needed} more turn(s) "
                f"clear the gap\n"
                f"         (CALLBACK_MIN_TURN_GAP in app/services/learner_state.py)"
            )
        else:
            _fail("callback suppressed", decision.suppressed_reason)
            failures += 1

    print()
    if failures:
        print(f"{failures} check(s) FAILED")
    else:
        print("all checks passed — session 2 will connect the two papers")
    return failures


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check only, no seeding")
    parser.add_argument("--reset", action="store_true", help="clear the seeded signals")
    args = parser.parse_args()

    if args.reset:
        return await reset()
    if not args.check:
        # Reset first, so seeding is a replay rather than an accumulation.
        # Observations pile up otherwise and the score walks away from the
        # 0.35 the scenario depends on — which is also what happens for real
        # once the reader answers a quiz on this concept, and is correct
        # behaviour rather than a bug.
        await reset()
        seeded = await seed()
        if seeded:
            return seeded
    return await check()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
