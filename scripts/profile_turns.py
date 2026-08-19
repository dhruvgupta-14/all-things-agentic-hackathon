"""Where a turn's 40-70 seconds actually go (HANDOFF 6.3).

Runs representative turns through the real pipeline against the real model and
prints the span breakdown `app/services/timing.py` collects. No HTTP: this
drives `TurnPipeline` directly, so nothing here measures uvicorn, the proxy or
the browser — which is the point, since those were never the suspects.

    PYTHONPATH=. python scripts/profile_turns.py            # 3 turns
    PYTHONPATH=. python scripts/profile_turns.py --turns 5

**This spends model quota.** Each turn is a real agent loop.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import sys

from sqlalchemy import select

from app.db.base import async_session_factory
from app.db.models import Paper, Session, User, UserPaperAccess
from app.schemas.sse import decode

DEMO_SUBJECT = "local-dev-user"

# Representative of the demo: a definition, a mechanism, and a comparison.
# Deliberately different shapes — a one-search question and a multi-search one
# cost very different amounts, and an average over only easy questions would
# flatter the system.
QUESTIONS = [
    "What is the reparameterization trick?",
    "How is the training objective derived, step by step?",
    "How does this approach compare to the alternatives the paper discusses?",
    "What do the experiments actually show?",
    "Why is the KL term necessary?",
]


class _Collector(logging.Handler):
    """Pull the structured breakdown back out of the log record."""

    def __init__(self) -> None:
        super().__init__()
        self.breakdowns: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        breakdown = getattr(record, "breakdown", None)
        if breakdown is not None:
            self.breakdowns.append(
                {"total_ms": getattr(record, "total_ms", 0.0), "spans": breakdown}
            )


async def _demo_session(session) -> tuple[User, Session, Paper]:
    user = await session.scalar(
        select(User).where(User.auth_subject == DEMO_SUBJECT)
    )
    if user is None:
        raise SystemExit("no demo user — run verify_demo_ingestion.py --ingest")

    paper = await session.scalar(
        select(Paper)
        .join(UserPaperAccess, UserPaperAccess.paper_id == Paper.paper_id)
        .where(
            UserPaperAccess.user_id == user.user_id,
            UserPaperAccess.revoked_at.is_(None),
            Paper.processing_status == "ready",
        )
        .limit(1)
    )
    if paper is None:
        raise SystemExit("no ingested paper for the demo user")

    conversation = Session(user_id=user.user_id, active_paper_id=paper.paper_id)
    session.add(conversation)
    await session.flush()
    await session.commit()
    return user, conversation, paper


async def profile(count: int) -> int:
    from app.services.turns import TurnPipeline

    collector = _Collector()
    logging.getLogger("app.services.timing").addHandler(collector)
    logging.getLogger("app.services.timing").setLevel(logging.INFO)

    totals: list[float] = []

    async with async_session_factory() as session:
        user, conversation, paper = await _demo_session(session)
        print(f"paper: {paper.title}")
        print(f"session: {conversation.session_id}\n")

        for index, question in enumerate(QUESTIONS[:count], start=1):
            print(f"--- turn {index}: {question}")
            frames: list[str] = []
            async for frame in TurnPipeline(session).run(
                conversation, user.user_id, question
            ):
                frames.append(frame)

            events = decode("".join(frames))
            failure = next((e for e in events if e["event"] == "error"), None)
            if failure:
                print(f"    FAILED {failure['code']}: {failure['message'][:120]}\n")
                continue

            done = next(e for e in events if e["event"] == "done")
            tools = [
                tool
                for event in events
                if event["event"] == "state"
                for tool in (event.get("tools_called") or [])
            ]
            totals.append(done["latency_ms"])
            print(
                f"    {done['latency_ms']}ms · {done['grounding_status']} · "
                f"{len(tools)} tool call(s)\n"
            )

    if not collector.breakdowns:
        print("no timings collected")
        return 1

    print("\n=== per-turn breakdown ===")
    aggregate: dict[str, list[float]] = {}
    for index, entry in enumerate(collector.breakdowns, start=1):
        print(f"\nturn {index} — {entry['total_ms']:.0f}ms total")
        measured = 0.0
        for name, item in entry["spans"].items():
            measured += item["ms"]
            calls = f" ×{item['calls']}" if item["calls"] > 1 else ""
            print(f"  {name:<28} {item['ms']:>8.0f}ms  {item['pct']:>5.1f}%{calls}")
            aggregate.setdefault(name, []).append(item["ms"])
        unaccounted = entry["total_ms"] - measured
        print(f"  {'(unaccounted)':<28} {unaccounted:>8.0f}ms")

    print("\n=== median across turns ===")
    for name, values in sorted(
        aggregate.items(), key=lambda item: -statistics.median(item[1])
    ):
        print(f"  {name:<28} {statistics.median(values):>8.0f}ms")

    if totals:
        print(
            f"\ntotal: median {statistics.median(totals):.0f}ms · "
            f"min {min(totals):.0f}ms · max {max(totals):.0f}ms  "
            f"(HANDOFF 6.3 recorded 56000-70000ms)"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=3, help="how many to run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    return asyncio.run(profile(max(1, min(args.turns, len(QUESTIONS)))))


if __name__ == "__main__":
    sys.exit(main())
