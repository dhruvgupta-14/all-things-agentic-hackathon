"""Sweep conversation history past the retention window.

The application prunes on append, throttled to once an hour per process, so
this exists for the cases that does not cover: a long-idle deployment, a
scheduled run, or checking what would go before it goes.

    python scripts/prune_messages.py --dry-run
    python scripts/prune_messages.py
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from app.db.base import async_session_factory
from app.db.models import MESSAGE_RETENTION_DAYS, Message
from app.services.messages import MessageService


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be deleted"
    )
    args = parser.parse_args()

    cutoff = datetime.now(UTC) - timedelta(days=MESSAGE_RETENTION_DAYS)

    async with async_session_factory() as session:
        total = await session.scalar(select(func.count()).select_from(Message))
        expired = await session.scalar(
            select(func.count()).select_from(Message).where(Message.created_at < cutoff)
        )

        print(f"retention window : {MESSAGE_RETENTION_DAYS} days")
        print(f"cutoff           : {cutoff.isoformat()}")
        print(f"messages         : {total}")
        print(f"past retention   : {expired}")

        if args.dry_run:
            print("\ndry run — nothing deleted")
            return 0

        removed = await MessageService(session).prune_expired()
        await session.commit()
        print(f"\ndeleted {removed} message(s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
