"""Re-embed papers whose vectors were produced by a different model.

Cosine distance between embeddings from two different models is a meaningless
number that still sorts, so a mixed index returns confident nonsense rather
than an error. `papers.embedding_model` records what produced each set;
this command finds the mismatches and re-runs ingestion for them.

    python scripts/reindex.py --list                  # what is stale, change nothing
    python scripts/reindex.py --dry-run               # what would be re-ingested
    python scripts/reindex.py --stale                 # re-ingest everything stale
    python scripts/reindex.py --paper <uuid>          # re-ingest one paper

Idempotent in two senses: `ingest_paper` deletes and re-inserts a paper's own
sections and chunks inside one transaction, so re-running it is safe; and once
a paper carries the active model it is no longer stale, so a second `--stale`
run finds nothing to do.

Papers are committed one at a time. A failure part-way leaves the papers
already done correctly re-indexed rather than rolling back the whole batch —
re-running picks up where it stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from sqlalchemy import func, or_, select

from app.db.base import async_session_factory
from app.db.models import Paper
from app.ingestion.pipeline import (
    PermanentIngestionError,
    TransientIngestionError,
    ingest_paper,
)
from app.services.embeddings import get_embedder

# Only papers that were successfully ingested can be stale. A `failed` or
# `queued` paper has no vectors to be in the wrong space.
REINDEXABLE_STATUSES = ("ready", "partially_ready")


async def find_stale(session, active_model: str) -> list[Paper]:
    """Papers with vectors from another model, or from an unrecorded one."""
    statement = (
        select(Paper)
        .where(
            Paper.processing_status.in_(REINDEXABLE_STATUSES),
            or_(
                Paper.embedding_model.is_(None),
                Paper.embedding_model != active_model,
            ),
        )
        .order_by(Paper.created_at)
    )
    return list((await session.scalars(statement)).all())


async def list_state() -> None:
    active = get_embedder().model_name
    async with async_session_factory() as session:
        rows = (
            await session.execute(
                select(
                    Paper.embedding_model,
                    Paper.processing_status,
                    func.count().label("papers"),
                ).group_by(Paper.embedding_model, Paper.processing_status)
            )
        ).all()

        print(f"active embedding model: {active}\n")
        if not rows:
            print("  no papers")
            return

        print(f"  {'model':<24} {'status':<18} papers  state")
        for model, status, count in rows:
            stale = status in REINDEXABLE_STATUSES and model != active
            state = "STALE" if stale else "ok"
            print(f"  {str(model):<24} {status:<18} {count:>6}  {state}")


async def reindex(paper_ids: list[uuid.UUID] | None, stale_only: bool, dry_run: bool) -> int:
    embedder = get_embedder()
    active = embedder.model_name
    failures = 0

    async with async_session_factory() as session:
        if stale_only:
            targets = await find_stale(session, active)
        else:
            targets = list(
                (
                    await session.scalars(
                        select(Paper).where(Paper.paper_id.in_(paper_ids or []))
                    )
                ).all()
            )

        if not targets:
            print("nothing to re-index")
            return 0

        print(f"active embedding model: {active}")
        print(f"{len(targets)} paper(s) to re-index\n")

        for paper in targets:
            label = paper.title or str(paper.paper_id)
            if dry_run:
                print(f"  would re-index {label} (currently {paper.embedding_model})")
                continue

            try:
                # user_id is deliberately omitted: re-embedding must not
                # re-run per-reader canonicalization. Concepts already exist,
                # and which readers hold this paper is not this job's business.
                result = await ingest_paper(session, paper.paper_id, embedder=embedder)
                await session.commit()
                print(
                    f"  re-indexed {label}: "
                    f"{result.section_count} sections, {result.chunk_count} chunks"
                )
            except (PermanentIngestionError, TransientIngestionError) as exc:
                await session.rollback()
                failures += 1
                print(f"  FAILED {label}: {type(exc).__name__}: {exc}")
            except Exception as exc:  # noqa: BLE001 - one bad paper must not stop the batch
                await session.rollback()
                failures += 1
                print(f"  FAILED {label}: {exc}")

    if failures:
        print(f"\n{failures} paper(s) failed; re-run to retry just those")
    return failures


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="report state, change nothing")
    group.add_argument("--stale", action="store_true", help="re-index every stale paper")
    group.add_argument("--paper", type=uuid.UUID, action="append", help="re-index one paper")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list:
        await list_state()
        return 0

    return await reindex(args.paper, args.stale, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
