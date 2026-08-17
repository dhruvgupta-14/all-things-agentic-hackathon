"""Validate the real demo papers end to end, against live infrastructure.

This is deliberately **not** part of the pytest suite. It needs three things
the normal suite must never need:

* the two research PDFs in `demo_papers/`
* a real Gemini API key, and the quota to use it
* the persisted demo records in the local database

`pytest` runs offline on generated PDFs and must stay that way, so this lives
in `scripts/` next to the other verification tools — where it cannot be
collected by accident.

    python scripts/verify_demo_ingestion.py            # check what is there
    python scripts/verify_demo_ingestion.py --ingest   # upload + ingest first
    python scripts/verify_demo_ingestion.py --rebuild-concepts

What it checks is the §12 step-0 precondition: after both papers are ingested,
and *before any question is asked*, the concept graph must already connect
them. If it does not, the cross-paper callback cannot fire and the demo has no
differentiator.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import pathlib
import sys

from sqlalchemy import func, select, text

from app.config import get_settings
from app.db.base import async_session_factory
from app.db.models import Chunk, Concept, ConceptRelationship, Paper, Section, User
from app.ingestion.pipeline import canonicalize_existing_paper, ingest_paper
from app.services.storage import get_storage

DEMO_DIR = pathlib.Path("demo_papers")
DEMO_SUBJECT = "local-dev-user"

# Paper A must be ingested first: the cross-paper edge is written when the
# *second* paper's concepts are canonicalized against the first paper's.
PAPER_A_HINT = "auto-encoding"
PAPER_B_HINT = "denoising"

PASS = "PASS"
FAIL = "FAIL"


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        mark = PASS if ok else FAIL
        if not ok:
            self.failures += 1
        print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
        return ok

    def note(self, text: str) -> None:
        print(f"         {text}")


def _ordered_pdfs() -> list[pathlib.Path]:
    """Paper A first, then Paper B, then anything else."""
    found = sorted(DEMO_DIR.glob("*.pdf"))

    def rank(path: pathlib.Path) -> int:
        name = path.name.lower()
        if PAPER_A_HINT in name:
            return 0
        if PAPER_B_HINT in name:
            return 1
        return 2

    return sorted(found, key=rank)


async def _demo_user(session) -> User | None:
    return await session.scalar(
        select(User).where(User.auth_subject == DEMO_SUBJECT)
    )


async def ingest(rebuild_concepts_only: bool = False) -> None:
    """Upload and ingest both papers, in order, as the demo user."""
    settings = get_settings()
    if not settings.gemini_available:
        print("no Gemini credentials configured — concepts would be skipped")
        print("set GEMINI_API_KEY (or VERTEX_PROJECT) and re-run")
        raise SystemExit(2)

    storage = get_storage()

    async with async_session_factory() as session:
        user = await _demo_user(session)
        if user is None:
            user = User(auth_subject=DEMO_SUBJECT)
            session.add(user)
            await session.flush()
            await session.commit()
            print(f"created demo user {user.user_id}")

        if rebuild_concepts_only:
            # Concepts are derived from `papers.concept_candidates`, so they
            # can be rebuilt without re-parsing or re-embedding. Useful when a
            # transient model outage left the graph without its edges.
            await session.execute(
                text("DELETE FROM concepts WHERE user_id = :u"), {"u": user.user_id}
            )
            await session.commit()
            print("cleared derived concepts\n")

        for path in _ordered_pdfs():
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()

            paper = await session.scalar(
                select(Paper).where(Paper.content_hash == digest)
            )
            if paper is None:
                paper = Paper(
                    content_hash=digest,
                    storage_uri=storage.put(data, content_hash=digest),
                    original_filename=path.name,
                    processing_status="queued",
                )
                session.add(paper)
                await session.flush()
                await session.commit()

            print(f"--- {path.name} ---")
            try:
                if rebuild_concepts_only:
                    linked = await canonicalize_existing_paper(
                        session, paper.paper_id, user.user_id
                    )
                    await session.commit()
                    print(f"    canonicalized {linked} concepts")
                else:
                    result = await ingest_paper(
                        session, paper.paper_id, user_id=user.user_id
                    )
                    await session.commit()
                    print(
                        f"    {result.status}: {result.section_count} sections, "
                        f"{result.chunk_count} chunks, "
                        f"{result.concepts_linked} concepts"
                    )
            except Exception as exc:
                await session.rollback()
                print(f"    FAILED {type(exc).__name__}: {str(exc)[:180]}")
        print()


async def verify() -> int:
    report = Report()

    async with async_session_factory() as session:
        print("\n=== demo papers ===")
        user = await _demo_user(session)
        if not report.check("demo user exists", user is not None, DEMO_SUBJECT):
            return report.failures

        papers = list(
            (await session.scalars(select(Paper).order_by(Paper.created_at))).all()
        )
        demo = [
            paper
            for paper in papers
            if paper.original_filename
            and paper.original_filename.lower().endswith(".pdf")
            and (
                PAPER_A_HINT in paper.original_filename.lower()
                or PAPER_B_HINT in paper.original_filename.lower()
            )
        ]

        report.check("both demo papers present", len(demo) == 2, f"found {len(demo)}")

        for paper in demo:
            sections = await session.scalar(
                select(func.count())
                .select_from(Section)
                .where(Section.paper_id == paper.paper_id)
            )
            chunks = await session.scalar(
                select(func.count())
                .select_from(Chunk)
                .where(Chunk.paper_id == paper.paper_id)
            )
            embedded = await session.scalar(
                select(func.count())
                .select_from(Chunk)
                .where(
                    Chunk.paper_id == paper.paper_id, Chunk.embedding.isnot(None)
                )
            )
            label = (paper.title or paper.original_filename or "?")[:38]
            report.check(
                f"{label}: status ready",
                paper.processing_status in ("ready", "partially_ready"),
                paper.processing_status,
            )
            report.check(
                f"{label}: chunks embedded",
                embedded > 0,
                f"{sections} sections, {chunks} chunks, {embedded} embedded",
            )
            report.check(
                f"{label}: id",
                True,
                str(paper.paper_id),
            )

        if len(demo) != 2:
            return report.failures

        paper_a, paper_b = demo[0], demo[1]

        print("\n=== concept canonicalization ===")
        concepts = list(
            (
                await session.scalars(
                    select(Concept).where(Concept.user_id == user.user_id)
                )
            ).all()
        )
        report.check("concepts extracted", len(concepts) > 0, f"{len(concepts)} concepts")

        shared = [
            concept
            for concept in concepts
            if paper_a.paper_id in (concept.source_paper_ids or [])
            and paper_b.paper_id in (concept.source_paper_ids or [])
        ]
        report.check(
            "a concept is shared by both papers",
            bool(shared),
            ", ".join(c.canonical_name for c in shared) or "none",
        )
        for concept in shared:
            report.note(f"aliases: {concept.aliases}")
            report.note(f"source_paper_ids: {len(concept.source_paper_ids)} papers")

        print("\n=== cross-paper graph (ARCHITECTURE 12, step 0) ===")
        edges = list(
            (
                await session.scalars(
                    select(ConceptRelationship).where(
                        ConceptRelationship.user_id == user.user_id
                    )
                )
            ).all()
        )
        report.check("relationships exist", bool(edges), f"{len(edges)} edges")

        crossing = []
        for edge in edges:
            source = await session.get(Concept, edge.source_concept_id)
            target = await session.get(Concept, edge.target_concept_id)
            source_papers = set(source.source_paper_ids or [])
            target_papers = set(target.source_paper_ids or [])
            # An edge bridges the papers when it starts in B-only material and
            # reaches something the first paper also introduced.
            if (
                paper_b.paper_id in source_papers
                and paper_a.paper_id not in source_papers
                and paper_a.paper_id in target_papers
            ):
                crossing.append((edge, source.canonical_name, target.canonical_name))

        report.check(
            "an edge bridges paper B to paper A",
            bool(crossing),
            f"{len(crossing)} bridging edges",
        )
        for edge, source_name, target_name in crossing:
            report.note(
                f"{source_name} --{edge.relationship_type} "
                f"({edge.confidence:.2f})--> {target_name}"
            )

        component_of = [
            item for item in crossing if item[0].relationship_type == "component_of"
        ]
        report.check(
            "a component_of edge bridges the papers",
            bool(component_of),
            "the demo edge" if component_of else "none found",
        )

    print()
    if report.failures:
        print(f"{report.failures} check(s) FAILED")
    else:
        print("all checks passed — the cross-paper callback has its precondition")
    return report.failures


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ingest", action="store_true", help="upload and ingest both papers first"
    )
    parser.add_argument(
        "--rebuild-concepts",
        action="store_true",
        help="re-derive concepts from stored candidates, without re-parsing",
    )
    args = parser.parse_args()

    if args.ingest or args.rebuild_concepts:
        if not DEMO_DIR.is_dir() or not _ordered_pdfs():
            print(f"no PDFs found in {DEMO_DIR}/")
            return 2
        await ingest(rebuild_concepts_only=args.rebuild_concepts)

    return await verify()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
