"""Seed a synthetic corpus and measure whether retrieval actually uses HNSW.

The question this answers is not "is there an index" but "does the planner
choose it once a scope filter is applied". pgvector's HNSW scan filters after
traversing the graph, so a highly selective `paper_id IN (...)` predicate can
push the planner to a sequential scan instead — which is correct, but changes
the latency story completely.

    python scripts/benchmark_retrieval.py --papers 50 --chunks-per-paper 100
    python scripts/benchmark_retrieval.py --cleanup

The corpus is tagged so cleanup removes exactly what this script created and
nothing else.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
import uuid

from sqlalchemy import text

from app.db.base import async_session_factory
from app.services.embeddings import HashingEmbedder

# Every synthetic paper is marked with this, so cleanup is exact.
BENCHMARK_TAG = "benchmark-corpus"

VOCAB = """
attention transformer gradient descent embedding softmax encoder decoder
convolution recurrent normalisation dropout regularisation optimiser
learning protein folding diffusion crystallography backbone residue
sequence alignment graph neural message passing aggregation node edge
spectral clustering kernel bayesian posterior variational inference
likelihood entropy divergence prior reinforcement policy reward
trajectory actor critic advantage discount horizon
""".split()

TERMS = ["method", "result", "baseline", "dataset", "benchmark", "ablation"]


def _sentence(rng: random.Random) -> str:
    words = rng.sample(VOCAB, k=rng.randint(8, 16))
    return f"The {rng.choice(TERMS)} shows that {' '.join(words)}."


def _paragraph(rng: random.Random) -> str:
    return " ".join(_sentence(rng) for _ in range(rng.randint(4, 8)))


async def seed(papers: int, chunks_per_paper: int, seed_value: int) -> None:
    rng = random.Random(seed_value)
    embedder = HashingEmbedder()
    started = time.perf_counter()

    async with async_session_factory() as session:
        total = 0
        for index in range(papers):
            paper_id = uuid.uuid4()
            section_id = uuid.uuid4()

            await session.execute(
                text(
                    "INSERT INTO papers (paper_id, content_hash, storage_uri, title,"
                    " processing_status, embedding_model)"
                    " VALUES (:pid, :hash, :uri, :title, 'ready', :model)"
                ),
                {
                    "pid": paper_id,
                    "hash": uuid.uuid4().hex + uuid.uuid4().hex[:32],
                    "uri": f"file://{BENCHMARK_TAG}/{paper_id}.pdf",
                    "title": f"{BENCHMARK_TAG} paper {index}",
                    "model": embedder.model_name,
                },
            )
            await session.execute(
                text(
                    "INSERT INTO sections (section_id, paper_id, ordinal, heading,"
                    " section_path, section_role, page_start, page_end)"
                    " VALUES (:sid, :pid, 0, 'Method', '1', 'method', 1, 4)"
                ),
                {"sid": section_id, "pid": paper_id},
            )

            texts = [_paragraph(rng) for _ in range(chunks_per_paper)]
            vectors = embedder.embed_batch(texts)
            rows = [
                {
                    "cid": uuid.uuid4(),
                    "pid": paper_id,
                    "sid": section_id,
                    "ordinal": ordinal,
                    "content": body,
                    "hash": uuid.uuid4().hex + uuid.uuid4().hex[:32],
                    "tokens": max(1, len(body) // 4),
                    "embedding": str(vector),
                }
                for ordinal, (body, vector) in enumerate(zip(texts, vectors, strict=True))
            ]
            await session.execute(
                text(
                    "INSERT INTO chunks (chunk_id, paper_id, section_id, ordinal,"
                    " content, content_hash, token_count, page_start, page_end,"
                    " embedding, is_indexable)"
                    " VALUES (:cid, :pid, :sid, :ordinal, :content, :hash, :tokens,"
                    " 1, 4, CAST(:embedding AS vector), true)"
                ),
                rows,
            )
            total += len(rows)

        await session.commit()

    elapsed = time.perf_counter() - started
    print(f"seeded {papers} papers / {total} chunks in {elapsed:.1f}s")


# Below this many chunks a sequential scan is genuinely the cheaper plan, so
# flagging it would be crying wolf.
SEQ_SCAN_IS_SUSPICIOUS_ABOVE = 2000


async def _explain(
    session, label: str, sql: str, params: dict, corpus_size: int = 0
) -> None:
    plan = (
        await session.execute(text(f"EXPLAIN (ANALYZE, BUFFERS) {sql}"), params)
    ).scalars().all()
    joined = "\n".join(plan)

    timing = next((line for line in plan if "Execution Time" in line), "").strip()

    if "ix_chunks_embedding_hnsw" in joined:
        verdict = "HNSW (approximate)"
    elif "Seq Scan" in joined:
        if corpus_size > SEQ_SCAN_IS_SUSPICIOUS_ABOVE:
            # At this size a seq scan is a mis-costing: check random_page_cost
            # is 1.1 and not the spinning-disk default of 4.0.
            verdict = "SEQ SCAN <- check random_page_cost"
        else:
            verdict = "seq scan (correct at this corpus size)"
    else:
        scan = next(
            (
                line.split(" using ", 1)[1].split(" on ", 1)[0].strip()
                for line in plan
                if " Scan using " in line
            ),
            "unknown",
        )
        # An exact scan over a small, highly selective scope is the better
        # plan than approximate ANN, and has perfect recall.
        verdict = f"exact scan via {scan}"
    print(f"\n--- {label} -> {verdict}")
    print(f"    {timing}")
    for line in plan[:6]:
        print(f"    {line}")


async def measure() -> None:
    embedder = HashingEmbedder()
    query = str(embedder.embed_query("attention transformer gradient embedding"))

    async with async_session_factory() as session:
        stats = (
            await session.execute(
                text(
                    "SELECT count(*) AS chunks,"
                    " count(DISTINCT paper_id) AS papers FROM chunks"
                    " WHERE is_indexable AND embedding IS NOT NULL"
                )
            )
        ).first()
        print(f"corpus: {stats.chunks} indexable chunks across {stats.papers} papers")

        await session.execute(text("ANALYZE chunks"))

        paper_ids = (
            await session.execute(
                text("SELECT DISTINCT paper_id FROM chunks LIMIT 10")
            )
        ).scalars().all()

        base = (
            "SELECT chunk_id, embedding <=> CAST(:q AS vector) AS distance"
            " FROM chunks WHERE is_indexable AND embedding IS NOT NULL"
        )
        order = " ORDER BY embedding <=> CAST(:q AS vector) LIMIT 8"

        total = stats.chunks
        await _explain(
            session, "unfiltered (whole corpus)", base + order, {"q": query}, total
        )

        await _explain(
            session,
            f"scoped to {len(paper_ids)} papers",
            base + " AND paper_id = ANY(:ids)" + order,
            {"q": query, "ids": paper_ids},
            total,
        )

        await _explain(
            session,
            "scoped to 1 paper (most selective)",
            base + " AND paper_id = ANY(:ids)" + order,
            {"q": query, "ids": paper_ids[:1]},
            total,
        )


async def cleanup() -> None:
    async with async_session_factory() as session:
        result = await session.execute(
            text("DELETE FROM papers WHERE title LIKE :tag"),
            {"tag": f"{BENCHMARK_TAG}%"},
        )
        await session.commit()
        print(f"removed {result.rowcount} benchmark papers (sections/chunks cascade)")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=int, default=50)
    parser.add_argument("--chunks-per-paper", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cleanup", action="store_true", help="remove the corpus and exit")
    parser.add_argument("--measure-only", action="store_true")
    args = parser.parse_args()

    if args.cleanup:
        await cleanup()
        return

    if not args.measure_only:
        await seed(args.papers, args.chunks_per_paper, args.seed)
    await measure()


if __name__ == "__main__":
    asyncio.run(main())
