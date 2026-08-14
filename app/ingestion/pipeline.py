"""The ingestion job: one job, phases run in process (ARCHITECTURE 8.1).

Phases run fetch -> parse -> section -> chunk -> embed. Phase 6 (analyze and
canonicalize into the user's concept graph) needs Gemini and is not wired up;
it contributes concept candidates rather than gating retrieval, so a paper
that reaches the end of phase 5 is genuinely able to answer questions.

A paper finishes `ready`, or `partially_ready` when some pages could not be
read — in which case `unreadable_pages` says which, so the answer can be
honest about the gap.

Idempotency: each run deletes and re-inserts this paper's own sections and
chunks inside one transaction, so a retry from the top is safe and cheap. That
is what lets the retry contract re-run the whole job rather than resume a
stage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Paper, Section
from app.ingestion.chunker import chunk_sections
from app.ingestion.parser import (
    PdfCorruptError,
    PdfEncryptedError,
    PdfNotExtractableError,
    parse_pdf,
)
from app.ingestion.sectioner import detect_sections
from app.services.embeddings import Embedder, get_embedder
from app.services.storage import ObjectNotFoundError, Storage, get_storage

logger = logging.getLogger(__name__)


class TransientIngestionError(Exception):
    """Worth retrying: the queue should hand this back with a 503."""


class PermanentIngestionError(Exception):
    """Not worth retrying. Carries the typed code stored on the paper row.

    Returning 5xx for a corrupt PDF is the classic mistake — the queue retries
    it five times and buries the real reason.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class IngestionResult:
    paper_id: str
    status: str
    section_count: int
    chunk_count: int
    unreadable_pages: list[int]


async def _set_phase(session: AsyncSession, paper: Paper, phase: str) -> None:
    paper.processing_phase = phase
    await session.flush()


async def ingest_paper(
    session: AsyncSession,
    paper_id,
    *,
    storage: Storage | None = None,
    embedder: Embedder | None = None,
) -> IngestionResult:
    """Run the ingestion job for one paper.

    Terminal states are written by this function; the caller only decides what
    HTTP status the queue sees.
    """
    storage = storage or get_storage()

    paper = await session.scalar(select(Paper).where(Paper.paper_id == paper_id))
    if paper is None:
        raise PermanentIngestionError("paper_not_found", f"No paper {paper_id}")

    paper.processing_status = "processing"
    await _set_phase(session, paper, "fetch")

    try:
        try:
            data = storage.get(paper.storage_uri)
        except ObjectNotFoundError as exc:
            raise PermanentIngestionError("original_missing", str(exc)) from exc

        await _set_phase(session, paper, "parse")
        try:
            document = parse_pdf(data)
        except PdfEncryptedError as exc:
            raise PermanentIngestionError("pdf_encrypted", str(exc)) from exc
        except PdfNotExtractableError as exc:
            raise PermanentIngestionError("pdf_not_extractable", str(exc)) from exc
        except PdfCorruptError as exc:
            raise PermanentIngestionError("pdf_corrupt", str(exc)) from exc

        paper.page_count = document.page_count
        paper.extractable_text_ratio = document.extractable_text_ratio
        paper.unreadable_pages = document.unreadable_pages or None
        if document.title and not paper.title:
            paper.title = document.title[:1000]

        await _set_phase(session, paper, "section")
        detected = detect_sections(document.pages)
        if not detected:
            raise PermanentIngestionError(
                "no_sections", "No readable text could be organised into sections."
            )

        # Idempotent re-ingest: chunks cascade from sections.
        await session.execute(delete(Section).where(Section.paper_id == paper.paper_id))
        await session.flush()

        section_ids: dict[int, object] = {}
        for detected_section in detected:
            row = Section(
                paper_id=paper.paper_id,
                ordinal=detected_section.ordinal,
                heading=(detected_section.heading or None),
                section_path=detected_section.section_path[:200],
                section_role=detected_section.section_role,
                page_start=detected_section.page_start,
                page_end=detected_section.page_end,
            )
            session.add(row)
            await session.flush()
            section_ids[detected_section.ordinal] = row.section_id

        await _set_phase(session, paper, "chunk")
        chunks = chunk_sections(detected)
        if not chunks:
            raise PermanentIngestionError("no_chunks", "Sections produced no chunks.")

        chunk_rows: list[Chunk] = []
        for chunk in chunks:
            row = Chunk(
                paper_id=paper.paper_id,
                section_id=section_ids[chunk.section_ordinal],
                ordinal=chunk.ordinal,
                content=chunk.content,
                content_hash=chunk.content_hash,
                token_count=chunk.token_count,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                is_indexable=chunk.is_indexable,
            )
            session.add(row)
            chunk_rows.append(row)
        await session.flush()

        await _set_phase(session, paper, "embed")
        embedder = embedder or get_embedder()
        # Only indexable chunks are embedded: reference entries are stored for
        # completeness but never retrieved, so a vector for them is waste.
        indexable = [row for row in chunk_rows if row.is_indexable]
        if indexable:
            try:
                vectors = embedder.embed_batch([row.content for row in indexable])
            except Exception as exc:
                # An embedding outage is the archetypal transient failure:
                # the parse work is sound and a retry will very likely succeed.
                raise TransientIngestionError(f"embedding failed: {exc}") from exc

            if len(vectors) != len(indexable):
                raise TransientIngestionError(
                    f"embedder returned {len(vectors)} vectors for {len(indexable)} chunks"
                )
            for row, vector in zip(indexable, vectors, strict=True):
                row.embedding = vector

        paper.embedding_model = embedder.model_name
        await session.flush()

        # Phase 6 (analyze + canonicalize) needs Gemini and is not wired up.
        # It adds concept candidates; it does not gate retrieval, so a paper
        # with vectors is genuinely ready to answer questions.
        paper.processing_status = "partially_ready" if document.is_partial else "ready"
        paper.processing_phase = None
        paper.error_code = None
        await session.flush()

        logger.info(
            "ingested paper",
            extra={
                "paper_id": str(paper.paper_id),
                "sections": len(detected),
                "chunks": len(chunks),
            },
        )

        return IngestionResult(
            paper_id=str(paper.paper_id),
            status=paper.processing_status,
            section_count=len(detected),
            chunk_count=len(chunks),
            unreadable_pages=document.unreadable_pages,
        )

    except PermanentIngestionError as exc:
        paper.processing_status = "failed"
        paper.processing_phase = None
        paper.error_code = exc.code
        await session.flush()
        logger.warning(
            "ingestion failed permanently",
            extra={"paper_id": str(paper.paper_id), "error_code": exc.code},
        )
        raise
    except TransientIngestionError:
        # Already classified; re-raise rather than re-wrapping the message.
        raise
    except Exception as exc:
        # Unknown failures are assumed transient: the paper stays `processing`
        # so a retry can pick it up, rather than being written off as failed.
        logger.exception("ingestion failed transiently: %s", exc)
        raise TransientIngestionError(str(exc)) from exc
