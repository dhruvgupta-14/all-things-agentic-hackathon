import hashlib
import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import Principal, get_current_user
from app.config import get_settings
from app.db.base import get_db
from app.db.models import Paper, UserPaperAccess
from app.ingestion.parser import PdfCorruptError, PdfEncryptedError, probe_page_count
from app.services import embeddings, storage
from app.services.tasks import TaskDispatchError, dispatch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/papers", tags=["papers"])

PDF_MAGIC = b"%PDF-"
_READ_CHUNK = 1024 * 1024


def _serialize(paper: Paper, *, nickname=None, last_opened_at=None) -> dict:
    # A paper embedded with a different model is readable but not searchable:
    # its vectors live in another space. Surface that rather than letting it
    # look healthy while silently returning no evidence.
    active_model = embeddings.get_embedder().model_name
    return {
        "paper_id": str(paper.paper_id),
        "title": paper.title,
        "nickname": nickname,
        "processing_status": paper.processing_status,
        "processing_phase": paper.processing_phase,
        "error_code": paper.error_code,
        "page_count": paper.page_count,
        "unreadable_pages": paper.unreadable_pages,
        "embedding_model": paper.embedding_model,
        # So the client can say how long a paper has gone without progress.
        # Ingestion is pushed to a queue that cannot report back that it gave
        # up, so a paper can sit at `queued` indefinitely with no error
        # anywhere, and an indicator that spins forever tells the reader
        # nothing.
        #
        # `updated_at`, not `created_at`. A `papers` row is reused when the
        # same bytes are uploaded again, so a retried paper carries the
        # original creation time — which made the client call it stalled the
        # instant it was re-queued, while it was in fact ingesting normally.
        # A database trigger touches `updated_at` on every write, including
        # each phase change, so this measures *no progress* rather than *age*.
        "updated_at": paper.updated_at.isoformat() if paper.updated_at else None,
        "needs_reindex": (
            paper.processing_status in ("ready", "partially_ready")
            and paper.embedding_model != active_model
        ),
        "last_opened_at": last_opened_at,
    }


async def _read_capped(upload: UploadFile, limit: int) -> bytes:
    """Read the upload, refusing anything over the cap.

    Read incrementally rather than calling `.read()`: an unbounded read would
    have the oversized file in memory before the limit could be applied.
    """
    buffer = bytearray()
    while piece := await upload.read(_READ_CHUNK):
        buffer.extend(piece)
        if len(buffer) > limit:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File exceeds the {limit // (1024 * 1024)} MB limit.",
            )
    return bytes(buffer)


@router.get("")
async def list_papers(
    principal: Principal = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """Papers this user may see.

    Possessing a paper_id grants nothing: the join through `user_paper_access`
    is the only thing that makes a paper visible, and a revoked grant drops out
    here immediately (ARCHITECTURE 4.2).
    """
    rows = await session.execute(
        select(Paper, UserPaperAccess.nickname, UserPaperAccess.last_opened_at)
        .join(UserPaperAccess, UserPaperAccess.paper_id == Paper.paper_id)
        .where(
            UserPaperAccess.user_id == principal.user_id,
            UserPaperAccess.revoked_at.is_(None),
        )
        .order_by(UserPaperAccess.last_opened_at.desc().nullslast())
    )

    return [
        _serialize(paper, nickname=nickname, last_opened_at=last_opened_at)
        for paper, nickname, last_opened_at in rows.all()
    ]


@router.get("/{paper_id}")
async def get_paper(
    paper_id: uuid.UUID,
    principal: Principal = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """Status for one paper. This is what the ingestion progress UI polls."""
    row = (
        await session.execute(
            select(Paper, UserPaperAccess.nickname, UserPaperAccess.last_opened_at)
            .join(UserPaperAccess, UserPaperAccess.paper_id == Paper.paper_id)
            .where(
                Paper.paper_id == paper_id,
                UserPaperAccess.user_id == principal.user_id,
                UserPaperAccess.revoked_at.is_(None),
            )
        )
    ).first()

    if row is None:
        # Deliberately indistinguishable from a paper that does not exist:
        # a 403 here would confirm the id is real to someone without access.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such paper.")

    paper, nickname, last_opened_at = row
    return _serialize(paper, nickname=nickname, last_opened_at=last_opened_at)


@router.delete("/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_paper(
    paper_id: uuid.UUID,
    principal: Principal = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> None:
    """Remove a paper from this reader's library.

    This revokes the reader's grant (ARCHITECTURE 4.2) rather than deleting the
    paper. Two reasons, and the first is not negotiable:

    * **Papers are shared by content hash.** Phases 1-5 are paper-scoped, so
      one `papers` row and its chunks may back several readers. Deleting it
      because one of them tidied up would silently empty another reader's
      library — including papers they are mid-session on.
    * Revocation is read-time. `user_paper_access` is joined on every listing
      and every retrieval scope, so a revoked paper disappears immediately and
      cannot come back through a stale session.

    Re-uploading the same file un-revokes the grant, which is why this is safe
    to offer without a confirmation dialog on the server side.

    Deliberately *not* done here: erasing the concepts this paper contributed
    to the reader's graph. Those are learner memory built from several papers,
    and dropping them would quietly undo learning history. Full erasure is a
    separate, explicit operation (ARCHITECTURE 19, `app/services/erasure.py`).
    """
    grant = await session.scalar(
        select(UserPaperAccess).where(
            UserPaperAccess.user_id == principal.user_id,
            UserPaperAccess.paper_id == paper_id,
            UserPaperAccess.revoked_at.is_(None),
        )
    )

    if grant is None:
        # Indistinguishable from a paper that does not exist, exactly as the
        # read route is: a 403 would confirm the id is real to someone without
        # access.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such paper.")

    grant.revoked_at = func.now()
    await session.commit()
    logger.info(
        "revoked paper access",
        extra={"paper_id": str(paper_id), "user_id": str(principal.user_id)},
    )


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def upload_paper(
    file: UploadFile = File(...),
    principal: Principal = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """Accept a PDF, then ingest it asynchronously (ARCHITECTURE 8.1).

    Everything cheap and rejectable happens here so the user gets a real error
    at the boundary; parsing and chunking happen in the background job.
    """
    settings = get_settings()
    data = await _read_capped(file, settings.max_upload_bytes)

    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is empty.")

    # Sniff the content rather than trusting the declared type or extension.
    if not data.startswith(PDF_MAGIC):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Not a PDF.")

    try:
        pages = probe_page_count(data)
    except PdfEncryptedError:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "PDF is password protected."
        ) from None
    except PdfCorruptError:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "PDF could not be read."
        ) from None

    if pages > settings.max_page_count:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"PDF has {pages} pages; the limit is {settings.max_page_count}.",
        )

    content_hash = hashlib.sha256(data).hexdigest()
    existing = await session.scalar(
        select(Paper).where(Paper.content_hash == content_hash)
    )

    if existing is not None:
        await _grant_access(session, principal.user_id, existing.paper_id, file.filename)

        if existing.processing_status == "failed":
            # Re-uploading the file is the only recovery a reader has, and it
            # used to do nothing: dedupe matched on content hash, granted
            # access, enqueued canonicalization, and returned the same `failed`
            # row. A paper that failed for a transient reason — a Vertex
            # outage, an enqueue that was refused — was unrecoverable by the
            # one action anybody would think to try.
            #
            # Re-running is safe: every phase deletes and re-inserts its own
            # paper's rows. A permanent failure simply fails the same way
            # again, with the same message, which is the honest outcome.
            existing.processing_status = "queued"
            existing.processing_phase = None
            existing.error_code = None
            await session.commit()
            await _dispatch_ingestion(session, existing, principal.user_id)
            return _serialize(existing)

        # Same bytes, already parsed. Skip phases 1-5 — the chunks are
        # paper-scoped and safely shared (ARCHITECTURE 8.4). Only phase 6b
        # still has to run, because concepts are per-reader.
        await session.commit()
        try:
            await dispatch("canonicalize", existing.paper_id, principal.user_id)
        except TaskDispatchError:
            # Not fatal, unlike the ingest branch: the paper is already parsed
            # and fully searchable, and what is lost is the reader's concept
            # edges. Refusing the upload over that would deny access to a paper
            # that is sitting right there, ready.
            logger.error(
                "canonicalization not scheduled",
                extra={"paper_id": str(existing.paper_id)},
            )
        return _serialize(existing)

    backend = storage.get_storage()
    storage_uri = backend.put(data, content_hash=content_hash)

    paper = Paper(
        content_hash=content_hash,
        storage_uri=storage_uri,
        original_filename=(file.filename or None),
        page_count=pages,
        processing_status="queued",
    )
    session.add(paper)
    await session.flush()

    await _grant_access(session, principal.user_id, paper.paper_id, file.filename)
    await session.commit()

    await _dispatch_ingestion(session, paper, principal.user_id)
    return _serialize(paper)


async def _dispatch_ingestion(
    session: AsyncSession, paper: Paper, user_id: uuid.UUID
) -> None:
    """Enqueue ingestion, or leave the paper in a state the reader can see.

    Shared by the new-paper and retry-a-failed-paper paths so they cannot drift
    into handling a refused enqueue differently.
    """
    try:
        await dispatch("ingest", paper.paper_id, user_id)
    except TaskDispatchError:
        # The paper row is already committed, so doing nothing here would leave
        # it `queued` forever — the exact silent stall this dispatch path was
        # built to remove. Record a terminal state the user can actually see,
        # and tell them the upload did not take.
        paper.processing_status = "failed"
        paper.processing_phase = None
        paper.error_code = "enqueue_failed"
        await session.commit()
        logger.error("ingestion not scheduled", extra={"paper_id": str(paper.paper_id)})
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Could not schedule processing. Please try again.",
        ) from None


async def _grant_access(
    session: AsyncSession, user_id: uuid.UUID, paper_id: uuid.UUID, filename: str | None
) -> None:
    """Grant, or un-revoke, this user's access to the paper."""
    grant = await session.scalar(
        select(UserPaperAccess).where(
            UserPaperAccess.user_id == user_id,
            UserPaperAccess.paper_id == paper_id,
        )
    )
    if grant is None:
        session.add(
            UserPaperAccess(
                user_id=user_id,
                paper_id=paper_id,
                nickname=(filename or None),
            )
        )
    else:
        grant.revoked_at = None
    await session.flush()
