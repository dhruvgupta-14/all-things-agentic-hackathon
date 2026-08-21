"""The asynchronous-work seam.

Ingestion takes 30-60 seconds, so it cannot happen on the request path. Where
it happens instead depends on how this process is deployed:

  * **Cloud Tasks** (ARCHITECTURE 8, and the deployment design): the upload
    enqueues an HTTPS push back to `/internal/ingest` on this same service,
    signed with an OIDC token. The queue owns the retry budget, so a job
    survives an instance being reclaimed, a deploy, or a crash.
  * **In-process** via FastAPI `BackgroundTasks`: correct for local
    development, where there is no queue and the server outlives the request.

On Cloud Run the in-process path is **data loss** — the instance is reclaimed
once the response is sent and takes the unfinished job with it, leaving the
paper `queued` forever with nothing to notice it. `dispatch` therefore refuses
that path outright unless `APP_ENV=local`, rather than silently taking it.

The job bodies below are shared by both callers. They return an outcome instead
of raising, because the two callers need opposite things from a failure: the
background path has nobody to tell, and the HTTP path must translate it into
the status code that decides whether Cloud Tasks retries (ARCHITECTURE 8.2).
"""

from __future__ import annotations

import json
import logging
import uuid
from functools import lru_cache
from typing import Literal

from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.db.base import async_session_factory
from app.ingestion.pipeline import (
    PermanentIngestionError,
    TransientIngestionError,
    canonicalize_existing_paper,
    ingest_paper,
)

logger = logging.getLogger(__name__)

# The path the queue pushes back to. Also spelled out in the Cloud Tasks target
# URL, so it lives here rather than being duplicated in the router.
INGEST_PATH = "/internal/ingest"

Job = Literal["ingest", "canonicalize"]

#: `ok` finished; `failed` is terminal and must not be retried; `retry` is
#: worth another attempt. These map onto the 8.2 retry contract in the router.
Outcome = Literal["ok", "failed", "retry"]


class TaskDispatchError(RuntimeError):
    """The job could not be scheduled. The caller must not pretend it was."""


# ---------------------------------------------------------------------------
# The jobs
# ---------------------------------------------------------------------------


async def run_ingestion_job(
    paper_id: uuid.UUID, user_id: uuid.UUID | None = None
) -> Outcome:
    """Run one ingestion job to a terminal state, in its own session.

    `user_id` is whose concept graph phase 6b canonicalizes into. It is the
    uploader, not an argument the model can influence.
    """
    async with async_session_factory() as session:
        try:
            await ingest_paper(session, paper_id, user_id=user_id)
            await session.commit()
            return "ok"
        except PermanentIngestionError:
            # The pipeline already recorded `failed` and the typed code; that
            # write is the point of this commit. Retrying a corrupt PDF five
            # times would only bury the real reason.
            await session.commit()
            return "failed"
        except TransientIngestionError:
            # Roll back to `queued` so a retry can pick the paper up rather
            # than finding it stranded mid-phase.
            await session.rollback()
            logger.warning("ingestion left retryable", extra={"paper_id": str(paper_id)})
            return "retry"
        except Exception:
            await session.rollback()
            logger.exception("ingestion job crashed", extra={"paper_id": str(paper_id)})
            # Retry rather than give up: every phase deletes and re-inserts its
            # own rows, so a re-run from the top is safe, and an unexpected
            # exception is more often a blip than a bad paper.
            return "retry"


async def run_canonicalization_job(paper_id: uuid.UUID, user_id: uuid.UUID) -> Outcome:
    """Phase 6b alone, for a reader of an already-ingested paper.

    Failure does not cost the reader access — the paper is fully searchable —
    but it does cost the cross-paper concept edges the callback depends on, so
    under a queue it is worth retrying. Canonicalization upserts, so a retry is
    idempotent.
    """
    async with async_session_factory() as session:
        try:
            linked = await canonicalize_existing_paper(session, paper_id, user_id)
            await session.commit()
            logger.info(
                "canonicalized shared paper",
                extra={"paper_id": str(paper_id), "concepts": linked},
            )
            return "ok"
        except Exception:
            await session.rollback()
            logger.exception(
                "canonicalization failed", extra={"paper_id": str(paper_id)}
            )
            return "retry"


JOBS = {
    "ingest": run_ingestion_job,
    "canonicalize": run_canonicalization_job,
}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _tasks_client():
    """One client per process. Building it resolves credentials and opens a
    gRPC channel, which is far too much work to repeat per upload."""
    from google.cloud import tasks_v2

    return tasks_v2.CloudTasksClient()


def _create_task(settings: Settings, payload: dict) -> str:
    """Enqueue one HTTPS push. Synchronous — the client is."""
    from google.cloud import tasks_v2

    client = _tasks_client()
    parent = client.queue_path(
        settings.project_id,
        settings.cloud_tasks_location,
        settings.cloud_tasks_queue,
    )
    base = settings.service_base_url.rstrip("/")
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{base}{INGEST_PATH}",
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(payload).encode("utf-8"),
            # The audience is the service's base URL, not the full target: the
            # value the route checks against should not change when a path
            # does. See app/auth/oidc.py.
            "oidc_token": {
                "service_account_email": settings.service_account_email,
                "audience": base,
            },
        }
    }
    # Deliberately unnamed. A name would de-duplicate, but Cloud Tasks keeps a
    # tombstone for roughly an hour after one completes, and a re-enqueue
    # matching it is *silently discarded* — which would break re-ingesting the
    # same paper after a reindex, in exactly the invisible way this whole
    # change exists to remove. Duplicate delivery is the safer failure: every
    # phase deletes and re-inserts its own paper's rows, so a second run
    # converges on the same result.
    return client.create_task(parent=parent, task=task).name


async def dispatch(
    job: Job,
    paper_id: uuid.UUID,
    user_id: uuid.UUID | None,
    *,
    background_tasks,
) -> str:
    """Schedule `job`, and say which way it went.

    Raises:
        TaskDispatchError: it was not scheduled at all. Never returns normally
            in that case — a caller that reported success would leave the user
            watching a paper that is never going to move.
    """
    settings = get_settings()

    if settings.uses_cloud_tasks:
        payload = {
            "job": job,
            "paper_id": str(paper_id),
            "user_id": str(user_id) if user_id else None,
        }
        try:
            # In a thread: the Cloud Tasks client is synchronous, and this runs
            # inside a request that is holding the upload open.
            name = await run_in_threadpool(_create_task, settings, payload)
        except Exception as exc:
            logger.exception(
                "could not enqueue %s", job, extra={"paper_id": str(paper_id)}
            )
            raise TaskDispatchError(str(exc)) from exc

        logger.info("enqueued %s", job, extra={"paper_id": str(paper_id), "task": name})
        return "cloud-tasks"

    if settings.app_env != "local":
        # The failure this guard exists for is not visible at startup: the
        # service boots, uploads are accepted, and papers simply never leave
        # `queued`. Refuse instead.
        raise TaskDispatchError(
            f"APP_ENV is {settings.app_env!r} and Cloud Tasks is not configured. "
            "In-process background work does not survive Cloud Run reclaiming "
            "the instance. Set CLOUD_TASKS_QUEUE, SERVICE_ACCOUNT_EMAIL and "
            "SERVICE_BASE_URL."
        )

    background_tasks.add_task(JOBS[job], paper_id, user_id)
    return "in-process"
