"""The asynchronous-work seam.

Locally this runs the ingestion job in-process via FastAPI `BackgroundTasks`.
That is explicitly **not** the deployment design: ARCHITECTURE 2.1 rejects
`BackgroundTasks` because work started in-process dies when Cloud Run reclaims
the instance, leaving a paper stuck in `queued` forever. Deployment replaces
`enqueue_ingestion` with a Cloud Tasks push to `/internal/ingest`, and the
retry contract in section 8.2 becomes meaningful at that point.

The job body below is already written for that world: it owns its own session
and commits its own terminal state, so moving it behind an HTTP route is a
change of caller, not a rewrite.
"""

from __future__ import annotations

import logging
import uuid

from app.db.base import async_session_factory
from app.ingestion.pipeline import (
    PermanentIngestionError,
    TransientIngestionError,
    ingest_paper,
)

logger = logging.getLogger(__name__)


async def run_ingestion_job(paper_id: uuid.UUID) -> None:
    """Run one ingestion job to a terminal state, in its own session."""
    async with async_session_factory() as session:
        try:
            await ingest_paper(session, paper_id)
            await session.commit()
        except PermanentIngestionError:
            # The pipeline already recorded `failed` and the typed code; that
            # write is the point of this commit.
            await session.commit()
        except TransientIngestionError:
            # Roll back to `queued` so a retry can pick the paper up rather
            # than finding it stranded mid-phase.
            await session.rollback()
            logger.warning("ingestion left retryable", extra={"paper_id": str(paper_id)})
        except Exception:
            await session.rollback()
            logger.exception("ingestion job crashed", extra={"paper_id": str(paper_id)})
