"""The Cloud Tasks push target (ARCHITECTURE 8.2, section 15 row "Ingestion worker").

One route, reachable only by our own queue. It is not part of the browser API:
no Firebase token is accepted here and no `Principal` is resolved, because the
caller is a queue, not a person. Authorization is the OIDC check in
`app/auth/oidc.py`.

**The status code is the contract, not the response body.** Cloud Tasks reads
it and nothing else:

    transient failure  ->  503  ->  retried with backoff, five attempts
    permanent failure  ->  200  ->  not retried; the paper row already says why

Returning 5xx for a corrupt PDF is the classic mistake — the queue would retry
it five times, and the useful error would be buried under four identical ones.
Returning 200 for a Vertex timeout is the opposite mistake and worse: the paper
stays broken and nothing ever tries again.

`user_id` arrives in the payload, and that is safe here in a way it would never
be on a browser route: this body was written by `dispatch` at upload time and
signed by the queue, so it carries the uploader's id rather than a
client-supplied one.
"""

from __future__ import annotations

import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel

from app.auth.oidc import require_cloud_tasks
from app.services.tasks import INGEST_PATH, JOBS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", tags=["internal"], include_in_schema=False)


class TaskPayload(BaseModel):
    """What `dispatch` puts on the queue.

    Strict about the job name: `Literal` means an unknown value is a 422 from
    the framework rather than a `KeyError` that Cloud Tasks would read as a
    500 and retry five times.
    """

    job: Literal["ingest", "canonicalize"]
    paper_id: uuid.UUID
    user_id: uuid.UUID | None = None


@router.post(INGEST_PATH.removeprefix("/internal"))
async def ingest(
    payload: TaskPayload,
    response: Response,
    caller: str = Depends(require_cloud_tasks),
) -> dict:
    logger.info(
        "push received",
        extra={
            "job": payload.job,
            "paper_id": str(payload.paper_id),
            "caller": caller,
        },
    )

    if payload.job == "canonicalize" and payload.user_id is None:
        # Phase 6b is per-reader; there is no graph to canonicalize into
        # without one. Permanent, so 200: a retry would find the same payload.
        logger.error("canonicalize push carried no user_id")
        return {"status": "failed", "reason": "user_id required"}

    outcome = await JOBS[payload.job](payload.paper_id, payload.user_id)

    if outcome == "retry":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {"status": outcome}
