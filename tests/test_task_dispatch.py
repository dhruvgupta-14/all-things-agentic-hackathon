"""Where ingestion actually gets scheduled, and what happens when it cannot be.

`BackgroundTasks` on Cloud Run is silent data loss: the instance is reclaimed
once the response is sent, the half-finished job goes with it, and the paper
sits at `queued` forever with nothing logging a complaint. The tests here exist
because every failure in this area looks like success from the outside — the
upload returns 202 either way.
"""

import uuid
from unittest import mock

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Paper
from app.services.tasks import TaskDispatchError, dispatch
from tests.conftest import build_pdf
from tests.test_ingestion_pipeline import PAPER_PAGES

QUEUE_ENV = {
    "CLOUD_TASKS_QUEUE": "ingestion",
    "CLOUD_TASKS_LOCATION": "us-central1",
    "SERVICE_ACCOUNT_EMAIL": "paper-companion@proj.iam.gserviceaccount.com",
    "SERVICE_BASE_URL": "https://companion-abc.a.run.app",
    "GCP_PROJECT": "proj",
}


class Recorder:
    """Stands in for FastAPI's BackgroundTasks."""

    def __init__(self):
        self.tasks = []

    def add_task(self, fn, *args):
        self.tasks.append((fn, args))


# --------------------------------------------------------------------------
# Which path
# --------------------------------------------------------------------------


async def test_local_development_runs_the_job_in_process(settings_env):
    """No queue locally, and the dev server outlives the request, so this is
    the correct path — not a degraded one."""
    settings_env(APP_ENV="local", CLOUD_TASKS_QUEUE=None)
    background = Recorder()

    where = await dispatch(
        "ingest", uuid.uuid4(), uuid.uuid4(), background_tasks=background
    )

    assert where == "in-process"
    assert len(background.tasks) == 1


async def test_a_deployment_without_a_queue_refuses_rather_than_dropping_papers(
    settings_env,
):
    """The whole point. A deployed service with no queue configured must not
    quietly fall back to in-process work — that returns 202 to the user and
    then loses the paper."""
    settings_env(APP_ENV="staging", CLOUD_TASKS_QUEUE=None)
    background = Recorder()

    with pytest.raises(TaskDispatchError) as raised:
        await dispatch("ingest", uuid.uuid4(), uuid.uuid4(), background_tasks=background)

    assert "CLOUD_TASKS_QUEUE" in str(raised.value)
    assert background.tasks == [], "it must not have scheduled anything"


async def test_a_partial_queue_configuration_is_not_treated_as_configured(
    settings_env,
):
    """A queue name with no service URL has nowhere to push. Counting that as
    "on" would fail at the first upload instead of at deploy."""
    settings_env(APP_ENV="staging", CLOUD_TASKS_QUEUE="ingestion", SERVICE_BASE_URL=None)

    with pytest.raises(TaskDispatchError):
        await dispatch("ingest", uuid.uuid4(), None, background_tasks=Recorder())


# --------------------------------------------------------------------------
# What gets enqueued
# --------------------------------------------------------------------------


async def test_the_push_targets_this_service_with_an_oidc_token(settings_env):
    settings_env(APP_ENV="staging", **QUEUE_ENV)
    paper_id, user_id = uuid.uuid4(), uuid.uuid4()

    with mock.patch("app.services.tasks._create_task", return_value="tasks/1") as create:
        where = await dispatch(
            "ingest", paper_id, user_id, background_tasks=Recorder()
        )

    assert where == "cloud-tasks"
    (_settings, payload), _ = create.call_args
    assert payload == {
        "job": "ingest",
        "paper_id": str(paper_id),
        "user_id": str(user_id),
    }


def test_the_task_body_is_shaped_the_way_cloud_tasks_expects(settings_env):
    """Built rather than mocked: the URL, the audience and the signing identity
    are the three fields that decide whether the push is accepted, and a typo
    in any of them is a 401 nobody sees until a paper stops processing."""
    from app.config import get_settings
    from app.services.tasks import _create_task

    settings_env(APP_ENV="staging", **QUEUE_ENV)
    client = mock.MagicMock()
    client.queue_path.return_value = "projects/proj/locations/us-central1/queues/ingestion"

    with mock.patch("app.services.tasks._tasks_client", return_value=client):
        _create_task(get_settings(), {"job": "ingest", "paper_id": "p", "user_id": "u"})

    task = client.create_task.call_args.kwargs["task"]
    request = task["http_request"]

    assert request["url"] == "https://companion-abc.a.run.app/internal/ingest"
    assert request["oidc_token"] == {
        "service_account_email": QUEUE_ENV["SERVICE_ACCOUNT_EMAIL"],
        # The base URL, not the full target: the audience must not shift when a
        # path does. app/auth/oidc.py checks against exactly this value.
        "audience": "https://companion-abc.a.run.app",
    }
    assert b'"job": "ingest"' in request["body"]

    # Unnamed on purpose: a named task is de-duplicated against an hour-long
    # tombstone, and the silently-discarded re-enqueue would be undebuggable.
    assert "name" not in task


async def test_an_enqueue_failure_is_raised_not_swallowed(settings_env):
    settings_env(APP_ENV="staging", **QUEUE_ENV)

    with (
        mock.patch("app.services.tasks._create_task", side_effect=RuntimeError("boom")),
        pytest.raises(TaskDispatchError),
    ):
        await dispatch("ingest", uuid.uuid4(), None, background_tasks=Recorder())


# --------------------------------------------------------------------------
# What the uploader is told
# --------------------------------------------------------------------------


async def test_an_unschedulable_upload_fails_visibly(
    client: AsyncClient, db_session: AsyncSession, dev_auth, storage_dir, monkeypatch
):
    """The paper row is committed before the enqueue, so a failure here would
    otherwise leave it `queued` forever — the exact stall this work removes.
    It must end in a state the user can see, and a status code they can act
    on."""

    async def refuse(*args, **kwargs):
        raise TaskDispatchError("no queue")

    monkeypatch.setattr("app.routers.papers.dispatch", refuse)

    response = await client.post(
        "/api/papers",
        files={"file": ("paper.pdf", build_pdf(PAPER_PAGES), "application/pdf")},
    )

    assert response.status_code == 503

    # By filename, not "the only paper": the fixtures seed a decoy corpus, and
    # picking the first row would have asserted about someone else's paper.
    paper = await db_session.scalar(
        select(Paper).where(Paper.original_filename == "paper.pdf")
    )
    assert paper.processing_status == "failed"
    assert paper.error_code == "enqueue_failed"


async def test_a_duplicate_upload_still_succeeds_when_canonicalization_cannot_be_scheduled(
    client: AsyncClient, dev_auth, storage_dir, monkeypatch
):
    """Asymmetric with the case above, deliberately: the paper is already
    parsed and searchable, so refusing access over missing concept edges would
    withhold something that is sitting right there, ready."""
    dispatched: list[str] = []

    async def flaky(job, paper_id, user_id, *, background_tasks):
        dispatched.append(job)
        if job == "canonicalize":
            raise TaskDispatchError("no queue")
        return "in-process"

    monkeypatch.setattr("app.routers.papers.dispatch", flaky)

    data = build_pdf(PAPER_PAGES)
    first = await client.post(
        "/api/papers", files={"file": ("a.pdf", data, "application/pdf")}
    )
    second = await client.post(
        "/api/papers", files={"file": ("b.pdf", data, "application/pdf")}
    )

    assert first.status_code == 202
    assert second.status_code == 202
    assert dispatched == ["ingest", "canonicalize"]
