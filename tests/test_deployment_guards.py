"""Guards on things that fail silently rather than loudly.

A deployment pointed at a regional Vertex endpoint still starts, still
answers, and still cites correctly — it just quietly runs on
`gemini-2.5-flash` because Gemini 3.x returns 404 there, which drops the build
below HK-1's "Flash-class 3.5+" without a single error in the logs. That is
exactly the class of failure a test has to catch, because operating it will
not.
"""

import asyncio
import pathlib
from types import SimpleNamespace
from unittest import mock

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_the_provisioning_script_emits_the_global_endpoint():
    """It writes the staging `.env`, so it must not hand over the value that
    404s. This regressed once already."""
    script = (REPO / "scripts" / "provision_gcp.sh").read_text(encoding="utf-8")

    assert "VERTEX_LOCATION=global" in script
    assert "VERTEX_LOCATION=$REGION" not in script, (
        "the region is us-central1, where Gemini 3.x returns 404"
    )


def test_the_provisioning_script_names_a_class_that_exists():
    """Its verify hint pointed at `VertexEmbedder`, which never existed."""
    script = (REPO / "scripts" / "provision_gcp.sh").read_text(encoding="utf-8")

    assert "VertexEmbedder" not in script
    assert "get_embedder" in script


def test_the_default_model_satisfies_hk1(settings_env):
    """Any Flash-class Gemini 3.5+ qualifies; 2.5 does not."""
    from app.config import get_settings

    model = get_settings().gemini_model
    major = model.removeprefix("gemini-").split("-")[0]

    assert float(major) >= 3.5, f"{model} does not satisfy HK-1"


def test_the_container_honours_cloud_runs_port():
    """Hardcoding 8000 produces an image that passes locally and fails its
    health check on deploy."""
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")

    assert "${PORT" in dockerfile
    assert "USER " in dockerfile, "PDF parsing handles untrusted bytes"


def test_the_build_context_excludes_secrets_and_local_state():
    ignore = (REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()

    for entry in (".env", "venv", ".pgdata", "frontend/node_modules"):
        assert entry in ignore, f"{entry} would be baked into a pushed image"

    # A bare `frontend` line is what this used to say, and it would now break
    # the build rather than tighten it: the SPA sources have to reach the build
    # context for the node stage to compile them.
    assert "frontend" not in ignore


def test_the_image_builds_the_spa_and_ships_only_its_output():
    """The frontend calls the API on relative paths, so it has to be served
    from the same origin. Shipping the API alone produces a deployment where
    the page loads and every request 404s against the static host."""
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")

    assert "npm ci" in dockerfile, "npm install would ignore the lockfile"
    assert "npm run build" in dockerfile
    # From the node stage, and only `dist`: node_modules must not reach the
    # registry, and the built bundle must land where SPA_DIST_DIR points.
    assert "COPY --from=spa /spa/dist ./frontend/dist" in dockerfile


@pytest.mark.parametrize(
    "setting", ["random_page_cost", "hnsw.iterative_scan"]
)
def test_the_query_plan_settings_travel_with_the_application(setting: str):
    """Both are the difference between a working query and a silently wrong
    one, and neither can be left to server configuration being applied."""
    from app.db.base import SERVER_SETTINGS

    assert SERVER_SETTINGS.get(setting), f"{setting} is not set per-connection"


# --------------------------------------------------------------------------
# Cloud SQL
# --------------------------------------------------------------------------


def test_there_is_one_database_and_it_is_cloud_sql(settings_env):
    """No branch left to take.

    Both DSNs used to have a second form reaching a local Postgres on
    DB_HOST/DB_PORT, chosen by whether CLOUD_SQL_INSTANCE happened to be set.
    That meant the application could be exercised at length against a database
    configured differently from the one it would run on — which is exactly how
    `hnsw.iterative_scan` came to be wrong on the first Cloud SQL instance
    while every local run looked healthy.
    """
    from app.config import Settings, get_settings

    settings = get_settings()

    # No host in either URL: the connector supplies the connection, and a stray
    # host would silently dial somewhere else.
    assert settings.database_url == "postgresql+asyncpg://"
    # pg8000, not psycopg: the connector's psycopg driver needs a unix domain
    # socket, which Windows does not have, and migrations run from whatever
    # laptop the operator is using.
    assert settings.sync_database_url == "postgresql+pg8000://"

    # The settings that selected the local path are gone, not merely unused.
    for field in ("db_host", "db_port", "uses_cloud_sql"):
        assert field not in Settings.model_fields

    # And the instance is required, so an unset one is a startup failure rather
    # than a silent fallback.
    assert Settings.model_fields["cloud_sql_instance"].is_required()


def test_the_query_plan_settings_reach_the_cloud_sql_connection():
    """`SERVER_SETTINGS` must be handed to the *creator*, not `connect_args`.

    SQLAlchemy gives connection-making entirely to a creator and ignores
    `connect_args` once one is supplied. An earlier version passed them the
    usual way and they were dropped without a word: the first Cloud SQL
    instance came up with `hnsw.iterative_scan = off`, which makes a filtered
    vector query return *nothing* at scale, silently.

    Asserting the text appears somewhere in the branch is what let that
    through, so this checks it is actually threaded into the creator call.
    """
    from app.db.cloud_sql import async_creator

    settings = SimpleNamespace(
        cloud_sql_instance="p:r:i",
        db_user="app",
        db_password="secret",
        db_name="paper_companion",
        cloud_sql_ip_type="PUBLIC",
    )
    captured = {}

    async def fake_connect_async(instance, driver, **kwargs):
        captured.update(kwargs)
        return object()

    with mock.patch(
        "app.db.cloud_sql._async_connector",
        return_value=SimpleNamespace(connect_async=fake_connect_async),
    ):
        asyncio.run(async_creator(settings, {"hnsw.iterative_scan": "strict_order"})())

    assert captured.get("server_settings") == {"hnsw.iterative_scan": "strict_order"}


def test_the_cloud_sql_engine_pre_pings():
    """Cloud SQL closes idle connections; a pooled-but-dead one would surface
    as a failed turn rather than a reconnect."""
    source = (REPO / "app" / "db" / "base.py").read_text(encoding="utf-8")

    assert "pool_pre_ping=True" in source


def test_the_connector_is_not_shared_across_event_loops():
    """A `Connector` binds to the loop it was built on and refuses any other.

    One process-wide singleton looked right and broke the moment a script
    called `asyncio.run()` twice.
    """
    source = (REPO / "app" / "db" / "cloud_sql.py").read_text(encoding="utf-8")

    assert "get_running_loop" in source
    assert "Connector(loop=loop)" in source


def test_alembic_uses_the_connector():
    """Migrations run before the revision that needs them is deployed, so they
    cannot rely on Cloud Run's unix socket."""
    env = (REPO / "alembic" / "env.py").read_text(encoding="utf-8")

    assert "sync_creator" in env


def test_alembics_only_escape_hatch_is_explicit():
    """`ALEMBIC_DATABASE_URL` points migrations at the test database, and it is
    the one place in this repository that reaches a database other than Cloud
    SQL. It must stay an argument to the migration tool — if the application
    ever read it, the local path would be back under a different name."""
    env = (REPO / "alembic" / "env.py").read_text(encoding="utf-8")
    assert "ALEMBIC_DATABASE_URL" in env

    app_sources = [
        path.read_text(encoding="utf-8")
        for path in (REPO / "app").rglob("*.py")
    ]
    assert not any("ALEMBIC_DATABASE_URL" in source for source in app_sources)
