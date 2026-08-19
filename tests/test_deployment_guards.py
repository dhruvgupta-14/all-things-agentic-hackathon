"""Guards on things that fail silently rather than loudly.

A deployment pointed at a regional Vertex endpoint still starts, still
answers, and still cites correctly — it just quietly runs on
`gemini-2.5-flash` because Gemini 3.x returns 404 there, which drops the build
below HK-1's "Flash-class 3.5+" without a single error in the logs. That is
exactly the class of failure a test has to catch, because operating it will
not.
"""

import pathlib

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
    ignore = (REPO / ".dockerignore").read_text(encoding="utf-8")

    for entry in (".env", "venv", ".pgdata", "frontend"):
        assert entry in ignore, f"{entry} would be baked into a pushed image"


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


def test_local_runs_never_touch_the_connector(settings_env):
    """`CLOUD_SQL_INSTANCE` unset must keep the plain host:port DSN, and must
    not import a connector that would then want credentials."""
    from app.config import get_settings

    settings_env(CLOUD_SQL_INSTANCE=None)
    settings = get_settings()

    assert settings.uses_cloud_sql is False
    assert "@" in settings.database_url, "expected a host:port DSN"
    assert "@" in settings.sync_database_url


def test_setting_an_instance_switches_both_engines(settings_env):
    """The async engine and Alembic's sync engine must agree about where the
    database is, or a migration lands somewhere the app never reads."""
    from app.config import get_settings

    settings_env(CLOUD_SQL_INSTANCE="proj:us-central1:paper-companion")
    settings = get_settings()

    assert settings.uses_cloud_sql is True
    # No host in either URL: the connector supplies the socket, and a stray
    # host would silently dial somewhere else.
    assert settings.database_url == "postgresql+asyncpg://"
    assert settings.sync_database_url == "postgresql+psycopg://"


def test_the_query_plan_settings_survive_the_cloud_sql_path():
    """`SERVER_SETTINGS` must travel however the connection is made — they are
    the difference between a correct query plan and a silently wrong one."""
    source = (REPO / "app" / "db" / "base.py").read_text(encoding="utf-8")
    cloud_sql_branch = source.split("if settings.uses_cloud_sql:")[1]

    assert "SERVER_SETTINGS" in cloud_sql_branch
    # Cloud SQL closes idle connections; a pooled-but-dead one would surface
    # as a failed turn rather than a reconnect.
    assert "pool_pre_ping" in cloud_sql_branch


def test_alembic_uses_the_connector_when_configured():
    """Migrations run before the revision that needs them is deployed, so they
    cannot rely on Cloud Run's unix socket."""
    env = (REPO / "alembic" / "env.py").read_text(encoding="utf-8")

    assert "uses_cloud_sql" in env
    assert "sync_creator" in env
