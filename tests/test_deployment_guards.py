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
