"""The SPA is served from the API container, which makes route order matter.

`mount_spa` mounts at "/", and Starlette matches routes in the order they were
registered. Mounted before the API routers it would answer every request,
including `/api/...`; mounted after, it only sees what nothing else claimed.
That ordering is a one-line mistake with a whole-application blast radius, and
it does not fail loudly — the frontend would receive `index.html` where it
expected JSON and report a parse error, not a routing bug.
"""

import pathlib

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.spa import REPO_ROOT, mount_spa, spa_dist

INDEX = "<!doctype html><title>companion</title><script src=/assets/app-abc123.js>"


def build_dist(tmp_path: pathlib.Path) -> pathlib.Path:
    """A minimal Vite-shaped bundle: an index and one hashed asset."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(INDEX, encoding="utf-8")
    (dist / "assets" / "app-abc123.js").write_text("console.log(1)", encoding="utf-8")
    return dist


def build_app(dist: pathlib.Path) -> FastAPI:
    """An application shaped like the real one: API first, SPA last."""
    app = FastAPI()

    @app.get("/api/papers")
    async def papers():
        return {"papers": []}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    mount_spa(app, dist)
    return app


def test_a_missing_bundle_leaves_the_api_running(tmp_path):
    """The backend is run alone during development, with the SPA on Vite's
    port. Refusing to start without a built bundle would make the API depend on
    the frontend's toolchain."""
    app = FastAPI()

    assert mount_spa(app, tmp_path / "nothing-here") is False

    with TestClient(app) as client:
        assert client.get("/").status_code == 404


def test_the_index_is_served_at_the_root(tmp_path):
    with TestClient(build_app(build_dist(tmp_path))) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "companion" in response.text


def test_the_index_is_revalidated_but_the_assets_need_not_be(tmp_path):
    """index.html names content-hashed bundles. A browser holding a cached copy
    across a deploy asks for files that no longer exist — a blank page, with
    404s in a console nobody has open during a demo."""
    with TestClient(build_app(build_dist(tmp_path))) as client:
        index = client.get("/")
        asset = client.get("/assets/app-abc123.js")

    assert index.headers["cache-control"] == "no-cache"
    assert asset.status_code == 200


def test_the_api_still_wins_at_its_own_paths(tmp_path):
    with TestClient(build_app(build_dist(tmp_path))) as client:
        assert client.get("/api/papers").json() == {"papers": []}
        assert client.get("/health").json() == {"status": "ok"}


def test_an_unknown_api_path_is_json_not_the_index(tmp_path):
    """The failure this whole module exists for. A catch-all that answers
    `/api/typo` with `index.html` turns every backend 404 into an HTML body the
    client tries to parse as an error payload."""
    with TestClient(build_app(build_dist(tmp_path))) as client:
        response = client.get("/api/does-not-exist")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert "<!doctype" not in response.text.lower()


def test_an_unknown_page_is_a_404(tmp_path):
    """There is no client-side router, so serving the index for every unmatched
    path would answer typos with 200 and hide real link rot."""
    with TestClient(build_app(build_dist(tmp_path))) as client:
        assert client.get("/no-such-page").status_code == 404


def test_the_dist_path_does_not_depend_on_the_working_directory():
    """Cloud Run's entry point and a developer's `uvicorn` invocation disagree
    about the working directory; resolving against it would be a blank page."""
    assert spa_dist("frontend/dist") == REPO_ROOT / "frontend" / "dist"
    assert spa_dist(str(REPO_ROOT / "elsewhere")) == REPO_ROOT / "elsewhere"


def test_the_real_application_registers_the_spa_last():
    """Guards the ordering in main.py itself, which the fixtures above cannot:
    they build their own app."""
    source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")

    assert source.index("include_router") < source.index("mount_spa(app")
