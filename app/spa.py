"""Serving the built SPA from the API container.

The frontend was written same-origin: every call in `api/client.js` and
`api/stream.js` is a relative path, proxied to the backend by Vite in
development. That is a deliberate design (see the comment at the top of
`vite.config.js`) and it buys two real things — no CORS middleware anywhere,
and an SSE stream the browser treats as first-party.

Hosting the bundle separately would throw both away: the deployed SPA would
need an absolute API base URL, the backend would need CORS, and the preflight
and credential rules would then also apply to the streaming endpoint, which is
the one place a subtle mistake shows up as a turn that never starts. Putting
the bundle in the same container keeps the property the code already relies on
instead of breaking it and patching it back.

The cost is that a frontend change requires an API deploy. For one service with
one deploy pipeline that is a smaller price than a second origin.

Ordering matters: this mounts at "/", so it must be registered **after** every
API router. Starlette matches routes in registration order, so an unknown
`/api/...` path still reaches the API's 404 and comes back as JSON rather than
being answered with `index.html` — which would surface as the frontend trying
to parse a page as an error body.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def spa_dist(configured: str) -> Path:
    """Resolve the configured dist directory against the repo root.

    Relative to the repo rather than the working directory: Cloud Run's entry
    point and a developer's `uvicorn` invocation do not agree about where the
    process was started from, and the difference would be a blank page rather
    than an error.
    """
    path = Path(configured)
    return path if path.is_absolute() else REPO_ROOT / path


def mount_spa(app: FastAPI, dist: Path) -> bool:
    """Serve `dist` at the application root. Returns whether it was mounted.

    A missing directory is not an error. The backend is run on its own during
    development — the SPA is on Vite's port then — and refusing to start
    because a bundle has not been built would make the API depend on the
    frontend's toolchain.
    """
    index = dist / "index.html"
    if not index.is_file():
        logger.info("no SPA bundle at %s; serving the API only", dist)
        return False

    @app.get("/", include_in_schema=False)
    async def spa_index() -> FileResponse:
        # `no-cache` means revalidate, not "do not store". index.html names the
        # content-hashed asset files, so a cached copy of it after a deploy
        # asks for bundles that no longer exist — a white screen with 404s in
        # the console. The assets themselves are safe to cache: their names
        # change when their contents do.
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    app.mount("/", StaticFiles(directory=dist), name="spa")
    logger.info("serving SPA from %s", dist)
    return True
