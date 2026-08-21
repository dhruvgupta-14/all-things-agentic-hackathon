# Cloud Run image for the whole application (ARCHITECTURE 19: one service, not
# two). The API and the built SPA ship together and are served from one origin.
#
# Four things here are load-bearing rather than boilerplate:
#
#   * Cloud Run assigns the port at runtime through $PORT. Hardcoding 8000
#     produces a container that passes locally and fails its health check on
#     deploy, which is a slow way to learn this.
#   * The image runs as a non-root user. Nothing in the app needs root, and
#     PDF parsing is the one place untrusted bytes are handled.
#   * The SPA is built here rather than hosted separately. The frontend calls
#     the API on relative paths, so same-origin is what makes it work without
#     CORS — including for the SSE stream. See app/spa.py.
#   * The node stage does not survive into the final image: only `dist` is
#     copied across, so node_modules never reaches the registry.

# ---------------------------------------------------------------------------
FROM node:22-slim AS spa
# ---------------------------------------------------------------------------
WORKDIR /spa

# The lockfile first, and `npm ci` rather than `npm install`: ci installs
# exactly what the lockfile pins and fails if package.json disagrees, so the
# bundle in the image matches the one that was tested.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base
# ---------------------------------------------------------------------------

# PyMuPDF ships manylinux wheels, so no build toolchain is needed. If that ever
# stops being true the failure is a compile error at build time, not a runtime
# surprise — which is why there is no speculative build-essential here.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: they change far less often than the source, so this layer
# stays cached across ordinary deploys.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY scripts ./scripts

# Where SPA_DIST_DIR points by default, resolved against the repo root — which
# in this image is /app.
COPY --from=spa /spa/dist ./frontend/dist

# Non-root. Cloud Run does not require it; handling uploaded PDFs does.
RUN useradd --create-home --uid 1001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Documentation only — Cloud Run ignores EXPOSE and injects $PORT.
EXPOSE 8080

# Single worker on purpose. Turns hold an SSE stream open for tens of seconds,
# so concurrency comes from Cloud Run running more instances, not from more
# workers competing for one instance's memory. `--timeout-keep-alive` is raised
# past the default 5s because an idle SSE connection between events must not be
# reaped mid-turn.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --timeout-keep-alive 120"]
