"""Is this actually ready to deploy? (Phase 3 preparation.)

Read-only and free: it makes no API calls, creates nothing, and deploys
nothing. Run it before `provision_gcp.sh`, and again before the first
`gcloud run deploy`.

    PYTHONPATH=. python scripts/preflight_deploy.py

Every check here corresponds to something that has already gone wrong once, or
to a guarantee the architecture treats as structural. The `VERTEX_LOCATION`
one in particular: a deployment that quietly falls back to `gemini-2.5-flash`
still works, still answers, and no longer satisfies HK-1 — the kind of failure
you find out about from a judge rather than from a stack trace.
"""

from __future__ import annotations

import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

failures = 0
warnings = 0


def ok(label: str, detail: str = "") -> None:
    print(f"  [PASS] {label}" + (f" — {detail}" if detail else ""))


def fail(label: str, detail: str = "") -> None:
    global failures
    failures += 1
    print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))


def warn(label: str, detail: str = "") -> None:
    global warnings
    warnings += 1
    print(f"  [WARN] {label}" + (f" — {detail}" if detail else ""))


def check_vertex_location() -> None:
    """The one that would ship a silently non-compliant build."""
    print("\n=== Vertex endpoint ===")

    from app.config import get_settings

    settings = get_settings()

    if not settings.vertex_project:
        warn("VERTEX_PROJECT is unset", "deployment must not run on local stubs")
    else:
        ok("VERTEX_PROJECT set", settings.vertex_project)

    if settings.vertex_location == "global":
        ok("VERTEX_LOCATION is 'global'")
    else:
        fail(
            "VERTEX_LOCATION is not 'global'",
            f"{settings.vertex_location!r} — Gemini 3.x 404s on a regional "
            f"endpoint and only 2.5-flash answers, silently failing HK-1",
        )

    model = settings.gemini_model
    if model.startswith("gemini-3") or model.startswith("gemini-4"):
        ok("model satisfies HK-1 (Flash-class 3.5+)", model)
    else:
        fail("model does not satisfy HK-1", model)

    # The provisioning script writes the staging env, so it must not undo this.
    script = (REPO / "scripts" / "provision_gcp.sh").read_text(encoding="utf-8")
    if "VERTEX_LOCATION=global" in script:
        ok("provision_gcp.sh emits VERTEX_LOCATION=global")
    else:
        fail(
            "provision_gcp.sh does not emit VERTEX_LOCATION=global",
            "it would hand the deployment the value that 404s",
        )


def check_container() -> None:
    print("\n=== container ===")

    dockerfile = REPO / "Dockerfile"
    if not dockerfile.exists():
        fail("Dockerfile is missing")
        return
    text = dockerfile.read_text(encoding="utf-8")

    if "${PORT" in text:
        ok("honours Cloud Run's $PORT")
    else:
        fail("does not honour $PORT", "Cloud Run assigns the port at runtime")

    if "USER " in text:
        ok("runs as a non-root user")
    else:
        warn("runs as root", "PDF parsing handles untrusted bytes")

    ignore = REPO / ".dockerignore"
    if not ignore.exists():
        fail(".dockerignore is missing", "the build context would include .env")
    else:
        entries = ignore.read_text(encoding="utf-8")
        if ".env" in entries:
            ok(".dockerignore excludes .env")
        else:
            fail(".dockerignore does not exclude .env", "secrets would ship in the image")


def check_runtime_config() -> None:
    print("\n=== runtime configuration ===")

    from app.config import get_settings

    settings = get_settings()

    if settings.app_env == "local":
        warn("APP_ENV is 'local'", "set to 'staging' or 'production' when deploying")
    else:
        ok("APP_ENV", settings.app_env)

    # The dev bypass is refused outside local, but shipping it set is still a
    # loaded gun pointed at the next person who changes APP_ENV.
    if settings.auth_dev_bypass_subject and settings.app_env != "local":
        fail(
            "AUTH_DEV_BYPASS_SUBJECT is set outside local",
            "it is refused at runtime, but must not be in a deployed config",
        )
    elif settings.auth_dev_bypass_subject:
        warn("AUTH_DEV_BYPASS_SUBJECT set", "must be blank in the deployed config")
    else:
        ok("AUTH_DEV_BYPASS_SUBJECT is blank")

    if settings.app_env != "local" and not settings.firebase_project_id:
        fail("FIREBASE_PROJECT_ID is unset", "/me and /papers return 503 without it")
    elif settings.firebase_project_id:
        ok("FIREBASE_PROJECT_ID set", settings.firebase_project_id)
    else:
        warn("FIREBASE_PROJECT_ID unset", "required once the dev bypass is off")

    if settings.storage_bucket:
        ok("STORAGE_BUCKET set — GCS backend", settings.storage_bucket)
    elif settings.app_env == "local":
        warn("STORAGE_BUCKET unset", "uploads go to a local directory")
    else:
        fail(
            "STORAGE_BUCKET is unset",
            "uploads would go to the container filesystem, which Cloud Run "
            "discards on every restart",
        )

    if settings.uses_cloud_tasks:
        ok(
            "Cloud Tasks configured",
            f"{settings.cloud_tasks_queue} → {settings.service_base_url}",
        )
    elif settings.app_env == "local":
        ok("Cloud Tasks unset — ingestion runs in-process, correct locally")
    else:
        fail(
            "Cloud Tasks is not configured",
            "a deployed upload is refused with 503 rather than quietly run "
            "in-process, which does not survive instance reclaim",
        )

    if settings.service_base_url and not settings.service_base_url.startswith("https://"):
        fail(
            "SERVICE_BASE_URL is not https",
            "it is the OIDC audience; the token would never match",
        )

    if settings.retrieval_min_similarity is None:
        ok("RETRIEVAL_MIN_SIMILARITY is unset, as it must be")
    else:
        warn(
            "RETRIEVAL_MIN_SIMILARITY is set",
            "each embedder carries the floor for its own vector space",
        )


def check_database() -> None:
    print("\n=== database ===")

    from app.db.base import SERVER_SETTINGS

    if SERVER_SETTINGS.get("random_page_cost") == "1.1":
        ok("random_page_cost travels with the application")
    else:
        fail("random_page_cost is not set per-connection")

    if SERVER_SETTINGS.get("hnsw.iterative_scan"):
        ok("hnsw.iterative_scan set", SERVER_SETTINGS["hnsw.iterative_scan"])
    else:
        fail(
            "hnsw.iterative_scan is not set",
            "filtered vector queries silently return nothing at scale",
        )

    from app.config import get_settings

    settings = get_settings()
    if settings.uses_cloud_sql:
        ok("Cloud SQL connector configured", settings.cloud_sql_instance)
        if settings.cloud_sql_instance.count(":") == 2:
            ok("instance connection name looks right", "project:region:instance")
        else:
            fail(
                "CLOUD_SQL_INSTANCE is not project:region:instance",
                settings.cloud_sql_instance,
            )
        if settings.database_url == "postgresql+asyncpg://":
            ok("no host in the DSN — the connector supplies it")
        else:
            fail("a host survived in the DSN", "it would dial somewhere else")
    else:
        warn(
            "CLOUD_SQL_INSTANCE unset",
            "fine locally; set it to project:region:instance when deploying",
        )

    print(
        "  [NOTE] Cloud SQL needs the `random_page_cost` database flag set to\n"
        "         1.1 as well — the per-connection setting covers the app, not\n"
        "         psql sessions or anything else that connects."
    )
    print(
        "  [NOTE] The service account needs roles/cloudsql.client, which\n"
        "         provision_gcp.sh already grants."
    )


def check_migrations() -> None:
    print("\n=== migrations ===")
    versions = sorted((REPO / "alembic" / "versions").glob("*.py"))
    if versions:
        ok(f"{len(versions)} migration(s) present")
        print(
            "  [NOTE] Cloud Run does not run migrations. `alembic upgrade head`\n"
            "         is a deliberate step before the first deploy of a revision."
        )
    else:
        fail("no migrations found")


def main() -> int:
    os.environ.setdefault("PYTHONPATH", str(REPO))
    print("Deployment preflight — read-only, makes no API calls")

    check_vertex_location()
    check_container()
    check_runtime_config()
    check_database()
    check_migrations()

    print()
    if failures:
        print(f"{failures} blocking issue(s), {warnings} warning(s)")
        return 1
    print(f"No blocking issues. {warnings} warning(s) — expected while APP_ENV=local.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
