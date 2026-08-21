from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "local"

    db_user: str
    db_password: str
    db_name: str
    db_host: str
    db_port: int

    # Firebase project whose ID tokens this deployment accepts. Verification is
    # refused outright when unset — an unpinned audience would accept tokens
    # minted by any Firebase project.
    firebase_project_id: str | None = None

    # Local development only. When set, requests are authenticated as this
    # Firebase subject with no token presented. Refused outright unless
    # app_env == "local", so a stray value in a deployed environment fails
    # closed rather than opening a hole.
    auth_dev_bypass_subject: str | None = None

    # Cloud SQL, as an instance connection name: `project:region:instance`.
    # Unset for every local run, where `DB_HOST`/`DB_PORT` reach Postgres in
    # Docker directly. Set, it switches both engines onto the Cloud SQL Python
    # connector, which handles TLS and IAM without a proxy sidecar or a
    # long-lived certificate.
    #
    # The unix socket at /cloudsql/<instance> is the other supported route on
    # Cloud Run and needs no code at all — leave this unset and point DB_HOST
    # at the socket path. The connector is preferred because it also works
    # from a laptop, which is where the pre-deploy `alembic upgrade head` runs.
    cloud_sql_instance: str | None = None
    # PUBLIC unless the instance is on a VPC, in which case PRIVATE.
    cloud_sql_ip_type: str = "PUBLIC"

    # Object storage. Local development writes to a directory; deployment sets
    # storage_bucket and the GCS backend takes over.
    storage_bucket: str | None = None
    local_storage_dir: str = ".storage"

    # Upload guards, applied at the boundary before anything is persisted.
    max_upload_bytes: int = 30 * 1024 * 1024
    max_page_count: int = 200

    # Gemini access. Two transports reach the same models:
    #
    #   gemini_api_key  Google AI Studio (https://aistudio.google.com/apikey).
    #                   Free tier, no billing account. The development path.
    #   vertex_project  Vertex AI via application default credentials. Needs a
    #                   billing-enabled project. The deployment path.
    #
    # With both set, Vertex wins: a deployment that has been given a project
    # should not silently keep using a developer's personal API key. With
    # neither, the deterministic local stubs are used.
    gemini_api_key: str | None = None
    vertex_project: str | None = None
    vertex_location: str = "us-central1"

    # Pinned deliberately, not `gemini-flash-latest`. HK-1 requires a
    # Flash-class Gemini 3.5+, and an alias reports no resolvable version — it
    # cannot be shown to satisfy the mandate, and it can move underneath a
    # recorded demo. This resolves to an explicit dated build.
    gemini_model: str = "gemini-3.5-flash"

    # Retrieval tuning. Leave the floor unset: cosine scores are not comparable
    # between embedding models, so each embedder carries the floor for its own
    # vector space (0.25 for the lexical stub, 0.58 for gemini-embedding-001).
    # Set this only to override both deliberately.
    retrieval_top_k: int = 8
    retrieval_min_similarity: float | None = None

    @field_validator("retrieval_min_similarity", mode="before")
    @classmethod
    def _blank_means_unset(cls, value: object) -> object:
        """Treat `RETRIEVAL_MIN_SIMILARITY=` as "not set".

        A blank line in `.env` arrives as an empty string, which is not a
        float. Without this, copying `.env.example` verbatim crashes the app
        at import time — the setting is optional, so blank must mean absent.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def dev_bypass_active(self) -> bool:
        return bool(self.auth_dev_bypass_subject) and self.app_env == "local"

    @property
    def gemini_available(self) -> bool:
        """Is a real model backend configured, by either transport?

        Mirrors the branch `get_embedder()` and `get_adjudicator()` take, so a
        caller can refuse up front rather than discovering halfway through an
        ingest that concepts were silently skipped by the local stubs.
        """
        return bool(self.vertex_project or self.gemini_api_key)

    @property
    def uses_cloud_sql(self) -> bool:
        """Is this deployment talking to Cloud SQL through the connector?

        Set `CLOUD_SQL_INSTANCE` to an instance connection name
        (`project:region:instance`) to switch. Unset — every local run — keeps
        the plain host:port DSN and never imports the connector.
        """
        return bool(self.cloud_sql_instance)

    @property
    def database_url(self) -> str:
        """The async DSN.

        Over the Cloud SQL connector the host and port are supplied by the
        connector itself, so the URL carries only the driver: SQLAlchemy is
        handed an `async_creator` and never dials anything directly.
        """
        if self.uses_cloud_sql:
            return "postgresql+asyncpg://"
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def sync_database_url(self) -> str:
        """Alembic runs its migrations synchronously.

        psycopg locally, **pg8000 over the Cloud SQL connector**. The
        connector's psycopg driver talks over a unix domain socket, which
        Windows does not have — so migrating a cloud instance from a Windows
        laptop fails with `NotImplementedError` before it reaches the database.
        pg8000 uses TCP and works on every platform the team runs.
        """
        if self.uses_cloud_sql:
            return "postgresql+pg8000://"
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()