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
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def sync_database_url(self) -> str:
        """Alembic runs its migrations synchronously, over psycopg."""
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()