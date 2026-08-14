from functools import lru_cache

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

    # Vertex AI. Unset locally, which selects the deterministic hashing
    # embedder instead of gemini-embedding-001.
    vertex_project: str | None = None
    vertex_location: str = "us-central1"

    # Retrieval tuning. The floor is deliberately conservative: returning
    # nothing is a better failure than grounding an answer in a weak match.
    retrieval_top_k: int = 8
    retrieval_min_similarity: float = 0.25

    @property
    def dev_bypass_active(self) -> bool:
        return bool(self.auth_dev_bypass_subject) and self.app_env == "local"

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