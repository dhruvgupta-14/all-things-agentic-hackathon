"""Configuration.

There is one deployment shape and no second mode. Every managed service this
application depends on — Firebase, Cloud SQL, Cloud Storage, Cloud Tasks,
Vertex AI — is **required**, so a misconfigured process fails at startup naming
the missing setting rather than starting and behaving differently.

That is a deliberate reversal. Earlier versions defaulted each of these to a
local substitute: a hashing embedder, a filesystem directory, in-process
background work, an auth bypass. Every one of those made a broken configuration
look healthy — the service answered questions, ingested papers and returned
citations, just not with anything real behind it. A missing setting is now a
crash, which is the only failure mode that cannot be mistaken for success.

Running on localhost is still fine. It is an address, not a mode: the process
reads the same settings, authenticates against the same Firebase project, and
talks to the same Cloud SQL instance and bucket as the deployed revision.
"""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Identity ---------------------------------------------------------
    # The Firebase project whose ID tokens this deployment accepts. Pinned, not
    # optional: verification checks a token's audience against this value, and
    # an unpinned audience would accept tokens minted by any Firebase project.
    firebase_project_id: str

    # --- Database ---------------------------------------------------------
    # Cloud SQL, as an instance connection name: `project:region:instance`.
    # Reached through the Cloud SQL Python connector, which handles TLS and IAM
    # without a proxy sidecar and works identically from Cloud Run and from a
    # laptop — which matters because `alembic upgrade head` runs from whichever
    # machine the operator is using, before the revision that needs it exists.
    cloud_sql_instance: str
    # PUBLIC unless the instance is on a VPC, in which case PRIVATE.
    cloud_sql_ip_type: str = "PUBLIC"
    db_user: str
    db_password: str
    db_name: str

    # --- Object storage ---------------------------------------------------
    # Private GCS bucket holding uploaded PDFs.
    storage_bucket: str

    # Upload guards, applied at the boundary before anything is persisted.
    max_upload_bytes: int = 30 * 1024 * 1024
    max_page_count: int = 200

    # --- Models -----------------------------------------------------------
    # Vertex AI over application default credentials: the metadata server on
    # Cloud Run, `gcloud auth application-default login` elsewhere.
    vertex_project: str
    # `global`, not a region, and this is not a preference. Gemini 3.x is not
    # served from regional endpoints: measured on this project, gemini-3.5-flash
    # returns 404 in us-central1 and only gemini-2.5-flash answers there, which
    # would silently drop the deployment below HK-1's "Flash-class 3.5+". The
    # global endpoint serves gemini-embedding-001 too, so one value covers both.
    vertex_location: str = "global"

    # Pinned deliberately, not `gemini-flash-latest`. HK-1 requires a
    # Flash-class Gemini 3.5+, and an alias reports no resolvable version — it
    # cannot be shown to satisfy the mandate, and it can move underneath a
    # recorded demo. This resolves to an explicit dated build.
    gemini_model: str = "gemini-3.5-flash"

    # --- Asynchronous work ------------------------------------------------
    # Cloud Tasks. Ingestion takes 30-60 seconds and is pushed back to
    # /internal/ingest on this same service, signed with an OIDC token.
    cloud_tasks_queue: str
    cloud_tasks_location: str = "us-central1"
    # The identity the queue mints its token as, and the identity
    # /internal/ingest requires. A push from anything else is refused.
    service_account_email: str
    # This service's own public https URL — the queue pushes back to it, and it
    # is the OIDC audience, which is why it is pinned rather than derived from
    # the request: `Host` is client-controlled.
    service_base_url: str
    # The project holding the queue. Defaults to `vertex_project`, the same
    # project in every deployment of this system; set it only if they diverge.
    gcp_project: str | None = None

    # --- Frontend ---------------------------------------------------------
    # The built SPA, served from this process at "/". Relative paths resolve
    # against the repository root, not the working directory.
    spa_dist_dir: str = "frontend/dist"

    # --- Retrieval --------------------------------------------------------
    # Leave the floor unset: cosine scores are not comparable between embedding
    # models, so the embedder carries the floor for its own vector space. Set
    # this only to override it deliberately.
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

    @field_validator("service_base_url")
    @classmethod
    def _must_be_https(cls, value: str) -> str:
        """It is the OIDC audience Cloud Tasks signs for.

        A mismatch is not a cosmetic problem: it is every ingestion push
        failing verification with a 401, and papers that never leave `queued`.
        """
        if not value.startswith("https://"):
            raise ValueError(
                f"SERVICE_BASE_URL must be an https URL, got {value!r}. It is the "
                "OIDC audience for Cloud Tasks pushes, not a display string."
            )
        return value.rstrip("/")

    @property
    def project_id(self) -> str:
        """The GCP project for Cloud Tasks."""
        return self.gcp_project or self.vertex_project

    @property
    def database_url(self) -> str:
        """The async DSN.

        Host and port are supplied by the Cloud SQL connector, so the URL
        carries only the driver: SQLAlchemy is handed an `async_creator` and
        never dials anything directly. A host here would silently dial
        somewhere else.
        """
        return "postgresql+asyncpg://"

    @property
    def sync_database_url(self) -> str:
        """Alembic runs its migrations synchronously.

        pg8000 rather than psycopg: the connector's psycopg driver talks over a
        unix domain socket, which Windows does not have — so migrating from a
        Windows laptop fails with `NotImplementedError` before it reaches the
        database. pg8000 uses TCP and works on every platform the team runs.
        """
        return "postgresql+pg8000://"


@lru_cache
def get_settings() -> Settings:
    return Settings()
