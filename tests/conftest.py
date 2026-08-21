"""Test harness.

Every test runs inside an outer transaction that is rolled back afterwards, so
the suite leaves no rows behind and needs no separate test database. Sessions
join that transaction with `create_savepoint`, which means application code
calling `session.commit()` releases a savepoint rather than committing for
real — the rollback still wins.

Two things changed when the application became production-only, and both are
deliberately confined to this file:

**Settings.** The application requires Firebase, Cloud SQL, GCS, Cloud Tasks
and Vertex to be configured, and would otherwise refuse to import. The block
below supplies obviously-fake values *before* `app` is imported, and because
environment variables outrank `.env`, the suite can never pick up real
credentials from a developer's file. A test that escapes its fakes therefore
fails trying to reach `test-project`, rather than quietly billing the real one.

**Backends.** There are no production fallbacks left to lean on, so the fakes
in `tests/fakes.py` are substituted at the module-level factories. That is why
application code calls `embeddings.get_embedder()` rather than importing the
name: one patch point instead of one per importing module.
"""

import io
import os
import uuid
from collections.abc import AsyncGenerator

# Before any `app` import: these decide whether Settings validates at all.
os.environ.setdefault("FIREBASE_PROJECT_ID", "test-project")
os.environ.setdefault("CLOUD_SQL_INSTANCE", "test-project:us-central1:test-instance")
os.environ.setdefault("DB_USER", "test")
os.environ.setdefault("DB_PASSWORD", "test")
os.environ.setdefault("DB_NAME", "test")
os.environ.setdefault("STORAGE_BUCKET", "test-bucket")
os.environ.setdefault("VERTEX_PROJECT", "test-project")
os.environ.setdefault("CLOUD_TASKS_QUEUE", "test-queue")
os.environ.setdefault(
    "SERVICE_ACCOUNT_EMAIL", "test-sa@test-project.iam.gserviceaccount.com"
)
os.environ.setdefault("SERVICE_BASE_URL", "https://test-service.invalid")
os.environ.setdefault("RETRIEVAL_MIN_SIMILARITY", "")

# The one database this repository talks to that is not Cloud SQL. The suite
# needs a throwaway Postgres with pgvector; `docker compose up -d db` provides
# it, and `ALEMBIC_DATABASE_URL` points migrations at the same place. The
# application has no knowledge of this and no code path that would use it.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://app:app_local_dev_password@localhost:5432/paper_companion",
)

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool  # noqa: E402

from app.auth.dependencies import Principal, get_current_user  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db.base import SERVER_SETTINGS, get_db  # noqa: E402
from app.main import app  # noqa: E402
from tests import fakes  # noqa: E402


@pytest_asyncio.fixture
async def db_connection() -> AsyncGenerator[AsyncConnection, None]:
    # A per-test engine on NullPool, rather than the application's module-level
    # engine: pytest-asyncio runs each test in a fresh event loop, and a pooled
    # asyncpg connection carried over from a closed loop fails with "another
    # operation is in progress".
    #
    # It carries the application's SERVER_SETTINGS, though. Without them the
    # suite runs on planner defaults the application never uses, so a query
    # that only misbehaves under the real configuration passes here — which is
    # how the HNSW post-filtering bug stayed invisible.
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=NullPool,
        connect_args={"server_settings": SERVER_SETTINGS},
    )
    try:
        async with engine.connect() as connection:
            transaction = await connection.begin()
            try:
                yield connection
            finally:
                await transaction.rollback()
    finally:
        await engine.dispose()


async def _seed_decoy_data(session: AsyncSession) -> None:
    """Put data in the database that no test owns.

    A test that queries global state — `count(*) FROM papers`, or "the only
    concept" — passes by accident on an empty database and breaks the moment
    anyone ingests a real paper. Rather than police that by review, every test
    starts with rows it does not own, so a global query is wrong immediately
    and visibly instead of months later.

    This lives inside the test's transaction and rolls back with it.
    """
    from app.db.models import Concept, Paper, User

    stranger = User(auth_subject=f"decoy-{uuid.uuid4()}")
    session.add(stranger)
    await session.flush()

    # Deliberately never granted to the test user, so anything that respects
    # `user_paper_access` still sees nothing.
    paper = Paper(
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        storage_uri=f"memory://decoy-{uuid.uuid4()}.pdf",
        processing_status="ready",
        title="decoy paper owned by nobody in this test",
        embedding_model="some-other-model",
    )
    session.add(paper)
    await session.flush()

    session.add(
        Concept(
            user_id=stranger.user_id,
            canonical_name="Decoy Concept",
            normalized_name=f"decoy concept {uuid.uuid4()}",
            source_paper_ids=[paper.paper_id],
        )
    )
    await session.flush()


@pytest_asyncio.fixture
async def db_session(db_connection: AsyncConnection) -> AsyncGenerator[AsyncSession, None]:
    factory = async_sessionmaker(
        bind=db_connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    async with factory() as session:
        await _seed_decoy_data(session)
        yield session


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as http_client:
            yield http_client
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture(autouse=True)
def fake_backends(monkeypatch: pytest.MonkeyPatch):
    """Keep the suite hermetic.

    Nothing here has a local fallback any more: `get_embedder()` builds a real
    Vertex client and `get_storage()` reaches a real bucket. Without this every
    test that touched either would make live calls — burning quota, adding
    seconds per test, and making results depend on the network.

    Patched at the module attribute, which is the whole reason application code
    calls `embeddings.get_embedder()` instead of importing the name. A test
    that wants the real thing patches it back.
    """
    from app.services import adjudication, analysis, embeddings, quizzes, storage

    memory_storage = fakes.InMemoryStorage()

    monkeypatch.setattr(embeddings, "get_embedder", fakes.HashingEmbedder)
    monkeypatch.setattr(analysis, "get_analyzer", fakes.HeuristicAnalyzer)
    monkeypatch.setattr(adjudication, "get_adjudicator", fakes.ConservativeAdjudicator)
    monkeypatch.setattr(quizzes, "get_quiz_author", fakes.StubQuizAuthor)
    monkeypatch.setattr(quizzes, "get_grader", fakes.StubGrader)
    monkeypatch.setattr(storage, "get_storage", lambda: memory_storage)

    get_settings.cache_clear()
    yield memory_storage
    get_settings.cache_clear()


@pytest.fixture
def storage_backend(fake_backends) -> fakes.InMemoryStorage:
    """The in-memory bucket this test's uploads land in."""
    return fake_backends


@pytest.fixture
def settings_env(monkeypatch: pytest.MonkeyPatch):
    """Rewrite settings for one test, clearing the lru_cache on both sides.

    `None` sets the variable to empty rather than deleting it. Deleting only
    removes it from the process environment, and pydantic-settings would then
    fall back to the value in `.env` — which would make the suite behave
    differently depending on what the developer happens to have configured.
    An empty string is falsy for every optional setting, and it overrides the
    dotenv file.
    """

    def _apply(**values: str | None) -> None:
        for key, value in values.items():
            if value is None:
                monkeypatch.setenv(key.upper(), "")
            else:
                monkeypatch.setenv(key.upper(), value)
        get_settings.cache_clear()

    yield _apply
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def signed_in(db_session: AsyncSession) -> AsyncGenerator[str, None]:
    """A signed-in caller, as a subject unique to this test.

    This replaces the development auth bypass, which no longer exists: there is
    one authentication path now, and it verifies a real Firebase token. Minting
    one per test would mean network calls and a live Firebase project, so the
    harness overrides the dependency instead — the routes below it are
    unchanged, and `get_current_user` itself is covered by `test_auth.py`.

    A fresh subject per test, deliberately: a shared one would make every test
    see whatever papers happen to be in the database, so one test's upload
    would break another's isolation assertions.
    """
    from app.db.models import User

    subject = f"test-{uuid.uuid4()}"
    user = User(
        auth_subject=subject,
        email=f"{subject}@example.test",
        display_name="Test Reader",
    )
    db_session.add(user)
    await db_session.flush()

    principal = Principal(
        user_id=user.user_id, auth_subject=subject, email=user.email
    )
    app.dependency_overrides[get_current_user] = lambda: principal
    try:
        yield subject
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def build_pdf(pages: list[str]) -> bytes:
    """A real PDF carrying the given text, one entry per page.

    Generated rather than checked in, so a test's input is visible in the test
    that uses it.
    """
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER

    for body in pages:
        cursor = height - 72
        for line in body.split("\n"):
            pdf.setFont("Helvetica", 11)
            pdf.drawString(72, cursor, line[:110])
            cursor -= 15
            if cursor < 72:  # spill onto the same page rather than losing text
                cursor = height - 72
        pdf.showPage()

    pdf.save()
    return buffer.getvalue()


def build_pdf_with_hidden_text(visible: str, hidden: str) -> bytes:
    """A PDF carrying text a human cannot see.

    White-on-white is the classic prompt-injection vector for documents: the
    reader sees a normal paper, the extractor sees the attacker's text.
    """
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    _, height = LETTER

    pdf.setFont("Helvetica", 11)
    pdf.setFillColorRGB(0, 0, 0)
    cursor = height - 72
    for line in visible.split("\n"):
        pdf.drawString(72, cursor, line[:110])
        cursor -= 15

    # Same page, white ink on the white background.
    pdf.setFillColorRGB(1, 1, 1)
    for line in hidden.split("\n"):
        pdf.drawString(72, cursor, line[:110])
        cursor -= 15

    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def build_two_column_pdf(left: list[str], right: list[str]) -> bytes:
    """A two-column page, the layout that breaks naive reading order."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    _, height = LETTER
    pdf.setFont("Helvetica", 10)
    pdf.setFillColorRGB(0, 0, 0)

    for x, column in ((60, left), (330, right)):
        cursor = height - 90
        for line in column:
            pdf.drawString(x, cursor, line[:44])
            cursor -= 13

    pdf.showPage()
    pdf.save()
    return buffer.getvalue()
