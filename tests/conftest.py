"""Test harness.

Every test runs inside an outer transaction that is rolled back afterwards, so
the suite leaves no rows behind and needs no separate test database. Sessions
join that transaction with `create_savepoint`, which means application code
calling `session.commit()` releases a savepoint rather than committing for
real — the rollback still wins.
"""

import io
import uuid
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.config import get_settings
from app.db.base import SERVER_SETTINGS, get_db
from app.main import app


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
        get_settings().database_url,
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
    anyone ingests a real paper locally. Rather than police that by review,
    every test starts with rows it does not own, so a global query is wrong
    immediately and visibly instead of months later.

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
        storage_uri=f"file://decoy-{uuid.uuid4()}.pdf",
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
        app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _offline_by_default(monkeypatch: pytest.MonkeyPatch):
    """Keep the suite hermetic.

    A developer with GEMINI_API_KEY in their `.env` would otherwise have every
    test that reaches `get_embedder()` or `get_analyzer()` make real API calls
    — burning quota, adding seconds per test, and making results depend on the
    network. Tests that want a real backend set the variable themselves via
    `settings_env`, which runs after this and wins.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("VERTEX_PROJECT", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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


@pytest.fixture
def dev_auth(settings_env):
    """Enable the local-development auth bypass, as a subject unique to this test.

    Deliberately *not* the `local-dev-user` a developer uses by hand. Sharing
    that subject makes every test see whatever papers happen to be sitting in
    the local database — so uploading a demo paper breaks the isolation suite,
    which is both alarming and entirely spurious. A fresh subject per test
    means each one starts with an empty account.
    """
    subject = f"test-{uuid.uuid4()}"
    settings_env(app_env="local", auth_dev_bypass_subject=subject)
    return subject


@pytest.fixture
def storage_dir(tmp_path, settings_env):
    """Point object storage at a throwaway directory."""
    target = tmp_path / "storage"
    settings_env(local_storage_dir=str(target))
    return target


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
