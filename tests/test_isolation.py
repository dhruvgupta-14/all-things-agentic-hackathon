"""The suite's own isolation guarantees.

These tests are about the harness rather than the product. They exist because
the suite once passed only on an empty database: three tests asserted things
like `count(*) FROM papers == 1`, which was true until someone ingested a real
paper locally, at which point the isolation suite failed for entirely spurious
reasons.

The fix was structural — every test transaction is seeded with rows it does
not own — and these tests verify that guard is actually in force.
"""

import uuid

from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Concept, Paper, User, UserPaperAccess
from tests.fakes import (
    ConservativeAdjudicator,
    HashingEmbedder,
    HeuristicAnalyzer,
    StubGrader,
    StubQuizAuthor,
)


async def test_the_database_is_never_empty_during_a_test(db_session: AsyncSession):
    """The guard has teeth.

    If this fails, the decoy seeding has stopped working and global-state bugs
    will pass silently again.
    """
    papers = await db_session.scalar(select(func.count()).select_from(Paper))
    concepts = await db_session.scalar(select(func.count()).select_from(Concept))

    assert papers > 0
    assert concepts > 0


async def test_listing_papers_ignores_papers_this_user_was_not_granted(
    client: AsyncClient, signed_in: str
):
    """Papers exist. None are this user's. The list is empty."""
    await client.get("/api/me")

    assert (await client.get("/api/papers")).json() == []


async def test_a_fresh_user_owns_no_concepts(db_session: AsyncSession):
    """Concepts exist globally; a new user's own set is still empty."""
    user = User(auth_subject=f"isolation-{uuid.uuid4()}")
    db_session.add(user)
    await db_session.flush()

    mine = await db_session.scalar(
        select(func.count()).select_from(Concept).where(Concept.user_id == user.user_id)
    )
    everyone = await db_session.scalar(select(func.count()).select_from(Concept))

    assert mine == 0
    assert everyone > mine, "the decoy concept should be invisible but present"


async def test_scoped_counts_are_exact_while_global_counts_are_not(
    db_session: AsyncSession,
):
    """The pattern every test should follow.

    A scoped count is a fact about the test's own data. A global count is a
    fact about whatever happens to be in the developer's database, which is
    not something a test may assert on.
    """
    user = User(auth_subject=f"isolation-{uuid.uuid4()}")
    db_session.add(user)
    await db_session.flush()

    paper = Paper(
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        storage_uri=f"file://{uuid.uuid4()}.pdf",
        processing_status="ready",
    )
    db_session.add(paper)
    await db_session.flush()
    db_session.add(UserPaperAccess(user_id=user.user_id, paper_id=paper.paper_id))
    await db_session.flush()

    granted = await db_session.scalar(
        select(func.count())
        .select_from(UserPaperAccess)
        .where(UserPaperAccess.user_id == user.user_id)
    )
    assert granted == 1

    everything = await db_session.scalar(select(func.count()).select_from(Paper))
    assert everything > 1, "a global count sees data this test does not own"


async def test_each_test_gets_its_own_auth_subject(signed_in: str):
    """Sharing `local-dev-user` with a human developer leaks their uploads."""
    assert signed_in.startswith("test-")
    assert signed_in != "local-dev-user"


async def test_the_suite_makes_no_real_model_calls():
    """`pytest` must be runnable offline and must not spend API quota."""
    from app.services.adjudication import get_adjudicator
    from app.services.analysis import get_analyzer
    from app.services.embeddings import get_embedder
    from app.services.quizzes import get_grader, get_quiz_author

    assert isinstance(get_embedder(), HashingEmbedder)
    assert isinstance(get_analyzer(), HeuristicAnalyzer)
    assert isinstance(get_adjudicator(), ConservativeAdjudicator)
    # Every model-calling factory belongs here. A new one that forgets to add
    # itself is exactly how the suite starts quietly spending quota again.
    assert isinstance(get_quiz_author(), StubQuizAuthor)
    assert isinstance(get_grader(), StubGrader)


def test_the_suite_does_not_require_the_demo_pdfs():
    """The demo papers are integration fixtures, not unit-test fixtures.

    Every PDF the suite needs is generated in `conftest.build_pdf`, so the
    tests run on a machine that has never downloaded a paper.
    """
    import pathlib

    # Assembled rather than written literally, so this file does not match
    # its own check.
    needle = "demo_" + "papers"
    offenders = [
        path.name
        for path in pathlib.Path("tests").rglob("*.py")
        if path.name != pathlib.Path(__file__).name
        and needle in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"tests must not read the demo PDF directory: {offenders}"
