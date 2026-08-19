"""Memory inside the turn pipeline: prefetch, `memory_read`, and the backstop.

`tests/test_memory.py` and `tests/test_signals.py` cover the services. These
cover the wiring — that a turn cannot claim to have read memory it did not
read, and cannot quietly stop accumulating it either.
"""

import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.runner import AgentOutcome
from app.db.models import (
    Chunk,
    Concept,
    Observation,
    Paper,
    Session,
    Turn,
    User,
    UserPaperAccess,
)
from app.ingestion.concepts import normalize_name
from app.schemas.sse import decode
from app.services.embeddings import get_embedder
from app.services.storage import LocalStorage
from tests.conftest import build_pdf

PAGES = [
    "Variational Inference\nAbstract\nThe evidence lower bound is optimised.\n"
    "1 Introduction\nVariational inference approximates an intractable posterior.",
    "2 Method\nThe evidence lower bound decomposes into reconstruction and KL.\n"
    "3 Results\nOptimising the bound improves held out likelihood here.",
]


@pytest.fixture
def storage(storage_dir) -> LocalStorage:
    return LocalStorage(storage_dir)


@pytest.fixture
def agent(monkeypatch):
    """A scripted agent that can also be told to record a signal."""
    captured: dict = {}

    def script(draft: str, *, signals: list[tuple[str, str]] | None = None):
        async def fake_run_turn(
            *,
            context,
            history,
            user_message,
            paper_title,
            session_key,
            **hints,
        ):
            captured.update(hints)
            captured["paper_scope"] = list(context.paper_scope)
            from app.agent.tools import (
                build_record_learning_signal,
                build_retrieve_paper_context,
            )

            await build_retrieve_paper_context(context)("evidence lower bound")
            for concept_name, signal_type in signals or []:
                await build_record_learning_signal(context)(
                    concept_name=concept_name, signal_type=signal_type
                )
            return AgentOutcome(
                text=draft,
                tools_called=list(context.tools_called),
                input_tokens=10,
                output_tokens=5,
            )

        monkeypatch.setattr("app.services.turns.run_turn", fake_run_turn)
        return captured

    return script


async def _reader(
    db_session: AsyncSession, storage: LocalStorage
) -> tuple[User, Paper, Session]:
    from app.ingestion.pipeline import ingest_paper
    from app.services.embeddings import HashingEmbedder

    user = User(auth_subject=f"turn-memory-{uuid.uuid4()}")
    db_session.add(user)
    await db_session.flush()

    content_hash = uuid.uuid4().hex + uuid.uuid4().hex[:32]
    paper = Paper(
        content_hash=content_hash,
        storage_uri=storage.put(build_pdf(PAGES), content_hash=content_hash),
        processing_status="queued",
    )
    db_session.add(paper)
    await db_session.flush()
    db_session.add(UserPaperAccess(user_id=user.user_id, paper_id=paper.paper_id))
    await db_session.flush()

    await ingest_paper(
        db_session,
        paper.paper_id,
        user_id=user.user_id,
        embedder=HashingEmbedder(),
        storage=storage,
    )

    conversation = Session(user_id=user.user_id, active_paper_id=paper.paper_id)
    db_session.add(conversation)
    await db_session.flush()
    return user, paper, conversation


async def _run(db_session, conversation, user, message: str) -> list[dict]:
    from app.services.turns import TurnPipeline

    frames = []
    async for frame in TurnPipeline(db_session).run(
        conversation, user.user_id, message
    ):
        frames.append(frame)
    return decode("".join(frames))


async def _seed_concept(db_session, user, name: str, **kwargs) -> Concept:
    """Give a concept a learned state.

    Upserts rather than inserts: ingesting the paper already canonicalizes its
    concepts for this user, so the row usually exists. Setting the state on it
    is also what actually happens — a concept is learned about after it is
    discovered, not instead.
    """
    normalized = normalize_name(name)
    concept = await db_session.scalar(
        select(Concept).where(
            Concept.user_id == user.user_id,
            Concept.normalized_name == normalized,
        )
    )
    if concept is None:
        concept = Concept(
            user_id=user.user_id,
            canonical_name=name,
            normalized_name=normalized,
            embedding=get_embedder().embed_query(name),
        )
        db_session.add(concept)

    for attribute, value in kwargs.items():
        setattr(concept, attribute, value)
    if concept.embedding is None:
        concept.embedding = get_embedder().embed_query(name)

    await db_session.flush()
    return concept


# --------------------------------------------------------------------------
# memory_read is derived, never asserted
# --------------------------------------------------------------------------


async def test_a_first_turn_reads_no_memory_and_says_so(
    db_session: AsyncSession, storage, agent
):
    """Nothing recorded yet — the honest answer is an empty event."""
    agent("The bound decomposes [1].")
    user, _, conversation = await _reader(db_session, storage)

    events = await _run(db_session, conversation, user, "what is the ELBO?")

    memory_event = next(e for e in events if e["event"] == "memory_used")
    assert memory_event["memory"] == []

    turn = await db_session.scalar(
        select(Turn).where(Turn.session_id == conversation.session_id)
    )
    assert turn.memory_read is False


async def test_a_matching_concept_is_prefetched_and_reported(
    db_session: AsyncSession, storage, agent
):
    """Step 4 is unconditional: the agent never had to ask for this."""
    agent("The bound decomposes [1].")
    user, _, conversation = await _reader(db_session, storage)
    await _seed_concept(
        db_session,
        user,
        "evidence lower bound",
        understanding_score=0.3,
        score_confidence=0.8,
        effective_style="numerical",
    )

    events = await _run(db_session, conversation, user, "evidence lower bound")

    # Ingesting the paper canonicalizes several nearby concepts, so the
    # prefetch legitimately returns more than the seeded one.
    memory_event = next(e for e in events if e["event"] == "memory_used")
    seeded = next(
        record
        for record in memory_event["memory"]
        if record["name"] == "evidence lower bound"
    )
    assert seeded["effective_style"] == "numerical"
    assert seeded["understanding_score"] == pytest.approx(0.3, abs=0.01)

    turn = await db_session.scalar(
        select(Turn).where(Turn.session_id == conversation.session_id)
    )
    assert turn.memory_read is True


async def test_the_prefetch_reaches_the_agent_as_a_summary(
    db_session: AsyncSession, storage, agent
):
    """Scores and a style — never conversation text."""
    captured = agent("The bound decomposes [1].")
    user, _, conversation = await _reader(db_session, storage)
    await _seed_concept(
        db_session,
        user,
        "evidence lower bound",
        understanding_score=0.3,
        score_confidence=0.8,
        effective_style="numerical",
    )

    await _run(db_session, conversation, user, "evidence lower bound")

    summary = captured["memory_summary"]
    assert summary is not None
    assert "evidence lower bound" in summary
    assert "numerical" in summary


async def test_prefetch_does_not_cross_users(db_session: AsyncSession, storage, agent):
    agent("The bound decomposes [1].")
    user, _, conversation = await _reader(db_session, storage)

    stranger = User(auth_subject=f"stranger-{uuid.uuid4()}")
    db_session.add(stranger)
    await db_session.flush()
    theirs = await _seed_concept(
        db_session,
        stranger,
        "evidence lower bound",
        understanding_score=0.1,
        score_confidence=0.9,
    )

    events = await _run(db_session, conversation, user, "evidence lower bound")

    # The reader has a concept of the same name from their own ingest, so the
    # assertion is about identity, not emptiness: the stranger's row must not
    # appear no matter how well it matches.
    returned = {
        record["concept_id"]
        for record in next(e for e in events if e["event"] == "memory_used")["memory"]
    }
    assert str(theirs.concept_id) not in returned


# --------------------------------------------------------------------------
# Signals and the backstop
# --------------------------------------------------------------------------


async def test_an_agent_signal_is_persisted_with_its_turn(
    db_session: AsyncSession, storage, agent
):
    agent(
        "The bound decomposes [1].",
        signals=[("evidence lower bound", "explicit_confusion")],
    )
    user, _, conversation = await _reader(db_session, storage)

    await _run(db_session, conversation, user, "what is the ELBO?")

    turn = await db_session.scalar(
        select(Turn).where(Turn.session_id == conversation.session_id)
    )
    observation = await db_session.scalar(
        select(Observation).where(Observation.user_id == user.user_id)
    )

    assert observation is not None
    assert observation.signal_type == "explicit_confusion"
    # The provenance link buffering exists to preserve.
    assert observation.turn_id == turn.turn_id
    assert observation.session_id == conversation.session_id


async def test_the_backstop_fires_when_the_agent_records_nothing(
    db_session: AsyncSession, storage, agent
):
    """ARCHITECTURE 14.2 — the agent forgetting must not stop accumulation."""
    agent("The bound decomposes [1].")
    user, _, conversation = await _reader(db_session, storage)
    await _seed_concept(
        db_session,
        user,
        "evidence lower bound",
        understanding_score=0.3,
        score_confidence=0.8,
    )

    await _run(db_session, conversation, user, "evidence lower bound")

    observation = await db_session.scalar(
        select(Observation).where(Observation.user_id == user.user_id)
    )
    assert observation is not None
    assert observation.signal_type == "reinforcement"
    assert observation.signal_source == "system"
    # Zero weight: it records that the concept came up, and claims nothing more.
    assert observation.weight == pytest.approx(0.0)


async def test_the_backstop_does_not_fire_when_the_agent_did_record(
    db_session: AsyncSession, storage, agent
):
    agent(
        "The bound decomposes [1].",
        signals=[("evidence lower bound", "explicit_confusion")],
    )
    user, _, conversation = await _reader(db_session, storage)
    await _seed_concept(
        db_session,
        user,
        "evidence lower bound",
        understanding_score=0.3,
        score_confidence=0.8,
    )

    await _run(db_session, conversation, user, "evidence lower bound")

    kinds = (
        await db_session.scalars(
            select(Observation.signal_type).where(Observation.user_id == user.user_id)
        )
    ).all()
    assert list(kinds) == ["explicit_confusion"]


async def test_a_turn_with_no_memory_at_all_writes_no_backstop(
    db_session: AsyncSession, storage, agent
):
    """The backstop needs a concept to attach to. Inventing one would be worse
    than recording nothing."""
    agent("The bound decomposes [1].")
    user, _, conversation = await _reader(db_session, storage)

    await _run(db_session, conversation, user, "what is the ELBO?")

    written = await db_session.scalar(
        select(func.count())
        .select_from(Observation)
        .where(Observation.user_id == user.user_id)
    )
    assert written == 0


# --------------------------------------------------------------------------
# The cross-paper callback, through the pipeline (ARCHITECTURE 12)
# --------------------------------------------------------------------------


async def _prior_paper_the_reader_struggled_with(
    db_session: AsyncSession, user: User, *, grant: bool = True
) -> tuple[Paper, Concept]:
    """A second paper, a weak concept from it, and an edge into this one."""
    prior = Paper(
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        storage_uri=f"file://prior-{uuid.uuid4()}.pdf",
        processing_status="ready",
        title="Auto-Encoding Variational Bayes",
    )
    db_session.add(prior)
    await db_session.flush()
    if grant:
        db_session.add(UserPaperAccess(user_id=user.user_id, paper_id=prior.paper_id))
        await db_session.flush()

    struggled = await _seed_concept(
        db_session,
        user,
        "Reparameterization trick",
        source_paper_ids=[prior.paper_id],
        understanding_score=0.31,
        score_confidence=0.72,
        effective_style="numerical",
    )
    asked_about = await _seed_concept(
        db_session,
        user,
        "evidence lower bound",
        understanding_score=0.8,
        score_confidence=0.6,
    )

    from app.db.models import ConceptRelationship

    db_session.add(
        ConceptRelationship(
            user_id=user.user_id,
            source_concept_id=struggled.concept_id,
            target_concept_id=asked_about.concept_id,
            relationship_type="component_of",
            confidence=0.86,
            discovery_method="model",
        )
    )
    await db_session.flush()
    return prior, struggled


async def test_a_cross_paper_callback_is_recorded_on_the_turn(
    db_session: AsyncSession, storage, agent
):
    """Step 13 — `callback_concept_id` with `memory_read`, which the CHECK
    constraint `callback_requires_memory` would reject otherwise."""
    agent("The bound decomposes [1].")
    user, _, conversation = await _reader(db_session, storage)
    _, struggled = await _prior_paper_the_reader_struggled_with(db_session, user)

    await _run(db_session, conversation, user, "evidence lower bound")

    turn = await db_session.scalar(
        select(Turn).where(Turn.session_id == conversation.session_id)
    )
    assert turn.callback_concept_id == struggled.concept_id
    assert turn.memory_read is True
    assert turn.callback_suppressed_reason is None
    assert turn.agent_action == "callback"
    assert turn.explanation_style == "numerical"


async def test_the_callback_expands_scope_to_exactly_two_papers(
    db_session: AsyncSession, storage, agent
):
    """ARCHITECTURE 12 step 7 — never "search all my papers"."""
    captured = agent("The bound decomposes [1].")
    user, active, conversation = await _reader(db_session, storage)
    prior, _ = await _prior_paper_the_reader_struggled_with(db_session, user)

    await _run(db_session, conversation, user, "evidence lower bound")

    assert set(captured["paper_scope"]) == {active.paper_id, prior.paper_id}
    assert captured["callback_hint"] is not None


async def test_a_revoked_grant_keeps_the_prior_paper_out_of_scope(
    db_session: AsyncSession, storage, agent
):
    """The turn still answers; it just answers without the callback."""
    captured = agent("The bound decomposes [1].")
    user, active, conversation = await _reader(db_session, storage)
    await _prior_paper_the_reader_struggled_with(db_session, user, grant=False)

    await _run(db_session, conversation, user, "evidence lower bound")

    assert captured["paper_scope"] == [active.paper_id]
    assert captured["callback_hint"] is None

    turn = await db_session.scalar(
        select(Turn).where(Turn.session_id == conversation.session_id)
    )
    assert turn.callback_concept_id is None
    assert turn.callback_suppressed_reason == "grant_revoked"


async def test_a_suppressed_callback_still_records_why(
    db_session: AsyncSession, storage, agent
):
    """Suppression is a feature and is measured — never a silent path."""
    agent("The bound decomposes [1].")
    user, _, conversation = await _reader(db_session, storage)

    await _run(db_session, conversation, user, "what is the ELBO?")

    turn = await db_session.scalar(
        select(Turn).where(Turn.session_id == conversation.session_id)
    )
    assert turn.callback_concept_id is None
    assert turn.callback_suppressed_reason is not None


# --------------------------------------------------------------------------
# Deterministic quiz routing (ARCHITECTURE 9.2 step 3, 11)
# --------------------------------------------------------------------------


async def test_a_pending_quiz_routes_straight_to_grading(
    db_session: AsyncSession, storage, agent, monkeypatch
):
    """No agent loop, and above all no classification call.

    Asking a model "is this a quiz answer?" moments after asking the question
    is a wasted call and a source of nondeterminism at the most measured point
    in the system, so the route is taken from session state alone.
    """
    from app.db.models import Quiz, QuizAttempt
    from app.services.quizzes import Grading

    captured = agent("This should never be composed.")
    user, paper, conversation = await _reader(db_session, storage)
    concept = await _seed_concept(db_session, user, "evidence lower bound")

    chunk_id = await db_session.scalar(
        select(Chunk.chunk_id).where(Chunk.paper_id == paper.paper_id).limit(1)
    )
    quiz = Quiz(
        user_id=user.user_id,
        concept_id=concept.concept_id,
        paper_id=paper.paper_id,
        question="What does the bound decompose into?",
        rubric={"must_mention": ["reconstruction", "KL"]},
        grounding_chunk_ids=[chunk_id],
    )
    db_session.add(quiz)
    await db_session.flush()
    conversation.activity = "QUIZ_PENDING"
    conversation.pending_quiz_id = quiz.quiz_id
    await db_session.flush()

    monkeypatch.setattr(
        "app.services.quizzes.get_grader",
        lambda: type(
            "G",
            (),
            {
                "model_name": "scripted",
                "grade": lambda self, **_: Grading(
                    grade="correct", missing_elements=[], confidence=0.9
                ),
            },
        )(),
    )

    events = await _run(
        db_session, conversation, user, "Reconstruction and the KL term."
    )

    # The agent was never invoked: its capture dict stays untouched.
    assert "memory_summary" not in captured

    answer = "".join(e["text"] for e in events if e["event"] == "token")
    assert "right" in answer.lower()

    attempt = await db_session.scalar(
        select(QuizAttempt).where(QuizAttempt.quiz_id == quiz.quiz_id)
    )
    assert attempt.grade == "correct"

    # Step 10 — the transition always happens.
    assert conversation.activity == "FREE"
    assert conversation.pending_quiz_id is None

    turn = await db_session.scalar(
        select(Turn)
        .where(Turn.session_id == conversation.session_id)
        .order_by(Turn.ordinal.desc())
    )
    assert turn.agent_action == "quiz_move_forward"
    assert turn.grounding_status == "n/a"


async def test_a_graded_answer_becomes_a_learning_signal(
    db_session: AsyncSession, storage, agent, monkeypatch
):
    """ARCHITECTURE 11 step 8 — at the highest weight class."""
    from app.db.models import Quiz
    from app.services.quizzes import Grading

    agent("unused")
    user, paper, conversation = await _reader(db_session, storage)
    concept = await _seed_concept(db_session, user, "evidence lower bound")

    chunk_id = await db_session.scalar(
        select(Chunk.chunk_id).where(Chunk.paper_id == paper.paper_id).limit(1)
    )
    quiz = Quiz(
        user_id=user.user_id,
        concept_id=concept.concept_id,
        paper_id=paper.paper_id,
        question="What does the bound decompose into?",
        rubric={"must_mention": ["reconstruction"]},
        grounding_chunk_ids=[chunk_id],
    )
    db_session.add(quiz)
    await db_session.flush()
    conversation.activity = "QUIZ_PENDING"
    conversation.pending_quiz_id = quiz.quiz_id
    await db_session.flush()

    monkeypatch.setattr(
        "app.services.quizzes.get_grader",
        lambda: type(
            "G",
            (),
            {
                "model_name": "scripted",
                "grade": lambda self, **_: Grading(
                    grade="incorrect", missing_elements=["reconstruction"], confidence=0.8
                ),
            },
        )(),
    )

    await _run(db_session, conversation, user, "no idea")

    observation = await db_session.scalar(
        select(Observation).where(Observation.user_id == user.user_id)
    )
    assert observation.signal_type == "quiz_incorrect"
    assert observation.signal_source == "quiz"
    assert observation.quiz_attempt_id is not None
    assert observation.turn_id is not None
    assert conversation.activity == "EXPLAINING"
