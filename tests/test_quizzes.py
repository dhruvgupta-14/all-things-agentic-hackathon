"""Adaptive learning checks (ARCHITECTURE 11).

The three things this has to get right, in order of how badly they would hurt:
a grade is never guessed, the rubric never reaches the agent, and the reader is
never stranded in QUIZ_PENDING.
"""

import hashlib
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Chunk,
    Concept,
    ConceptRelationship,
    Paper,
    Quiz,
    QuizAttempt,
    Section,
    Session,
    User,
)
from app.ingestion.concepts import normalize_name
from app.services.embeddings import get_embedder
from app.services.quizzes import (
    NEXT_EXPLAIN_DIFFERENTLY,
    NEXT_MOVE_FORWARD,
    NEXT_REVISIT_PREREQUISITE,
    REFUSED_ALREADY_PENDING,
    REFUSED_APPETITE_OFF,
    REFUSED_RATE_LIMITED,
    REFUSED_WELL_UNDERSTOOD,
    Grading,
    QuizService,
    QuizUnavailable,
    StubQuizAuthor,
)


class _ScriptedGrader:
    """Returns a fixed grade, and records that it was consulted."""

    model_name = "scripted"

    def __init__(self, grading: Grading | Exception, *, fail_times: int = 0) -> None:
        self._grading = grading
        self._fail_times = fail_times
        self.calls = 0

    def grade(self, *, question, rubric, answer):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise RuntimeError("grader had a bad day")
        if isinstance(self._grading, Exception):
            raise self._grading
        return self._grading


async def _setup(db_session: AsyncSession, **preferences):
    user = User(
        auth_subject=f"quiz-test-{uuid.uuid4()}", preferences=preferences or {}
    )
    db_session.add(user)
    await db_session.flush()

    paper = Paper(
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        storage_uri=f"file://{uuid.uuid4()}.pdf",
        processing_status="ready",
        title="A paper",
    )
    db_session.add(paper)
    await db_session.flush()

    section = Section(
        paper_id=paper.paper_id,
        section_path="2",
        ordinal=0,
        page_start=1,
        page_end=1,
        section_role="method",
    )
    db_session.add(section)
    await db_session.flush()

    content = "The evidence lower bound decomposes into reconstruction and KL."
    chunk = Chunk(
        paper_id=paper.paper_id,
        section_id=section.section_id,
        ordinal=0,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        token_count=12,
        page_start=1,
        page_end=1,
    )
    db_session.add(chunk)
    await db_session.flush()

    concept = Concept(
        user_id=user.user_id,
        canonical_name="Variational lower bound",
        normalized_name=normalize_name("Variational lower bound"),
        embedding=get_embedder().embed_query("Variational lower bound"),
    )
    db_session.add(concept)
    await db_session.flush()

    conversation = Session(user_id=user.user_id, active_paper_id=paper.paper_id)
    db_session.add(conversation)
    await db_session.flush()

    return user, paper, chunk, concept, conversation


async def _ask(db_session, user, paper, chunk, concept, conversation, grader=None):
    service = QuizService(
        db_session, author=StubQuizAuthor(), grader=grader or _ScriptedGrader(None)
    )
    quiz = await service.create(
        user_id=user.user_id,
        conversation=conversation,
        concept=concept,
        paper_id=paper.paper_id,
        passages=[(chunk.chunk_id, chunk.content)],
    )
    return service, quiz


# --------------------------------------------------------------------------
# The gate — the backend decides whether a check is allowed
# --------------------------------------------------------------------------


async def test_a_check_is_allowed_by_default(db_session: AsyncSession):
    user, _, _, concept, conversation = await _setup(db_session)

    decision = await QuizService(
        db_session, author=StubQuizAuthor(), grader=_ScriptedGrader(None)
    ).gate(user=user, conversation=conversation, concept=concept)

    assert decision.allowed


async def test_quiz_appetite_off_refuses(db_session: AsyncSession):
    user, _, _, concept, conversation = await _setup(db_session, quiz_appetite="off")

    decision = await QuizService(
        db_session, author=StubQuizAuthor(), grader=_ScriptedGrader(None)
    ).gate(user=user, conversation=conversation, concept=concept)

    assert not decision.allowed
    assert decision.reason == REFUSED_APPETITE_OFF


async def test_a_pending_quiz_refuses_another(db_session: AsyncSession):
    """Two open questions at once would make the routing ambiguous."""
    user, paper, chunk, concept, conversation = await _setup(db_session)
    service, _ = await _ask(db_session, user, paper, chunk, concept, conversation)

    decision = await service.gate(
        user=user, conversation=conversation, concept=concept
    )

    assert decision.reason == REFUSED_ALREADY_PENDING


async def test_a_well_understood_concept_is_not_worth_checking(
    db_session: AsyncSession,
):
    user, _, _, concept, conversation = await _setup(db_session)
    concept.understanding_score = 0.95
    concept.score_confidence = 0.9
    concept.last_reinforced_at = datetime.now(UTC)
    await db_session.flush()

    decision = await QuizService(
        db_session, author=StubQuizAuthor(), grader=_ScriptedGrader(None)
    ).gate(user=user, conversation=conversation, concept=concept)

    assert decision.reason == REFUSED_WELL_UNDERSTOOD


async def test_a_recent_check_rate_limits_the_next(db_session: AsyncSession):
    user, paper, chunk, concept, conversation = await _setup(db_session)
    service, _ = await _ask(db_session, user, paper, chunk, concept, conversation)

    # Clear the pending state so the rate limit is what refuses, not the state.
    conversation.activity = "FREE"
    conversation.pending_quiz_id = None
    await db_session.flush()

    decision = await service.gate(
        user=user, conversation=conversation, concept=concept
    )

    assert decision.reason == REFUSED_RATE_LIMITED


# --------------------------------------------------------------------------
# Creation
# --------------------------------------------------------------------------


async def test_asking_sets_the_state_and_its_payload_together(
    db_session: AsyncSession,
):
    """The CHECK constraint `quiz_pending_consistency` forbids disagreement."""
    user, paper, chunk, concept, conversation = await _setup(db_session)

    _, quiz = await _ask(db_session, user, paper, chunk, concept, conversation)

    assert conversation.activity == "QUIZ_PENDING"
    assert conversation.pending_quiz_id == quiz.quiz_id
    assert conversation.active_concept_id == concept.concept_id


async def test_a_quiz_is_grounded_in_this_turns_passages(db_session: AsyncSession):
    """A quiz not anchored in the paper tests world knowledge, not reading."""
    user, paper, chunk, concept, conversation = await _setup(db_session)

    _, quiz = await _ask(db_session, user, paper, chunk, concept, conversation)

    assert quiz.grounding_chunk_ids == [chunk.chunk_id]


async def test_a_quiz_without_passages_is_refused(db_session: AsyncSession):
    user, paper, _, concept, conversation = await _setup(db_session)
    service = QuizService(
        db_session, author=StubQuizAuthor(), grader=_ScriptedGrader(None)
    )

    with pytest.raises(QuizUnavailable):
        await service.create(
            user_id=user.user_id,
            conversation=conversation,
            concept=concept,
            paper_id=paper.paper_id,
            passages=[],
        )

    assert conversation.activity == "FREE"


async def test_the_rubric_is_stored_but_never_in_the_question(
    db_session: AsyncSession,
):
    """If the agent held the rubric it could leak the expected answer."""
    user, paper, chunk, concept, conversation = await _setup(db_session)

    _, quiz = await _ask(db_session, user, paper, chunk, concept, conversation)

    assert quiz.rubric["must_mention"]
    for point in quiz.rubric["must_mention"]:
        assert point not in quiz.question or point == concept.canonical_name


# --------------------------------------------------------------------------
# Grading — we never guess
# --------------------------------------------------------------------------


async def test_a_correct_answer_moves_forward(db_session: AsyncSession):
    user, paper, chunk, concept, conversation = await _setup(db_session)
    grader = _ScriptedGrader(Grading(grade="correct", missing_elements=[], confidence=0.9))
    service, _ = await _ask(
        db_session, user, paper, chunk, concept, conversation, grader
    )

    result = await service.grade_pending(
        conversation=conversation, user_id=user.user_id, answer_text="a good answer"
    )

    assert result.grade == "correct"
    assert result.next_action == NEXT_MOVE_FORWARD
    assert result.signal_type == "quiz_correct"
    assert conversation.activity == "FREE"
    assert conversation.pending_quiz_id is None


async def test_a_partial_answer_asks_for_a_different_explanation(
    db_session: AsyncSession,
):
    user, paper, chunk, concept, conversation = await _setup(db_session)
    grader = _ScriptedGrader(
        Grading(grade="partial", missing_elements=["the KL term"], confidence=0.7)
    )
    service, _ = await _ask(
        db_session, user, paper, chunk, concept, conversation, grader
    )

    result = await service.grade_pending(
        conversation=conversation, user_id=user.user_id, answer_text="half an answer"
    )

    assert result.next_action == NEXT_EXPLAIN_DIFFERENTLY
    assert result.missing_elements == ["the KL term"]
    assert conversation.activity == "EXPLAINING"


async def test_a_weak_prerequisite_redirects_the_next_action(
    db_session: AsyncSession,
):
    """The graph identifies the blocking concept, not the model.

    ARCHITECTURE 11 calls this a purely structural insight, and it is the one
    branch no amount of reading the answer could produce.
    """
    user, paper, chunk, concept, conversation = await _setup(db_session)

    blocker = Concept(
        user_id=user.user_id,
        canonical_name="KL divergence",
        normalized_name=normalize_name("KL divergence"),
        understanding_score=0.2,
        score_confidence=0.8,
        last_reinforced_at=datetime.now(UTC),
    )
    db_session.add(blocker)
    await db_session.flush()
    db_session.add(
        ConceptRelationship(
            user_id=user.user_id,
            source_concept_id=blocker.concept_id,
            target_concept_id=concept.concept_id,
            relationship_type="prerequisite_of",
            confidence=0.8,
            discovery_method="model",
        )
    )
    await db_session.flush()

    grader = _ScriptedGrader(
        Grading(grade="incorrect", missing_elements=["everything"], confidence=0.8)
    )
    service, _ = await _ask(
        db_session, user, paper, chunk, concept, conversation, grader
    )

    result = await service.grade_pending(
        conversation=conversation, user_id=user.user_id, answer_text="no idea"
    )

    assert result.next_action == NEXT_REVISIT_PREREQUISITE
    assert result.prerequisite_name == "KL divergence"


async def test_a_strong_prerequisite_does_not_redirect(db_session: AsyncSession):
    """Only a *weak* prerequisite is a blocker worth going back to."""
    user, paper, chunk, concept, conversation = await _setup(db_session)

    fine = Concept(
        user_id=user.user_id,
        canonical_name="KL divergence",
        normalized_name=normalize_name("KL divergence"),
        understanding_score=0.9,
        score_confidence=0.9,
        last_reinforced_at=datetime.now(UTC),
    )
    db_session.add(fine)
    await db_session.flush()
    db_session.add(
        ConceptRelationship(
            user_id=user.user_id,
            source_concept_id=fine.concept_id,
            target_concept_id=concept.concept_id,
            relationship_type="prerequisite_of",
            confidence=0.8,
            discovery_method="model",
        )
    )
    await db_session.flush()

    grader = _ScriptedGrader(
        Grading(grade="incorrect", missing_elements=[], confidence=0.8)
    )
    service, _ = await _ask(
        db_session, user, paper, chunk, concept, conversation, grader
    )

    result = await service.grade_pending(
        conversation=conversation, user_id=user.user_id, answer_text="no idea"
    )

    assert result.next_action == NEXT_EXPLAIN_DIFFERENTLY


async def test_the_grader_gets_exactly_one_retry(db_session: AsyncSession):
    user, paper, chunk, concept, conversation = await _setup(db_session)
    grader = _ScriptedGrader(
        Grading(grade="correct", missing_elements=[], confidence=0.9), fail_times=1
    )
    service, _ = await _ask(
        db_session, user, paper, chunk, concept, conversation, grader
    )

    result = await service.grade_pending(
        conversation=conversation, user_id=user.user_id, answer_text="an answer"
    )

    assert grader.calls == 2
    assert result.grade == "correct"


async def test_a_grade_is_never_guessed(db_session: AsyncSession):
    """`grade = NULL` with an error, which the CHECK constraint pairs."""
    user, paper, chunk, concept, conversation = await _setup(db_session)
    grader = _ScriptedGrader(RuntimeError("model down"))
    service, quiz = await _ask(
        db_session, user, paper, chunk, concept, conversation, grader
    )

    result = await service.grade_pending(
        conversation=conversation, user_id=user.user_id, answer_text="an answer"
    )

    assert result.grade is None
    assert result.grading_error is not None
    assert result.signal_type is None, "a failed grading says nothing about the reader"

    attempt = await db_session.scalar(
        select(QuizAttempt).where(QuizAttempt.quiz_id == quiz.quiz_id)
    )
    assert attempt.grade is None
    assert attempt.grading_error is not None


async def test_a_failed_grading_still_releases_the_reader(db_session: AsyncSession):
    """Nobody gets stranded in QUIZ_PENDING because the grader broke."""
    user, paper, chunk, concept, conversation = await _setup(db_session)
    grader = _ScriptedGrader(RuntimeError("model down"))
    service, _ = await _ask(
        db_session, user, paper, chunk, concept, conversation, grader
    )

    await service.grade_pending(
        conversation=conversation, user_id=user.user_id, answer_text="an answer"
    )

    assert conversation.activity != "QUIZ_PENDING"
    assert conversation.pending_quiz_id is None


async def test_an_unknown_grade_is_treated_as_a_failure(db_session: AsyncSession):
    """A grade outside the closed set is not a grade."""
    user, paper, chunk, concept, conversation = await _setup(db_session)
    grader = _ScriptedGrader(
        Grading(grade="brilliant", missing_elements=[], confidence=1.0)
    )
    service, _ = await _ask(
        db_session, user, paper, chunk, concept, conversation, grader
    )

    result = await service.grade_pending(
        conversation=conversation, user_id=user.user_id, answer_text="an answer"
    )

    assert result.grade is None


async def test_attempts_are_numbered_per_quiz(db_session: AsyncSession):
    user, paper, chunk, concept, conversation = await _setup(db_session)
    grader = _ScriptedGrader(
        Grading(grade="partial", missing_elements=[], confidence=0.5)
    )
    service, quiz = await _ask(
        db_session, user, paper, chunk, concept, conversation, grader
    )

    await service.grade_pending(
        conversation=conversation, user_id=user.user_id, answer_text="first"
    )
    conversation.activity = "QUIZ_PENDING"
    conversation.pending_quiz_id = quiz.quiz_id
    await db_session.flush()
    await service.grade_pending(
        conversation=conversation, user_id=user.user_id, answer_text="second"
    )

    numbers = (
        await db_session.scalars(
            select(QuizAttempt.attempt_no)
            .where(QuizAttempt.quiz_id == quiz.quiz_id)
            .order_by(QuizAttempt.attempt_no)
        )
    ).all()
    assert list(numbers) == [1, 2]


async def test_grading_is_scoped_to_this_users_quizzes(db_session: AsyncSession):
    user, paper, chunk, concept, conversation = await _setup(db_session)
    await _ask(db_session, user, paper, chunk, concept, conversation)

    stored = await db_session.scalar(
        select(func.count()).select_from(Quiz).where(Quiz.user_id == user.user_id)
    )
    assert stored == 1
