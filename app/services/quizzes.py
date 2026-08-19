"""Adaptive learning checks (ARCHITECTURE 11, 9.2 step 3, 14.2).

The split here is the one the architecture is most insistent about:

    the backend decides   whether a check is *allowed* — frequency, appetite,
                          which passages ground it, and every state transition
    the agent decides     whether a check is *useful* right now, and on what

And the routing is deterministic. When `sessions.activity = QUIZ_PENDING`, the
next message is graded — full stop. Asking a model "is this a quiz answer?"
immediately after asking the question is a wasted call and a source of
nondeterminism at the most measured moment in the system.

Two rules that are not negotiable:

**The rubric never reaches the agent.** `generate_quiz` returns `{quiz_id,
question}`. If the agent held the rubric it could leak the expected answer into
the question it asks, and the check would measure nothing.

**We never guess a grade.** The grader is one constrained call returning a
fixed schema. It gets one retry. After that the attempt is stored with
`grade = NULL` and a `grading_error`, and the reader is told plainly — the
CHECK constraint `grade_or_error` makes any other outcome unrepresentable.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import (
    Concept,
    ConceptRelationship,
    Quiz,
    QuizAttempt,
    Session,
    User,
)
from app.services.learner_state import decay_factor, is_callback_candidate

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)

QUIZ_DIFFICULTY = ("easy", "medium", "hard")

# Turns between checks. A companion that quizzes every other turn is a test,
# not a reading partner. Suppression is recorded the same way callbacks are.
QUIZ_MIN_TURN_GAP = 4

# `users.preferences.quiz_appetite` (ARCHITECTURE 4.1).
QUIZ_APPETITE_GAP_MULTIPLIER: dict[str, float | None] = {
    "off": None,
    "low": 2.0,
    "normal": 1.0,
    "high": 0.5,
}
DEFAULT_QUIZ_APPETITE = "normal"

# A concept already well understood is not worth checking.
QUIZ_SKIP_ABOVE_SCORE = 0.85

REFUSED_APPETITE_OFF = "quiz_appetite_off"
REFUSED_RATE_LIMITED = "quiz_rate_limited"
REFUSED_ALREADY_PENDING = "quiz_already_pending"
REFUSED_NO_GROUNDING = "quiz_no_grounding"
REFUSED_WELL_UNDERSTOOD = "quiz_not_needed"

MAX_ANSWER_CHARS = 4000
MAX_GRADING_ERROR_CHARS = 200


class QuizUnavailable(Exception):
    """The model could not author a question. Nothing is written."""


class GradingFailed(Exception):
    """The grader could not produce a valid grade. Recorded, never guessed."""


# --------------------------------------------------------------------------
# Model-facing schemas — constrained, single-shot, never an agent loop
# --------------------------------------------------------------------------


class AuthoredQuiz(BaseModel):
    question: str = Field(max_length=2000)
    # The rubric is a list of things a correct answer must contain. Kept
    # structured rather than prose so grading has something to check against.
    must_mention: list[str] = Field(default_factory=list, max_length=6)


class Grading(BaseModel):
    grade: str
    missing_elements: list[str] = Field(default_factory=list, max_length=6)
    confidence: float = Field(ge=0.0, le=1.0)


_AUTHOR_PROMPT = """\
You are writing one short comprehension check for someone reading a research
paper. It must be answerable *from the passages below and nothing else*.

Concept: {concept}
Difficulty: {difficulty}

Passages:
{passages}

Write one question that checks whether the reader understood the concept as
this paper presents it. Not a trivia question about wording — a question about
the idea. Then list the 2-4 things a correct answer must mention.

Do not include the answer in the question.
"""

_GRADER_PROMPT = """\
Grade this answer against the rubric. Be fair but not generous: a reader is
better served by an honest "partial" than a flattering "correct".

Question: {question}

A correct answer must mention:
{rubric}

The reader answered:
{answer}

Return grade as exactly one of: correct, partial, incorrect.
List the rubric points the answer did not cover, using the rubric's own wording.
Give your confidence between 0 and 1.
"""


class QuizAuthor(Protocol):
    @property
    def model_name(self) -> str: ...

    def write(
        self, *, concept: str, difficulty: str, passages: list[str]
    ) -> AuthoredQuiz: ...


class Grader(Protocol):
    @property
    def model_name(self) -> str: ...

    def grade(self, *, question: str, rubric: list[str], answer: str) -> Grading: ...


class _GeminiCall:
    """Shared transport for the two constrained calls."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._project = project
        self._location = location
        self._client = None

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self):
        # Shared per process: building one costs a ~12s credential and TLS
        # handshake on its first request (see app/services/genai_client.py).
        if self._client is None:
            from app.services.genai_client import get_genai_client

            self._client = get_genai_client(
                api_key=self._api_key,
                project=self._project,
                location=self._location,
            )
        return self._client

    def _structured(self, prompt: str, schema, *, temperature: float):
        from google.genai import types

        response = self._get_client().models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=temperature,
            ),
        )
        if response.parsed is None:
            raise ValueError("model returned no parseable response")
        return schema.model_validate(response.parsed, from_attributes=True)


class GeminiQuizAuthor(_GeminiCall):
    def write(
        self, *, concept: str, difficulty: str, passages: list[str]
    ) -> AuthoredQuiz:
        prompt = _AUTHOR_PROMPT.format(
            concept=concept,
            difficulty=difficulty,
            passages="\n\n".join(f"- {text}" for text in passages),
        )
        try:
            # Some variety in question wording is fine and makes repeated
            # checks on one concept less mechanical.
            return self._structured(prompt, AuthoredQuiz, temperature=0.4)
        except Exception as exc:
            raise QuizUnavailable(f"{type(exc).__name__}: {exc}") from exc


class GeminiGrader(_GeminiCall):
    def grade(self, *, question: str, rubric: list[str], answer: str) -> Grading:
        prompt = _GRADER_PROMPT.format(
            question=question,
            rubric="\n".join(f"- {point}" for point in rubric),
            answer=answer,
        )
        try:
            # Zero temperature: the same answer must grade the same way twice.
            # This is the most measured moment in the system.
            return self._structured(prompt, Grading, temperature=0.0)
        except Exception as exc:
            raise GradingFailed(f"{type(exc).__name__}: {exc}") from exc


class StubQuizAuthor:
    """Deterministic, offline. Keeps the suite from spending quota."""

    model_name = "stub-quiz-author"

    def write(
        self, *, concept: str, difficulty: str, passages: list[str]
    ) -> AuthoredQuiz:
        return AuthoredQuiz(
            question=f"In your own words, what is {concept} and why does it matter here?",
            must_mention=[concept, "why it matters"],
        )


class StubGrader:
    """Lexical overlap, not judgement. Enough to exercise every branch."""

    model_name = "stub-grader"

    def grade(self, *, question: str, rubric: list[str], answer: str) -> Grading:
        lowered = (answer or "").lower()
        missing = [point for point in rubric if point.lower() not in lowered]
        if not rubric:
            grade = "partial"
        elif not missing:
            grade = "correct"
        elif len(missing) == len(rubric):
            grade = "incorrect"
        else:
            grade = "partial"
        return Grading(grade=grade, missing_elements=missing, confidence=0.6)


def get_quiz_author() -> QuizAuthor:
    settings = get_settings()
    if settings.vertex_project:
        return GeminiQuizAuthor(
            settings.gemini_model,
            project=settings.vertex_project,
            location=settings.vertex_location,
        )
    if settings.gemini_api_key:
        return GeminiQuizAuthor(settings.gemini_model, api_key=settings.gemini_api_key)
    return StubQuizAuthor()


def get_grader() -> Grader:
    settings = get_settings()
    if settings.vertex_project:
        return GeminiGrader(
            settings.gemini_model,
            project=settings.vertex_project,
            location=settings.vertex_location,
        )
    if settings.gemini_api_key:
        return GeminiGrader(settings.gemini_model, api_key=settings.gemini_api_key)
    return StubGrader()


# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------

NEXT_MOVE_FORWARD = "move_forward"
NEXT_EXPLAIN_DIFFERENTLY = "explain_differently"
NEXT_REVISIT_PREREQUISITE = "revisit_prerequisite"


@dataclass(slots=True)
class GateDecision:
    allowed: bool
    reason: str | None = None


@dataclass(slots=True)
class GradedAttempt:
    attempt_id: uuid.UUID
    concept_id: uuid.UUID
    concept_name: str
    grade: str | None
    missing_elements: list[str]
    grading_error: str | None
    next_action: str
    prerequisite_name: str | None = None
    signal_type: str | None = None


_SIGNAL_FOR_GRADE = {
    "correct": "quiz_correct",
    "partial": "quiz_partial",
    "incorrect": "quiz_incorrect",
}


class QuizService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        author: QuizAuthor | None = None,
        grader: Grader | None = None,
    ) -> None:
        self._session = session
        self._author = author or get_quiz_author()
        self._grader = grader or get_grader()

    # -- the gate ----------------------------------------------------------

    async def gate(
        self, *, user: User, conversation: Session, concept: Concept | None
    ) -> GateDecision:
        """Step 1 — is a check *allowed*? Whether it is useful is the agent's call."""
        if conversation.activity == "QUIZ_PENDING":
            return GateDecision(False, REFUSED_ALREADY_PENDING)

        preferences = user.preferences or {}
        appetite = preferences.get("quiz_appetite", DEFAULT_QUIZ_APPETITE)
        multiplier = QUIZ_APPETITE_GAP_MULTIPLIER.get(appetite, 1.0)
        if multiplier is None:
            return GateDecision(False, REFUSED_APPETITE_OFF)

        if concept is not None:
            score = concept.user_override_score
            if score is None and concept.understanding_score is not None:
                score = concept.understanding_score * decay_factor(
                    concept.last_reinforced_at, _now()
                )
            # Well understood and we are sure of it: nothing to find out.
            if (
                score is not None
                and score >= QUIZ_SKIP_ABOVE_SCORE
                and (concept.score_confidence or 0) >= 0.3
            ):
                return GateDecision(False, REFUSED_WELL_UNDERSTOOD)

        minimum_gap = max(1, round(QUIZ_MIN_TURN_GAP * multiplier))
        if not await self._gap_satisfied(conversation, minimum_gap):
            return GateDecision(False, REFUSED_RATE_LIMITED)

        return GateDecision(True)

    async def _gap_satisfied(self, conversation: Session, minimum_gap: int) -> bool:
        last_quiz_turn = await self._session.scalar(
            select(func.max(Quiz.created_at)).where(
                Quiz.user_id == conversation.user_id
            )
        )
        if last_quiz_turn is None:
            return True

        from app.db.models import Turn

        turns_since = await self._session.scalar(
            select(func.count())
            .select_from(Turn)
            .where(
                Turn.user_id == conversation.user_id,
                Turn.created_at > last_quiz_turn,
            )
        )
        return (turns_since or 0) >= minimum_gap

    # -- creation ----------------------------------------------------------

    async def create(
        self,
        *,
        user_id: uuid.UUID,
        conversation: Session,
        concept: Concept,
        paper_id: uuid.UUID,
        passages: list[tuple[uuid.UUID, str]],
        difficulty: str = "medium",
    ) -> Quiz:
        """Step 2-3 — author, ground, persist, and set the state.

        `passages` are this turn's retrieval set. Grounding is mandatory: a
        quiz not anchored in the paper tests world knowledge rather than
        reading, and the CHECK constraint refuses a quiz without chunks.
        """
        if not passages:
            raise QuizUnavailable(REFUSED_NO_GROUNDING)

        if difficulty not in QUIZ_DIFFICULTY:
            difficulty = "medium"

        authored = self._author.write(
            concept=concept.canonical_name,
            difficulty=difficulty,
            passages=[text for _, text in passages],
        )

        quiz = Quiz(
            user_id=user_id,
            concept_id=concept.concept_id,
            paper_id=paper_id,
            question=authored.question,
            # The rubric lives here and is never returned to the agent.
            rubric={
                "must_mention": authored.must_mention,
                "difficulty": difficulty,
                "authored_by": self._author.model_name,
            },
            grounding_chunk_ids=[chunk_id for chunk_id, _ in passages],
        )
        self._session.add(quiz)
        await self._session.flush()

        # The CHECK constraint `quiz_pending_consistency` keeps these two in
        # step, so they are set together or not at all.
        conversation.activity = "QUIZ_PENDING"
        conversation.pending_quiz_id = quiz.quiz_id
        conversation.active_concept_id = concept.concept_id
        await self._session.flush()

        logger.info(
            "quiz created", extra={"quiz_id": str(quiz.quiz_id), "concept": concept.canonical_name}
        )
        return quiz

    # -- grading -----------------------------------------------------------

    async def grade_pending(
        self,
        *,
        conversation: Session,
        user_id: uuid.UUID,
        answer_text: str,
    ) -> GradedAttempt:
        """Steps 5-10 — the deterministic route's destination.

        One constrained call, one retry, then `grade = NULL` with a recorded
        error. The state transition happens either way: a reader must never be
        stranded in QUIZ_PENDING because grading failed.
        """
        quiz = await self._session.get(Quiz, conversation.pending_quiz_id)
        if quiz is None:
            # State and payload disagreed, which the CHECK should prevent.
            # Recover rather than strand the reader.
            conversation.activity = "FREE"
            conversation.pending_quiz_id = None
            raise GradingFailed("the pending quiz no longer exists")

        concept = await self._session.get(Concept, quiz.concept_id)
        rubric = list((quiz.rubric or {}).get("must_mention") or [])
        answer = (answer_text or "").strip()[:MAX_ANSWER_CHARS]

        grading: Grading | None = None
        error: str | None = None
        for attempt in range(2):  # one call, one retry
            try:
                candidate = self._grader.grade(
                    question=quiz.question, rubric=rubric, answer=answer
                )
                if candidate.grade not in _SIGNAL_FOR_GRADE:
                    raise GradingFailed(f"unknown grade {candidate.grade!r}")
                grading = candidate
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:MAX_GRADING_ERROR_CHARS]
                logger.warning("grading attempt %s failed: %s", attempt + 1, error)

        attempt_no = (
            await self._session.scalar(
                select(func.count())
                .select_from(QuizAttempt)
                .where(QuizAttempt.quiz_id == quiz.quiz_id)
            )
        ) or 0

        record = QuizAttempt(
            quiz_id=quiz.quiz_id,
            user_id=user_id,
            answer_text=answer or "(no answer)",
            grade=grading.grade if grading else None,
            missing_elements=list(grading.missing_elements) if grading else None,
            grader_confidence=grading.confidence if grading else None,
            # The CHECK constraint requires exactly one of grade/grading_error.
            grading_error=None if grading else (error or "grading failed"),
            attempt_no=attempt_no + 1,
        )
        self._session.add(record)
        await self._session.flush()

        next_action = NEXT_MOVE_FORWARD
        prerequisite_name: str | None = None
        if grading is None:
            # Told plainly, and the reader is released from the pending state.
            next_action = NEXT_MOVE_FORWARD
        elif grading.grade == "correct":
            next_action = NEXT_MOVE_FORWARD
        else:
            weak_prerequisite = await self._weak_prerequisite(user_id, quiz.concept_id)
            if grading.grade == "incorrect" and weak_prerequisite is not None:
                next_action = NEXT_REVISIT_PREREQUISITE
                prerequisite_name = weak_prerequisite.canonical_name
            else:
                next_action = NEXT_EXPLAIN_DIFFERENTLY

        # Step 10 — the transition. Always taken.
        conversation.pending_quiz_id = None
        conversation.activity = (
            "FREE" if next_action == NEXT_MOVE_FORWARD else "EXPLAINING"
        )
        await self._session.flush()

        return GradedAttempt(
            attempt_id=record.attempt_id,
            concept_id=quiz.concept_id,
            concept_name=concept.canonical_name if concept else "this concept",
            grade=grading.grade if grading else None,
            missing_elements=list(grading.missing_elements) if grading else [],
            grading_error=record.grading_error,
            next_action=next_action,
            prerequisite_name=prerequisite_name,
            signal_type=_SIGNAL_FOR_GRADE.get(grading.grade) if grading else None,
        )

    async def _weak_prerequisite(
        self, user_id: uuid.UUID, concept_id: uuid.UUID
    ) -> Concept | None:
        """The blocking concept, identified by the graph rather than the model.

        ARCHITECTURE 11 calls this a purely structural insight, and it is the
        one branch of the three-way next action that no amount of reading the
        answer could produce.
        """
        rows = (
            await self._session.execute(
                select(Concept)
                .join(
                    ConceptRelationship,
                    ConceptRelationship.source_concept_id == Concept.concept_id,
                )
                .where(
                    ConceptRelationship.user_id == user_id,
                    ConceptRelationship.target_concept_id == concept_id,
                    ConceptRelationship.relationship_type == "prerequisite_of",
                    Concept.merged_into_id.is_(None),
                )
            )
        ).scalars().all()

        now = _now()
        for concept in rows:
            score = concept.user_override_score
            if score is None and concept.understanding_score is not None:
                score = concept.understanding_score * decay_factor(
                    concept.last_reinforced_at, now
                )
            if is_callback_candidate(score, concept.score_confidence):
                return concept
        return None
