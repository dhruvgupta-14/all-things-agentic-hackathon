"""Transient model errors are waited out; real faults still fail closed.

Vertex returns 429 under shared-capacity pressure, and it has hit mid-demo.
Retrying is the easy part; the risk is that a second attempt re-runs the tools
and leaves the turn holding both attempts' retrieved passages, which would
renumber every citation marker against a set the reader never saw.
"""

import uuid

import pytest

from app.agent.runner import (
    MAX_TRANSIENT_ATTEMPTS,
    AgentUnavailable,
    _is_transient,
    _reset_for_retry,
)
from app.agent.tools import TurnToolContext
from app.services.retrieval import RetrievedChunk


def _chunk(marker: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        paper_id=uuid.uuid4(),
        content=f"passage {marker}",
        similarity=0.8,
        rank=marker,
        page_start=1,
        page_end=1,
        section_path="1",
        section_heading=None,
        section_role="method",
    )


def _context(**kwargs) -> TurnToolContext:
    return TurnToolContext(
        session=None, user_id=uuid.uuid4(), paper_scope=[], retrieval=None, **kwargs
    )


# --------------------------------------------------------------------------
# What counts as worth retrying
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "429 RESOURCE_EXHAUSTED. Resource exhausted. Please try again later.",
        "503 UNAVAILABLE",
        "504 DEADLINE_EXCEEDED",
    ],
)
def test_capacity_errors_are_transient(message: str):
    assert _is_transient(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    [
        "400 INVALID_ARGUMENT: the prompt was malformed",
        "403 PERMISSION_DENIED",
        "404 NOT_FOUND: publisher model not found",
        "no attribute 'canonical_name'",
    ],
)
def test_real_faults_are_not_retried(message: str):
    """A 404 on the model name is a configuration error. Retrying it three
    times just makes the same wrong deployment slower."""
    assert not _is_transient(RuntimeError(message))


def test_the_retry_budget_is_bounded():
    """Unbounded retries turn a capacity problem into a hung request."""
    assert 2 <= MAX_TRANSIENT_ATTEMPTS <= 5


# --------------------------------------------------------------------------
# What a retry must clean up
# --------------------------------------------------------------------------


async def test_a_retry_starts_from_an_empty_retrieval_set():
    """Otherwise markers renumber against passages from both attempts."""
    context = _context()
    context.retrieved.extend([_chunk(1), _chunk(2)])
    context.queries.append("first attempt")
    context.tools_called.append("retrieve_paper_context")

    await _reset_for_retry(context)

    assert context.retrieved == []
    assert context.queries == []
    assert context.tools_called == []


async def test_a_retry_does_not_forget_the_prefetched_memory():
    """The prefetch ran before the agent did. Clearing it would make the turn
    report `memory_read = False` for a read that genuinely happened."""
    from app.services.memory import MemoryRecord

    context = _context()
    context.remember(
        [
            MemoryRecord(
                concept_id=uuid.uuid4(),
                canonical_name="ELBO",
                understanding_score=0.3,
                score_confidence=0.8,
                effective_style="numerical",
                last_reinforced_at=None,
                evidence_count=2,
            )
        ]
    )

    await _reset_for_retry(context)

    assert context.memory_read is True
    assert len(context.memory_seen) == 1


async def test_a_retry_drops_signals_the_failed_attempt_buffered():
    """They describe an exchange the reader never received."""
    context = _context()
    context.pending_signals.append(object())

    await _reset_for_retry(context)

    assert context.pending_signals == []


async def test_a_quiz_from_a_failed_attempt_does_not_strand_the_reader(
    db_session,
):
    """The reader never saw the question, so their next message must not be
    graded against it."""
    from app.db.models import Session, User

    user = User(auth_subject=f"retry-{uuid.uuid4()}")
    db_session.add(user)
    await db_session.flush()
    conversation = Session(user_id=user.user_id, activity="FREE")
    db_session.add(conversation)
    await db_session.flush()

    # As `generate_quiz` would have left it.
    quiz_id = uuid.uuid4()
    conversation.activity = "QUIZ_PENDING"
    context = _context(conversation=conversation)
    context.quiz_asked = quiz_id

    await _reset_for_retry(context)

    assert conversation.activity == "FREE"
    assert conversation.pending_quiz_id is None
    assert context.quiz_asked is None


# --------------------------------------------------------------------------
# The pipeline still fails closed
# --------------------------------------------------------------------------


async def test_a_non_transient_failure_still_fails_the_turn(monkeypatch):
    """Fail-closed is the guarantee; the retry must not weaken it."""
    from app.agent import runner

    async def always_broken(*args, **kwargs):
        raise RuntimeError("400 INVALID_ARGUMENT")

    monkeypatch.setattr(runner, "_configure_transport", lambda: None)
    monkeypatch.setattr(runner, "build_agent", lambda *a, **k: object())

    class _Runner:
        def __init__(self, **kwargs):
            pass

        def run_async(self, **kwargs):
            raise RuntimeError("400 INVALID_ARGUMENT")

    monkeypatch.setattr(runner, "Runner", _Runner)

    with pytest.raises(AgentUnavailable):
        await runner.run_turn(
            context=_context(),
            history=[],
            user_message="hello",
            paper_title=None,
            session_key=str(uuid.uuid4()),
        )
