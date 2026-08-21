"""Asking about a paper that is not ready yet (ARCHITECTURE 8.4).

This was reported as the agent lying about a paper. A reader uploaded a PDF,
ingestion never ran, and the first question came back as:

    "The open paper does not contain any content about the impact of mergers
     and acquisitions on the pharmaceutical industry."

Which is a statement about the paper's *contents*, made about a paper whose
contents had never been read. It is not a hallucination in the usual sense —
retrieval genuinely returned nothing, and the agent reported that honestly —
but the reader cannot tell "I searched and it is not in there" from "there is
nothing to search yet", and the two call for opposite responses.

The fix is deterministic and sits in front of the agent: a paper that is not
`ready` or `partially_ready` is answered from its status, with no retrieval and
no model call. Same class of decision as the quiz route — anything that must be
*correct* rather than plausible does not get asked of a model.
"""

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Message, Paper, Session, Turn, User, UserPaperAccess
from app.schemas.sse import decode
from tests.conftest import build_pdf
from tests.fakes import InMemoryStorage


@pytest.fixture
def storage(storage_backend) -> InMemoryStorage:
    return storage_backend


@pytest.fixture
def agent_must_not_run(monkeypatch):
    """Fail loudly if the agent is reached at all.

    The saving is not incidental: running it costs a model call plus several
    tool calls to arrive at a worse answer.
    """

    async def refuse(*args, **kwargs):
        raise AssertionError("the agent ran against a paper that is not ready")

    monkeypatch.setattr("app.services.turns.run_turn", refuse)


async def _reader_with_paper(
    db_session: AsyncSession,
    storage: InMemoryStorage,
    subject: str,
    *,
    status: str,
    phase: str | None = None,
    error_code: str | None = None,
) -> tuple[User, Paper, Session]:
    """A reader whose open paper is in `status`. Deliberately never ingested."""
    user = User(auth_subject=subject)
    db_session.add(user)
    await db_session.flush()

    content_hash = uuid.uuid4().hex + uuid.uuid4().hex[:32]
    paper = Paper(
        content_hash=content_hash,
        storage_uri=storage.put(build_pdf(["Some text."]), content_hash=content_hash),
        original_filename="mergers-and-acquisitions.pdf",
        processing_status=status,
        processing_phase=phase,
        error_code=error_code,
    )
    db_session.add(paper)
    await db_session.flush()

    db_session.add(UserPaperAccess(user_id=user.user_id, paper_id=paper.paper_id))
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


def _text(events: list[dict]) -> str:
    return "".join(e["text"] for e in events if e["event"] == "token")


# --------------------------------------------------------------------------
# What the reader is told
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["queued", "processing"])
async def test_an_unprocessed_paper_says_so_instead_of_reporting_an_absence(
    db_session: AsyncSession, storage, agent_must_not_run, status: str
):
    user, _, conversation = await _reader_with_paper(
        db_session, storage, f"unready-{uuid.uuid4()}", status=status
    )

    events = await _run(
        db_session,
        conversation,
        user,
        "What is the impact of mergers and acquisitions on the pharma industry?",
    )
    said = _text(events).lower()

    assert "still being processed" in said
    # The failure that started this: a claim about what the paper contains.
    assert "does not contain" not in said
    assert "no content" not in said


async def test_the_current_phase_is_named_when_there_is_one(
    db_session: AsyncSession, storage, agent_must_not_run
):
    """"Processing" alone reads like a hang on a job that takes a minute."""
    user, _, conversation = await _reader_with_paper(
        db_session,
        storage,
        f"unready-{uuid.uuid4()}",
        status="processing",
        phase="embed",
    )

    events = await _run(db_session, conversation, user, "Explain the paper.")

    assert "embed" in _text(events)


async def test_a_failed_paper_gives_the_reason_and_a_way_forward(
    db_session: AsyncSession, storage, agent_must_not_run
):
    user, _, conversation = await _reader_with_paper(
        db_session,
        storage,
        f"unready-{uuid.uuid4()}",
        status="failed",
        error_code="pdf_encrypted",
    )

    events = await _run(db_session, conversation, user, "Explain the paper.")
    said = _text(events)

    assert "password protected" in said
    assert "unprotected copy" in said
    # The typed code is for the logs. It means nothing to a reader.
    assert "pdf_encrypted" not in said


async def test_an_unrecognised_error_code_does_not_leak_into_the_conversation(
    db_session: AsyncSession, storage, agent_must_not_run
):
    user, _, conversation = await _reader_with_paper(
        db_session,
        storage,
        f"unready-{uuid.uuid4()}",
        status="failed",
        error_code="some_future_code",
    )

    said = _text(await _run(db_session, conversation, user, "Explain the paper."))

    assert "some_future_code" not in said
    assert "could not be processed" in said


# --------------------------------------------------------------------------
# The turn is still a turn
# --------------------------------------------------------------------------


async def test_the_refusal_emits_the_same_event_contract(
    db_session: AsyncSession, storage, agent_must_not_run
):
    """The SPA renders one shape. A turn that skipped the agent must not skip
    the events the client waits on, or the composer never re-enables."""
    user, _, conversation = await _reader_with_paper(
        db_session, storage, f"unready-{uuid.uuid4()}", status="queued"
    )

    events = await _run(db_session, conversation, user, "Explain the paper.")
    names = [e["event"] for e in events]

    assert names[0] == "state"
    assert "token" in names
    assert names[-1] == "done"
    assert "citations" in names and "memory_used" in names
    # Nothing was retrieved, so nothing may be cited.
    citations = next(e for e in events if e["event"] == "citations")
    assert citations["citations"] == []


async def test_the_refusal_is_persisted_like_any_other_turn(
    db_session: AsyncSession, storage, agent_must_not_run
):
    """A turn that streams but is never written disappears on reload, and the
    reader is left with a question they know they asked and no answer."""
    user, _, conversation = await _reader_with_paper(
        db_session, storage, f"unready-{uuid.uuid4()}", status="queued"
    )

    await _run(db_session, conversation, user, "Explain the paper.")

    turn = await db_session.scalar(
        select(Turn).where(Turn.session_id == conversation.session_id)
    )
    assert turn is not None
    assert turn.agent_action == "paper_queued"
    # No claim about the paper was made, so there is no grounding to report.
    assert turn.grounding_status == "n/a"
    assert turn.tools_called == []

    messages = (
        await db_session.scalars(
            select(Message)
            .where(Message.session_id == conversation.session_id)
            .order_by(Message.ordinal)
        )
    ).all()
    assert [m.role for m in messages] == ["user", "assistant"]


async def test_a_ready_paper_is_untouched_by_this(
    db_session: AsyncSession, storage, monkeypatch
):
    """The guard must be narrow. A `ready` paper still runs the full pipeline."""
    from app.agent.runner import AgentUnavailable

    reached = {}

    async def fake_run_turn(*args, **kwargs):
        # Reaching here is the assertion. Failing the way the pipeline already
        # knows how to handle keeps this test about routing, not about what a
        # crashing agent does.
        reached["yes"] = True
        raise AgentUnavailable("stop here")

    monkeypatch.setattr("app.services.turns.run_turn", fake_run_turn)

    user, _, conversation = await _reader_with_paper(
        db_session, storage, f"ready-{uuid.uuid4()}", status="ready"
    )

    await _run(db_session, conversation, user, "Explain the paper.")

    assert reached.get("yes"), "a ready paper must still reach the agent"
