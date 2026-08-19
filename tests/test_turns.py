"""The turn pipeline and its SSE contract, with the model stubbed.

The agent is replaced by a scripted draft so these run offline and pin the
deterministic half: scope construction, citation verification, persistence,
and the event stream the SPA is built against. What the model actually says is
not the subject here.
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.runner import AgentOutcome
from app.db.models import (
    Message,
    Paper,
    Session,
    Turn,
    TurnRetrieval,
    User,
    UserPaperAccess,
)
from app.ingestion.pipeline import ingest_paper
from app.schemas.sse import decode
from app.services.embeddings import HashingEmbedder
from app.services.storage import LocalStorage
from tests.conftest import build_pdf

PAGES = [
    "Attention Mechanisms\nAbstract\nScaled dot product attention aids translation.\n"
    "1 Introduction\nRecurrent networks process tokens sequentially in order here.",
    "2 Method\nScaled dot product attention weights every token pair directly.\n"
    "3 Results\nTranslation quality improves over the recurrent baseline here.",
]


@pytest.fixture
def storage(storage_dir) -> LocalStorage:
    return LocalStorage(storage_dir)


@pytest.fixture
def scripted_agent(monkeypatch):
    """Replace the model with a fixed draft, and record what it was given."""
    captured: dict = {}

    def script(draft: str, *, call_tool: bool = True):
        async def fake_run_turn(
            *,
            context,
            history,
            user_message,
            paper_title,
            session_key,
            **hints,
        ):
            captured["history"] = history
            captured["user_message"] = user_message
            captured["paper_title"] = paper_title
            captured.update(hints)
            if call_tool:
                tool = _tool_for(context)
                await tool("attention weights tokens")
            return AgentOutcome(
                text=draft,
                tools_called=list(context.tools_called),
                input_tokens=100,
                output_tokens=20,
            )

        monkeypatch.setattr("app.services.turns.run_turn", fake_run_turn)
        return captured

    return script


def _tool_for(context):
    from app.agent.tools import build_retrieve_paper_context

    return build_retrieve_paper_context(context)


async def _reader_with_paper(
    db_session: AsyncSession, storage: LocalStorage, subject: str
) -> tuple[User, Paper, Session]:
    user = User(auth_subject=subject)
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
    await ingest_paper(
        db_session, paper.paper_id, storage=storage, embedder=HashingEmbedder()
    )

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


# --------------------------------------------------------------------------
# The SSE contract
# --------------------------------------------------------------------------


async def test_a_turn_emits_the_contracted_event_sequence(
    db_session: AsyncSession, storage, scripted_agent
):
    scripted_agent("Attention weights every token pair [1].")
    user, _, conversation = await _reader_with_paper(
        db_session, storage, f"turns-{uuid.uuid4()}"
    )

    events = await _run(db_session, conversation, user, "How does attention work?")
    names = [e["event"] for e in events]

    assert names[0] == "state"
    assert "token" in names
    assert names.index("citations") > names.index("token")
    assert "memory_used" in names
    assert names[-1] == "done"


async def test_tokens_are_only_streamed_after_verification(
    db_session: AsyncSession, storage, scripted_agent
):
    """Nothing streamed is ever retracted.

    An invented marker is removed *before* the first token, so a marker the
    reader sees always resolves.
    """
    scripted_agent("Grounded [1]. Invented [9].")
    user, _, conversation = await _reader_with_paper(
        db_session, storage, f"turns-{uuid.uuid4()}"
    )

    events = await _run(db_session, conversation, user, "How does attention work?")
    streamed = "".join(e["text"] for e in events if e["event"] == "token")

    assert "[9]" not in streamed
    assert "[1]" in streamed


async def test_the_citations_event_carries_what_the_overlay_needs(
    db_session: AsyncSession, storage, scripted_agent
):
    scripted_agent("Attention weights every token pair [1].")
    user, paper, conversation = await _reader_with_paper(
        db_session, storage, f"turns-{uuid.uuid4()}"
    )

    events = await _run(db_session, conversation, user, "How does attention work?")
    citations = next(e for e in events if e["event"] == "citations")["citations"]

    assert len(citations) == 1
    entry = citations[0]
    assert entry["marker"] == "[1]"
    assert entry["paper_id"] == str(paper.paper_id)
    assert entry["page_start"] >= 1
    assert entry["section_path"]


async def test_state_events_expose_the_agents_tool_choices(
    db_session: AsyncSession, storage, scripted_agent
):
    """The visible trace of agency, for the debug strip and the video."""
    scripted_agent("Answer [1].")
    user, _, conversation = await _reader_with_paper(
        db_session, storage, f"turns-{uuid.uuid4()}"
    )

    events = await _run(db_session, conversation, user, "How does attention work?")
    tools = [t for e in events if e["event"] == "state" for t in e["tools_called"]]

    assert "retrieve_paper_context" in tools


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


async def test_a_turn_persists_metadata_and_the_transcript_separately(
    db_session: AsyncSession, storage, scripted_agent
):
    scripted_agent("Attention weights every token pair [1].")
    user, _, conversation = await _reader_with_paper(
        db_session, storage, f"turns-{uuid.uuid4()}"
    )

    await _run(db_session, conversation, user, "How does attention work?")

    turn = await db_session.scalar(
        select(Turn).where(Turn.session_id == conversation.session_id)
    )
    assert turn is not None
    assert turn.grounding_status == "grounded"
    assert turn.tools_called == ["retrieve_paper_context"]
    assert turn.latency_ms is not None

    messages = (
        await db_session.scalars(
            select(Message)
            .where(Message.session_id == conversation.session_id)
            .order_by(Message.ordinal)
        )
    ).all()
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "How does attention work?"
    assert messages[0].turn_id == turn.turn_id


async def test_only_cited_retrievals_are_flagged(
    db_session: AsyncSession, storage, scripted_agent
):
    """A citation *is* a retrieval row with was_cited set."""
    scripted_agent("Only the first passage matters [1].")
    user, _, conversation = await _reader_with_paper(
        db_session, storage, f"turns-{uuid.uuid4()}"
    )

    await _run(db_session, conversation, user, "How does attention work?")

    turn = await db_session.scalar(
        select(Turn).where(Turn.session_id == conversation.session_id)
    )
    rows = (
        await db_session.scalars(
            select(TurnRetrieval).where(TurnRetrieval.turn_id == turn.turn_id)
        )
    ).all()

    assert rows, "the retrieval set is recorded whether or not it was cited"
    cited = [r for r in rows if r.was_cited]
    assert len(cited) == 1
    assert cited[0].citation_marker == "[1]"
    assert all(r.citation_marker is None for r in rows if not r.was_cited)


async def test_history_is_carried_into_the_next_turn(
    db_session: AsyncSession, storage, scripted_agent
):
    captured = scripted_agent("Answer [1].")
    user, _, conversation = await _reader_with_paper(
        db_session, storage, f"turns-{uuid.uuid4()}"
    )

    await _run(db_session, conversation, user, "First question?")
    await _run(db_session, conversation, user, "Second question?")

    history = captured["history"]
    assert [h.content for h in history] == ["First question?", "Answer [1]."]


async def test_turn_ordinals_increment_within_a_session(
    db_session: AsyncSession, storage, scripted_agent
):
    scripted_agent("Answer [1].")
    user, _, conversation = await _reader_with_paper(
        db_session, storage, f"turns-{uuid.uuid4()}"
    )

    await _run(db_session, conversation, user, "one?")
    await _run(db_session, conversation, user, "two?")

    ordinals = (
        await db_session.scalars(
            select(Turn.ordinal)
            .where(Turn.session_id == conversation.session_id)
            .order_by(Turn.ordinal)
        )
    ).all()
    assert list(ordinals) == [0, 1]


# --------------------------------------------------------------------------
# Grounding and scope
# --------------------------------------------------------------------------


async def test_an_ungrounded_answer_is_recorded_as_such(
    db_session: AsyncSession, storage, scripted_agent
):
    scripted_agent("I happen to know this already.", call_tool=False)
    user, _, conversation = await _reader_with_paper(
        db_session, storage, f"turns-{uuid.uuid4()}"
    )

    events = await _run(db_session, conversation, user, "How does attention work?")

    done = next(e for e in events if e["event"] == "done")
    assert done["grounding_status"] == "no_evidence"


async def test_a_session_with_no_paper_retrieves_nothing(
    db_session: AsyncSession, storage, scripted_agent
):
    """Empty scope must not fall back to searching everything."""
    scripted_agent("No paper is open.", call_tool=False)

    user = User(auth_subject=f"turns-{uuid.uuid4()}")
    db_session.add(user)
    await db_session.flush()
    conversation = Session(user_id=user.user_id)
    db_session.add(conversation)
    await db_session.flush()

    events = await _run(db_session, conversation, user, "Anything?")

    citations = next(e for e in events if e["event"] == "citations")["citations"]
    assert citations == []


async def test_a_revoked_grant_empties_the_scope_at_read_time(
    db_session: AsyncSession, storage, scripted_agent
):
    from datetime import UTC, datetime

    scripted_agent("Nothing to cite.", call_tool=True)
    user, paper, conversation = await _reader_with_paper(
        db_session, storage, f"turns-{uuid.uuid4()}"
    )

    grant = await db_session.scalar(
        select(UserPaperAccess).where(
            UserPaperAccess.user_id == user.user_id,
            UserPaperAccess.paper_id == paper.paper_id,
        )
    )
    grant.revoked_at = datetime.now(UTC)
    await db_session.flush()

    await _run(db_session, conversation, user, "How does attention work?")

    turn = await db_session.scalar(
        select(Turn).where(Turn.session_id == conversation.session_id)
    )
    retrievals = await db_session.scalar(
        select(func.count())
        .select_from(TurnRetrieval)
        .where(TurnRetrieval.turn_id == turn.turn_id)
    )
    assert retrievals == 0


# --------------------------------------------------------------------------
# The HTTP surface
# --------------------------------------------------------------------------


async def test_creating_a_session_requires_a_readable_paper(
    client: AsyncClient, db_session: AsyncSession, dev_auth
):
    """An ungranted paper is a 404, not a 403 — a 403 confirms the id."""
    await client.get("/api/me")
    paper = Paper(
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        storage_uri=f"file://{uuid.uuid4()}.pdf",
        processing_status="ready",
    )
    db_session.add(paper)
    await db_session.flush()

    response = await client.post("/api/sessions", json={"paper_id": str(paper.paper_id)})
    assert response.status_code == 404


async def test_a_session_belonging_to_someone_else_is_not_found(
    client: AsyncClient, db_session: AsyncSession, dev_auth
):
    await client.get("/api/me")
    stranger = User(auth_subject=f"stranger-{uuid.uuid4()}")
    db_session.add(stranger)
    await db_session.flush()
    theirs = Session(user_id=stranger.user_id)
    db_session.add(theirs)
    await db_session.flush()

    response = await client.get(f"/api/sessions/{theirs.session_id}")
    assert response.status_code == 404


async def test_a_session_can_be_opened_and_read_back(
    client: AsyncClient, dev_auth
):
    created = await client.post("/api/sessions", json={})
    assert created.status_code == 201

    session_id = created.json()["session_id"]
    fetched = await client.get(f"/api/sessions/{session_id}")

    assert fetched.status_code == 200
    assert fetched.json()["activity"] == "FREE"
    assert fetched.json()["turn_count"] == 0


async def test_listing_sessions_returns_only_the_callers_own(
    client: AsyncClient, db_session: AsyncSession, dev_auth
):
    """The rail is built from this, so a leak here is a leak on screen."""
    await client.get("/api/me")
    mine = (await client.post("/api/sessions", json={})).json()["session_id"]

    stranger = User(auth_subject=f"stranger-{uuid.uuid4()}")
    db_session.add(stranger)
    await db_session.flush()
    db_session.add(Session(user_id=stranger.user_id))
    await db_session.flush()

    listed = (await client.get("/api/sessions")).json()

    assert [s["session_id"] for s in listed] == [mine]


async def test_a_listed_session_carries_the_paper_title(
    client: AsyncClient, db_session: AsyncSession, dev_auth
):
    """Otherwise the rail would need one request per row to label itself."""
    await client.get("/api/me")
    user = await db_session.scalar(select(User).where(User.auth_subject == dev_auth))
    paper = Paper(
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        storage_uri=f"file://{uuid.uuid4()}.pdf",
        title="Auto-Encoding Variational Bayes",
        processing_status="ready",
    )
    db_session.add(paper)
    await db_session.flush()
    db_session.add(UserPaperAccess(user_id=user.user_id, paper_id=paper.paper_id))
    await db_session.flush()

    await client.post("/api/sessions", json={"paper_id": str(paper.paper_id)})
    listed = (await client.get("/api/sessions")).json()

    assert listed[0]["paper_title"] == "Auto-Encoding Variational Bayes"
    assert listed[0]["active_paper_id"] == str(paper.paper_id)


async def test_a_session_with_no_paper_lists_a_null_title(
    client: AsyncClient, dev_auth
):
    """The outer join must not drop the row."""
    await client.post("/api/sessions", json={})

    listed = (await client.get("/api/sessions")).json()

    assert len(listed) == 1
    assert listed[0]["paper_title"] is None


async def test_the_transcript_endpoint_returns_durable_history(
    client: AsyncClient, dev_auth
):
    """What a page reload rebuilds from."""
    session_id = (await client.post("/api/sessions", json={})).json()["session_id"]

    response = await client.get(f"/api/sessions/{session_id}/messages")
    assert response.status_code == 200
    assert response.json() == []
