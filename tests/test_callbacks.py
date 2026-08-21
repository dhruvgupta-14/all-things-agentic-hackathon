"""The cross-paper callback gate (ARCHITECTURE 12, 9.2 step 10).

The demo's decisive moment, so these are about the gate refusing as much as
firing. In particular step 7: memory pointing at a paper is not authorization
to read it, and a revoked grant has to stop a callback that is otherwise
perfectly good.
"""

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Concept, ConceptRelationship, Paper, Session, Turn, User, UserPaperAccess
from app.ingestion.concepts import normalize_name
from app.services.callbacks import (
    SUPPRESSED_GRANT_REVOKED,
    SUPPRESSED_NO_CANDIDATE,
    SUPPRESSED_NO_MEMORY,
    SUPPRESSED_PERSONALIZATION_OFF,
    SUPPRESSED_PROACTIVITY_OFF,
    SUPPRESSED_RATE_LIMITED,
    CallbackService,
)
from app.services.memory import MemoryService
from tests.fakes import HashingEmbedder


async def _user(session: AsyncSession, **preferences) -> User:
    user = User(
        auth_subject=f"callback-test-{uuid.uuid4()}", preferences=preferences or {}
    )
    session.add(user)
    await session.flush()
    return user


async def _paper(session: AsyncSession, title: str) -> Paper:
    paper = Paper(
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        storage_uri=f"file://{uuid.uuid4()}.pdf",
        processing_status="ready",
        title=title,
    )
    session.add(paper)
    await session.flush()
    return paper


async def _concept(
    session: AsyncSession,
    user: User,
    name: str,
    *,
    papers: list[uuid.UUID] | None = None,
    score: float | None = None,
    confidence: float | None = None,
    style: str | None = None,
) -> Concept:
    concept = Concept(
        user_id=user.user_id,
        canonical_name=name,
        normalized_name=normalize_name(name),
        embedding=HashingEmbedder().embed_query(name),
        source_paper_ids=papers or [],
        understanding_score=score,
        score_confidence=confidence,
        effective_style=style,
        last_reinforced_at=datetime.now(UTC),
    )
    session.add(concept)
    await session.flush()
    return concept


async def _edge(session: AsyncSession, user: User, source, target, kind="component_of"):
    session.add(
        ConceptRelationship(
            user_id=user.user_id,
            source_concept_id=source.concept_id,
            target_concept_id=target.concept_id,
            relationship_type=kind,
            confidence=0.86,
            discovery_method="model",
        )
    )
    await session.flush()


async def _scenario(
    db_session: AsyncSession,
    *,
    grant_prior: bool = True,
    weak_score: float = 0.31,
    weak_confidence: float = 0.72,
    **preferences,
):
    """The §12 setup: a weak concept in paper A, connected to one in paper B."""
    user = await _user(db_session, **preferences)
    paper_a = await _paper(db_session, "Auto-Encoding Variational Bayes")
    paper_b = await _paper(db_session, "Denoising Diffusion Probabilistic Models")

    db_session.add(UserPaperAccess(user_id=user.user_id, paper_id=paper_b.paper_id))
    if grant_prior:
        db_session.add(UserPaperAccess(user_id=user.user_id, paper_id=paper_a.paper_id))
    await db_session.flush()

    struggled = await _concept(
        db_session,
        user,
        "Variational lower bound",
        papers=[paper_a.paper_id],
        score=weak_score,
        confidence=weak_confidence,
        style="numerical",
    )
    asked_about = await _concept(
        db_session, user, "Reverse Process", papers=[paper_b.paper_id]
    )
    await _edge(db_session, user, struggled, asked_about)

    return user, paper_a, paper_b, struggled, asked_about


async def _decide(db_session, user, active_paper, query="reverse process"):
    memory = MemoryService(db_session, embedder=HashingEmbedder())
    prefetched = await memory.prefetch(user.user_id, query)
    return await CallbackService(db_session).decide(
        user=user,
        active_paper_id=active_paper.paper_id if active_paper else None,
        prefetched=prefetched,
    )


# --------------------------------------------------------------------------
# It fires
# --------------------------------------------------------------------------


async def test_the_documented_callback_fires(db_session: AsyncSession):
    """ARCHITECTURE 12 end to end: weak, confident, connected, authorized."""
    user, paper_a, paper_b, struggled, _ = await _scenario(db_session)

    decision = await _decide(db_session, user, paper_b)

    assert decision.fired
    assert decision.concept_id == struggled.concept_id
    assert decision.prior_paper_id == paper_a.paper_id
    assert decision.prior_paper_title == "Auto-Encoding Variational Bayes"
    assert decision.relationship_type == "component_of"
    assert decision.effective_style == "numerical"
    assert decision.suppressed_reason is None


async def test_the_hint_tells_the_agent_how_not_what_to_say(db_session: AsyncSession):
    """No canned sentence anywhere — the same machinery for any paper pair."""
    user, _, paper_b, _, _ = await _scenario(db_session)

    hint = (await _decide(db_session, user, paper_b)).hint()

    assert "Variational lower bound" in hint
    assert "numerical" in hint
    assert "cite" in hint.lower()


async def test_the_decision_is_deterministic(db_session: AsyncSession):
    """A callback that varied run to run would be untestable, and worse, would
    make the demo a coin flip."""
    user, _, paper_b, _, _ = await _scenario(db_session)

    first = await _decide(db_session, user, paper_b)
    second = await _decide(db_session, user, paper_b)

    assert first.concept_id == second.concept_id


# --------------------------------------------------------------------------
# Step 7 — the checkpoint to scrutinise
# --------------------------------------------------------------------------


async def test_a_revoked_grant_stops_an_otherwise_good_callback(
    db_session: AsyncSession,
):
    """Memory pointing at a paper is not authorization to read it."""
    user, _, paper_b, _, _ = await _scenario(db_session, grant_prior=False)

    decision = await _decide(db_session, user, paper_b)

    assert not decision.fired
    assert decision.suppressed_reason == SUPPRESSED_GRANT_REVOKED


async def test_a_concept_only_in_the_active_paper_is_not_a_callback(
    db_session: AsyncSession,
):
    """A callback is *cross*-paper. Connecting a paper to itself is just an
    answer."""
    user = await _user(db_session)
    paper = await _paper(db_session, "One paper")
    db_session.add(UserPaperAccess(user_id=user.user_id, paper_id=paper.paper_id))
    await db_session.flush()

    weak = await _concept(
        db_session, user, "Variational lower bound",
        papers=[paper.paper_id], score=0.2, confidence=0.9,
    )
    asked = await _concept(
        db_session, user, "Reverse Process", papers=[paper.paper_id]
    )
    await _edge(db_session, user, weak, asked)

    decision = await _decide(db_session, user, paper)

    assert not decision.fired
    assert decision.suppressed_reason == SUPPRESSED_GRANT_REVOKED


# --------------------------------------------------------------------------
# Steps 4-6 — weakness, rate limit, preferences
# --------------------------------------------------------------------------


async def test_a_well_understood_concept_is_not_a_callback(db_session: AsyncSession):
    user, _, paper_b, _, _ = await _scenario(db_session, weak_score=0.95)

    decision = await _decide(db_session, user, paper_b)

    assert decision.suppressed_reason == SUPPRESSED_NO_CANDIDATE


async def test_a_low_score_we_are_unsure_about_is_not_a_callback(
    db_session: AsyncSession,
):
    """"A reason to ask, not to announce" (ARCHITECTURE 10.2 step 4)."""
    user, _, paper_b, _, _ = await _scenario(
        db_session, weak_score=0.1, weak_confidence=0.05
    )

    decision = await _decide(db_session, user, paper_b)

    assert decision.suppressed_reason == SUPPRESSED_NO_CANDIDATE


async def test_nothing_in_memory_suppresses_with_a_reason(db_session: AsyncSession):
    user = await _user(db_session)
    paper = await _paper(db_session, "A paper")

    decision = await _decide(db_session, user, paper)

    assert decision.suppressed_reason == SUPPRESSED_NO_MEMORY


async def test_a_recent_callback_rate_limits_the_next_one(db_session: AsyncSession):
    """ARCHITECTURE 12 step 6 — one callback, then breathing room."""
    user, _, paper_b, struggled, _ = await _scenario(db_session)

    conversation = Session(user_id=user.user_id, active_paper_id=paper_b.paper_id)
    db_session.add(conversation)
    await db_session.flush()
    db_session.add(
        Turn(
            session_id=conversation.session_id,
            user_id=user.user_id,
            ordinal=0,
            memory_read=True,
            callback_concept_id=struggled.concept_id,
        )
    )
    await db_session.flush()

    decision = await _decide(db_session, user, paper_b)

    assert decision.suppressed_reason == SUPPRESSED_RATE_LIMITED


async def test_the_rate_limit_lifts_after_enough_turns(db_session: AsyncSession):
    from app.services.learner_state import CALLBACK_MIN_TURN_GAP

    user, _, paper_b, struggled, _ = await _scenario(db_session)

    conversation = Session(user_id=user.user_id, active_paper_id=paper_b.paper_id)
    db_session.add(conversation)
    await db_session.flush()

    long_ago = datetime.now(UTC) - timedelta(hours=1)
    db_session.add(
        Turn(
            session_id=conversation.session_id,
            user_id=user.user_id,
            ordinal=0,
            memory_read=True,
            callback_concept_id=struggled.concept_id,
            created_at=long_ago,
        )
    )
    for ordinal in range(1, CALLBACK_MIN_TURN_GAP + 1):
        db_session.add(
            Turn(
                session_id=conversation.session_id,
                user_id=user.user_id,
                ordinal=ordinal,
                created_at=long_ago + timedelta(minutes=ordinal),
            )
        )
    await db_session.flush()

    decision = await _decide(db_session, user, paper_b)

    assert decision.fired


async def test_personalization_off_suppresses_everything(db_session: AsyncSession):
    user, _, paper_b, _, _ = await _scenario(
        db_session, personalization_enabled=False
    )

    decision = await _decide(db_session, user, paper_b)

    assert decision.suppressed_reason == SUPPRESSED_PERSONALIZATION_OFF


async def test_proactivity_off_suppresses_callbacks(db_session: AsyncSession):
    user, _, paper_b, _, _ = await _scenario(db_session, proactivity="off")

    decision = await _decide(db_session, user, paper_b)

    assert decision.suppressed_reason == SUPPRESSED_PROACTIVITY_OFF


async def test_every_outcome_carries_a_reason_or_a_concept(db_session: AsyncSession):
    """Suppression is a feature and is measured — there is no silent path."""
    user, _, paper_b, _, _ = await _scenario(db_session, weak_score=0.99)

    decision = await _decide(db_session, user, paper_b)

    assert (decision.concept_id is None) != (decision.suppressed_reason is None)
    assert len(decision.suppressed_reason) <= 64


async def test_the_gate_never_reaches_another_readers_memory(
    db_session: AsyncSession,
):
    user, _, paper_b, _, _ = await _scenario(db_session)
    stranger = await _user(db_session)

    memory = MemoryService(db_session, embedder=HashingEmbedder())
    prefetched = await memory.prefetch(stranger.user_id, "reverse process")
    decision = await CallbackService(db_session).decide(
        user=stranger, active_paper_id=paper_b.paper_id, prefetched=prefetched
    )

    assert not decision.fired
