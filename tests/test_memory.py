"""Reading the learner model: scope, decay, graph traversal, record shape.

`MemoryService` is the `[D]` half of the differentiator. The rules it has to
keep are the ones the architecture calls structural: user scope is not
addressable, records carry no transcript, and a paper remembered is not a paper
authorized.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Concept,
    ConceptRelationship,
    Observation,
    Paper,
    User,
    UserPaperAccess,
)
from app.ingestion.concepts import normalize_name
from app.services.embeddings import get_embedder
from app.services.memory import MemoryService


async def _user(session: AsyncSession, tag: str = "memory") -> User:
    user = User(auth_subject=f"{tag}-test-{uuid.uuid4()}")
    session.add(user)
    await session.flush()
    return user


async def _concept(
    session: AsyncSession,
    user: User,
    name: str,
    *,
    score: float | None = None,
    confidence: float | None = None,
    style: str | None = None,
    last_reinforced_at: datetime | None = None,
    override: float | None = None,
    source_paper_ids: list[uuid.UUID] | None = None,
) -> Concept:
    # `normalized_name` is unique per user, and every test makes its own user,
    # so the real normalized form is safe here and keeps the exact-match path
    # under test rather than forcing everything through ANN.
    concept = Concept(
        user_id=user.user_id,
        canonical_name=name,
        normalized_name=normalize_name(name),
        embedding=get_embedder().embed_query(name),
        understanding_score=score,
        score_confidence=confidence,
        effective_style=style,
        last_reinforced_at=last_reinforced_at,
        user_override_score=override,
        source_paper_ids=source_paper_ids or [],
    )
    session.add(concept)
    await session.flush()
    return concept


def _service(session: AsyncSession) -> MemoryService:
    return MemoryService(session, embedder=get_embedder())


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------


async def test_lookup_never_crosses_users(db_session: AsyncSession):
    """The isolation guarantee, at the memory layer."""
    mine = await _user(db_session, "mine")
    theirs = await _user(db_session, "theirs")

    await _concept(db_session, theirs, "Their Secret Concept", score=0.2, confidence=0.9)

    records = await _service(db_session).lookup(mine.user_id, query="secret")

    assert records == []


async def test_by_name_is_user_scoped(db_session: AsyncSession):
    mine = await _user(db_session, "mine")
    theirs = await _user(db_session, "theirs")

    await _concept(db_session, theirs, "Reparameterization trick")

    assert (
        await _service(db_session).by_name(mine.user_id, "Reparameterization trick")
        is None
    )


# --------------------------------------------------------------------------
# Record shape and decay
# --------------------------------------------------------------------------


async def test_the_record_carries_no_transcript(db_session: AsyncSession):
    """Compact records only — scores and names, never conversation text."""
    user = await _user(db_session)
    concept = await _concept(
        db_session, user, "ELBO", score=0.4, confidence=0.8, style="numerical"
    )
    db_session.add(
        Observation(
            user_id=user.user_id,
            concept_id=concept.concept_id,
            signal_type="explicit_confusion",
            signal_source="explicit",
            weight=0.8,
            note="Lost track at the KL term.",
        )
    )
    await db_session.flush()

    records = await _service(db_session).lookup(user.user_id, concept_name="ELBO")
    payload = records[0].for_model()

    assert set(payload) == {
        "concept",
        "understanding_score",
        "score_confidence",
        "effective_style",
        "last_seen",
        "evidence_count",
        "evidence_note",
        "related",
    }
    # The one free-text field is the curated evidence line, not a transcript.
    assert payload["evidence_note"] == "Lost track at the KL term."


async def test_scores_are_decayed_at_read_time(db_session: AsyncSession):
    """ARCHITECTURE 17 — the stored row is raw; staleness is applied on read."""
    user = await _user(db_session)
    stale = datetime.now(UTC) - timedelta(days=30)
    concept = await _concept(
        db_session, user, "Forward Process", score=0.8, confidence=0.9,
        last_reinforced_at=stale,
    )

    records = await _service(db_session).lookup(
        user.user_id, concept_name="Forward Process"
    )

    assert concept.understanding_score == pytest.approx(0.8)
    assert records[0].understanding_score == pytest.approx(0.4, abs=0.02)


async def test_an_explicit_override_outranks_inference_and_does_not_decay(
    db_session: AsyncSession,
):
    """A correction the reader made about themselves is not an inference."""
    user = await _user(db_session)
    long_ago = datetime.now(UTC) - timedelta(days=120)
    await _concept(
        db_session, user, "KL divergence", score=0.1, confidence=0.9,
        last_reinforced_at=long_ago, override=0.95,
    )

    records = await _service(db_session).lookup(
        user.user_id, concept_name="KL divergence"
    )

    assert records[0].understanding_score == pytest.approx(0.95)


async def test_only_weak_filters_on_weak_and_confident(db_session: AsyncSession):
    user = await _user(db_session)
    await _concept(db_session, user, "Weak But Sure", score=0.2, confidence=0.8)
    await _concept(db_session, user, "Weak And Unsure", score=0.2, confidence=0.1)
    await _concept(db_session, user, "Strong", score=0.9, confidence=0.9)

    records = await _service(db_session).lookup(
        user.user_id, only_weak=True, include_related=False
    )

    names = {record.canonical_name for record in records}
    assert "Weak And Unsure" not in names, "low confidence is a reason to ask, not tell"
    assert "Strong" not in names


# --------------------------------------------------------------------------
# The graph
# --------------------------------------------------------------------------


async def test_traversal_finds_edges_in_both_directions(db_session: AsyncSession):
    """Symmetric types are stored once with a canonical orientation
    (ARCHITECTURE 4.10), so a source-only walk loses half the graph."""
    user = await _user(db_session)
    x = await _concept(db_session, user, "Forward Process")
    y = await _concept(db_session, user, "Variational lower bound")

    db_session.add(
        ConceptRelationship(
            user_id=user.user_id,
            source_concept_id=x.concept_id,
            target_concept_id=y.concept_id,
            relationship_type="prerequisite_of",
            confidence=0.7,
            discovery_method="model",
        )
    )
    await db_session.flush()

    service = _service(db_session)
    from_source = await service.neighbours(user.user_id, [x.concept_id])
    from_target = await service.neighbours(user.user_id, [y.concept_id])

    assert from_source[x.concept_id][0].name == "Variational lower bound"
    assert from_target[y.concept_id][0].name == "Forward Process"


async def test_traversal_is_user_scoped(db_session: AsyncSession):
    mine = await _user(db_session, "mine")
    theirs = await _user(db_session, "theirs")
    a = await _concept(db_session, theirs, "A")
    b = await _concept(db_session, theirs, "B")
    db_session.add(
        ConceptRelationship(
            user_id=theirs.user_id,
            source_concept_id=a.concept_id,
            target_concept_id=b.concept_id,
            relationship_type="component_of",
            confidence=0.9,
            discovery_method="model",
        )
    )
    await db_session.flush()

    edges = await _service(db_session).neighbours(mine.user_id, [a.concept_id])

    assert edges == {}


# --------------------------------------------------------------------------
# Provenance is not authorization
# --------------------------------------------------------------------------


async def test_source_papers_are_filtered_through_the_grant(db_session: AsyncSession):
    """ARCHITECTURE 12 step 7 — the critical checkpoint.

    A concept remembers which paper introduced it, and that memory outlives a
    revoked grant. Memory pointing at a paper is not authorization to read it.
    """
    user = await _user(db_session)
    granted = Paper(
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        storage_uri=f"file://{uuid.uuid4()}.pdf",
        processing_status="ready",
        title="Granted paper",
    )
    revoked = Paper(
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        storage_uri=f"file://{uuid.uuid4()}.pdf",
        processing_status="ready",
        title="Revoked paper",
    )
    db_session.add_all([granted, revoked])
    await db_session.flush()

    db_session.add(
        UserPaperAccess(user_id=user.user_id, paper_id=granted.paper_id)
    )
    db_session.add(
        UserPaperAccess(
            user_id=user.user_id,
            paper_id=revoked.paper_id,
            revoked_at=datetime.now(UTC),
        )
    )
    await db_session.flush()

    visible = await _service(db_session).visible_source_papers(
        user.user_id, [granted.paper_id, revoked.paper_id]
    )

    assert [title for _, title in visible] == ["Granted paper"]


async def test_prefetch_on_an_empty_memory_returns_nothing(db_session: AsyncSession):
    """First-ever turn. The agent must not be handed an empty shape it might
    mistake for "I have met them before"."""
    user = await _user(db_session)

    assert await _service(db_session).prefetch(user.user_id, "diffusion models") == []
