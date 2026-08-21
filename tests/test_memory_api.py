"""The learner-memory HTTP surface (ARCHITECTURE 15).

These routes show a reader everything the system believes about them, so the
things worth pinning are: another reader's memory is unreachable, the evidence
carries the provenance that makes it interrogable, and a correction outranks
inference without being silently overwritten.
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Concept,
    ConceptRelationship,
    Observation,
    Paper,
    Session,
    Turn,
    User,
    UserPaperAccess,
)
from app.ingestion.concepts import normalize_name
from tests.fakes import HashingEmbedder


async def _principal(db_session: AsyncSession, signed_in: str) -> User:
    """The user the dev bypass authenticates as for this test."""
    user = await db_session.scalar(select(User).where(User.auth_subject == signed_in))
    if user is None:
        user = User(auth_subject=signed_in)
        db_session.add(user)
        await db_session.flush()
    return user


async def _concept(
    db_session: AsyncSession,
    user: User,
    name: str,
    *,
    score=None,
    confidence=None,
    style=None,
    papers=None,
) -> Concept:
    concept = Concept(
        user_id=user.user_id,
        canonical_name=name,
        normalized_name=normalize_name(name),
        embedding=HashingEmbedder().embed_query(name),
        understanding_score=score,
        score_confidence=confidence,
        effective_style=style,
        source_paper_ids=papers or [],
    )
    db_session.add(concept)
    await db_session.flush()
    return concept


# --------------------------------------------------------------------------
# Listing
# --------------------------------------------------------------------------


async def test_listing_returns_this_readers_concepts(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    user = await _principal(db_session, signed_in)
    await _concept(db_session, user, "ELBO", score=0.3, confidence=0.8, style="numerical")

    body = (await client.get("/api/memory/concepts")).json()

    names = [c["canonical_name"] for c in body["concepts"]]
    assert "ELBO" in names
    entry = next(c for c in body["concepts"] if c["canonical_name"] == "ELBO")
    assert entry["effective_style"] == "numerical"
    assert entry["is_weak"] is True


async def test_listing_never_crosses_users(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    await _principal(db_session, signed_in)
    stranger = User(auth_subject=f"stranger-{uuid.uuid4()}")
    db_session.add(stranger)
    await db_session.flush()
    await _concept(
        db_session, stranger, "Their Private Concept", score=0.1, confidence=0.9
    )

    body = (await client.get("/api/memory/concepts")).json()

    assert all(
        c["canonical_name"] != "Their Private Concept" for c in body["concepts"]
    )


async def test_the_thresholds_travel_with_the_payload(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    """So the UI cannot drift from the gate that fires callbacks."""
    await _principal(db_session, signed_in)

    body = (await client.get("/api/memory/concepts")).json()

    assert body["weak_below"] == pytest.approx(0.40)
    assert body["confidence_floor"] == pytest.approx(0.30)


async def test_only_weak_filters(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    user = await _principal(db_session, signed_in)
    await _concept(db_session, user, "Weak Thing", score=0.2, confidence=0.8)
    await _concept(db_session, user, "Solid Thing", score=0.95, confidence=0.9)

    body = (await client.get("/api/memory/concepts?only_weak=true")).json()

    names = [c["canonical_name"] for c in body["concepts"]]
    assert "Weak Thing" in names
    assert "Solid Thing" not in names


# --------------------------------------------------------------------------
# Detail — the "why do you think that?" answer
# --------------------------------------------------------------------------


async def test_detail_carries_evidence_with_turn_provenance(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    """The link that turns a score into something a reader can interrogate."""
    user = await _principal(db_session, signed_in)
    concept = await _concept(db_session, user, "ELBO", score=0.35, confidence=0.7)

    conversation = Session(user_id=user.user_id)
    db_session.add(conversation)
    await db_session.flush()
    turn = Turn(session_id=conversation.session_id, user_id=user.user_id, ordinal=0)
    db_session.add(turn)
    await db_session.flush()

    db_session.add(
        Observation(
            user_id=user.user_id,
            concept_id=concept.concept_id,
            turn_id=turn.turn_id,
            signal_type="explicit_confusion",
            signal_source="explicit",
            weight=0.8,
            style_in_play="formal",
            note="Lost track at the KL term.",
        )
    )
    await db_session.flush()

    body = (await client.get(f"/api/memory/concepts/{concept.concept_id}")).json()

    assert len(body["evidence"]) == 1
    evidence = body["evidence"][0]
    assert evidence["signal_type"] == "explicit_confusion"
    assert evidence["note"] == "Lost track at the KL term."
    assert evidence["turn_id"] == str(turn.turn_id)


async def test_detail_includes_related_concepts(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    user = await _principal(db_session, signed_in)
    concept = await _concept(db_session, user, "ELBO")
    other = await _concept(db_session, user, "KL divergence")
    db_session.add(
        ConceptRelationship(
            user_id=user.user_id,
            source_concept_id=other.concept_id,
            target_concept_id=concept.concept_id,
            relationship_type="prerequisite_of",
            confidence=0.8,
            discovery_method="model",
        )
    )
    await db_session.flush()

    body = (await client.get(f"/api/memory/concepts/{concept.concept_id}")).json()

    assert [r["name"] for r in body["related"]] == ["KL divergence"]
    assert body["related"][0]["relationship_type"] == "prerequisite_of"


async def test_source_papers_are_filtered_through_the_grant(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    """Memory pointing at a paper is not authorization to read it."""
    user = await _principal(db_session, signed_in)
    paper = Paper(
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        storage_uri=f"file://{uuid.uuid4()}.pdf",
        processing_status="ready",
        title="Ungranted paper",
    )
    db_session.add(paper)
    await db_session.flush()
    concept = await _concept(db_session, user, "ELBO", papers=[paper.paper_id])

    body = (await client.get(f"/api/memory/concepts/{concept.concept_id}")).json()

    assert body["source_papers"] == []

    db_session.add(UserPaperAccess(user_id=user.user_id, paper_id=paper.paper_id))
    await db_session.flush()

    body = (await client.get(f"/api/memory/concepts/{concept.concept_id}")).json()
    assert [p["title"] for p in body["source_papers"]] == ["Ungranted paper"]


async def test_another_readers_concept_is_a_404_not_a_403(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    """A 403 would confirm the id is real."""
    await _principal(db_session, signed_in)
    stranger = User(auth_subject=f"stranger-{uuid.uuid4()}")
    db_session.add(stranger)
    await db_session.flush()
    theirs = await _concept(db_session, stranger, "Their Concept")

    response = await client.get(f"/api/memory/concepts/{theirs.concept_id}")

    assert response.status_code == 404


# --------------------------------------------------------------------------
# Correction
# --------------------------------------------------------------------------


async def test_a_correction_overrides_inference_and_is_recorded_as_evidence(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    """It outranks the score *and* joins the evidence trail — the score stays
    reproducible from `observations` either way."""
    user = await _principal(db_session, signed_in)
    concept = await _concept(db_session, user, "ELBO", score=0.2, confidence=0.8)

    response = await client.patch(
        f"/api/memory/concepts/{concept.concept_id}",
        json={"understanding_score": 0.9, "note": "I actually know this well."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["user_override_score"] == pytest.approx(0.9)
    assert body["understanding_score"] == pytest.approx(0.9)

    observation = await db_session.scalar(
        select(Observation).where(
            Observation.concept_id == concept.concept_id,
            Observation.signal_source == "user_stated",
        )
    )
    assert observation is not None
    assert observation.signal_type == "user_stated_known"
    assert observation.note == "I actually know this well."


async def test_a_correction_downward_records_the_other_signal(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    user = await _principal(db_session, signed_in)
    concept = await _concept(db_session, user, "ELBO", score=0.9, confidence=0.8)

    await client.patch(
        f"/api/memory/concepts/{concept.concept_id}",
        json={"understanding_score": 0.1},
    )

    observation = await db_session.scalar(
        select(Observation).where(
            Observation.concept_id == concept.concept_id,
            Observation.signal_source == "user_stated",
        )
    )
    assert observation.signal_type == "user_stated_unknown"


async def test_a_correction_cannot_reach_another_reader(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    await _principal(db_session, signed_in)
    stranger = User(auth_subject=f"stranger-{uuid.uuid4()}")
    db_session.add(stranger)
    await db_session.flush()
    theirs = await _concept(db_session, stranger, "Their Concept", score=0.5)

    response = await client.patch(
        f"/api/memory/concepts/{theirs.concept_id}",
        json={"understanding_score": 0.0},
    )

    assert response.status_code == 404
    await db_session.refresh(theirs)
    assert theirs.user_override_score is None


async def test_an_out_of_range_correction_is_refused(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    user = await _principal(db_session, signed_in)
    concept = await _concept(db_session, user, "ELBO")

    response = await client.patch(
        f"/api/memory/concepts/{concept.concept_id}",
        json={"understanding_score": 1.5},
    )

    assert response.status_code == 422


# --------------------------------------------------------------------------
# The graph
# --------------------------------------------------------------------------


async def test_the_graph_returns_nodes_and_typed_edges(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    user = await _principal(db_session, signed_in)
    a = await _concept(db_session, user, "ELBO", score=0.2, confidence=0.8)
    b = await _concept(db_session, user, "KL divergence")
    db_session.add(
        ConceptRelationship(
            user_id=user.user_id,
            source_concept_id=b.concept_id,
            target_concept_id=a.concept_id,
            relationship_type="prerequisite_of",
            confidence=0.75,
            discovery_method="model",
        )
    )
    await db_session.flush()

    body = (await client.get("/api/memory/graph")).json()

    ids = {node["concept_id"] for node in body["nodes"]}
    assert str(a.concept_id) in ids and str(b.concept_id) in ids

    edge = next(
        e for e in body["edges"] if e["source"] == str(b.concept_id)
    )
    assert edge["type"] == "prerequisite_of"
    assert edge["confidence"] == pytest.approx(0.75)

    weak = next(n for n in body["nodes"] if n["concept_id"] == str(a.concept_id))
    assert weak["is_weak"] is True


async def test_the_graph_is_user_scoped(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    await _principal(db_session, signed_in)
    stranger = User(auth_subject=f"stranger-{uuid.uuid4()}")
    db_session.add(stranger)
    await db_session.flush()
    theirs = await _concept(db_session, stranger, "Their Concept")

    body = (await client.get("/api/memory/graph")).json()

    assert str(theirs.concept_id) not in {n["concept_id"] for n in body["nodes"]}


async def test_an_edge_never_points_at_a_missing_node(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    """Otherwise the view draws a line into nothing."""
    user = await _principal(db_session, signed_in)
    await _concept(db_session, user, "ELBO")

    body = (await client.get("/api/memory/graph")).json()

    ids = {node["concept_id"] for node in body["nodes"]}
    for edge in body["edges"]:
        assert edge["source"] in ids and edge["target"] in ids


# --------------------------------------------------------------------------
# A turn's citations after reload (HANDOFF 6.5)
# --------------------------------------------------------------------------


async def test_a_turns_citations_survive_a_reload(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    """The real fix for inert pills: the pills come back from the database
    rather than from a localStorage cache that another device does not have."""
    from app.db.models import Chunk, Section, TurnRetrieval

    user = await _principal(db_session, signed_in)
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
        section_path="2.4",
        ordinal=0,
        page_start=4,
        page_end=4,
        section_role="method",
    )
    db_session.add(section)
    await db_session.flush()

    import hashlib

    content = "The reparameterization trick moves the randomness outside."
    chunk = Chunk(
        paper_id=paper.paper_id,
        section_id=section.section_id,
        ordinal=0,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        page_start=4,
        page_end=4,
    )
    db_session.add(chunk)
    await db_session.flush()

    conversation = Session(user_id=user.user_id, active_paper_id=paper.paper_id)
    db_session.add(conversation)
    await db_session.flush()
    turn = Turn(session_id=conversation.session_id, user_id=user.user_id, ordinal=0)
    db_session.add(turn)
    await db_session.flush()

    db_session.add(
        TurnRetrieval(
            turn_id=turn.turn_id,
            chunk_id=chunk.chunk_id,
            rank=1,
            similarity=0.78,
            was_cited=True,
            citation_marker="[1]",
        )
    )
    await db_session.flush()

    body = (await client.get(f"/api/turns/{turn.turn_id}/citations")).json()

    assert len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["marker"] == "[1]"
    assert citation["chunk_id"] == str(chunk.chunk_id)
    assert citation["section_path"] == "2.4"
    assert citation["page_start"] == 4


async def test_uncited_retrievals_are_not_citations(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    """A citation *is* a `was_cited` row. Retrieved-and-unused is not one."""
    from app.db.models import Chunk, Section, TurnRetrieval

    user = await _principal(db_session, signed_in)
    paper = Paper(
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        storage_uri=f"file://{uuid.uuid4()}.pdf",
        processing_status="ready",
    )
    db_session.add(paper)
    await db_session.flush()
    section = Section(
        paper_id=paper.paper_id,
        section_path="1",
        ordinal=0,
        page_start=1,
        page_end=1,
        section_role="introduction",
    )
    db_session.add(section)
    await db_session.flush()

    import hashlib

    content = "Retrieved but never cited."
    chunk = Chunk(
        paper_id=paper.paper_id,
        section_id=section.section_id,
        ordinal=0,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        page_start=1,
        page_end=1,
    )
    db_session.add(chunk)
    await db_session.flush()

    conversation = Session(user_id=user.user_id)
    db_session.add(conversation)
    await db_session.flush()
    turn = Turn(session_id=conversation.session_id, user_id=user.user_id, ordinal=0)
    db_session.add(turn)
    await db_session.flush()
    db_session.add(
        TurnRetrieval(
            turn_id=turn.turn_id,
            chunk_id=chunk.chunk_id,
            rank=1,
            similarity=0.4,
            was_cited=False,
        )
    )
    await db_session.flush()

    body = (await client.get(f"/api/turns/{turn.turn_id}/citations")).json()

    assert body["citations"] == []


async def test_another_readers_turn_is_a_404(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    await _principal(db_session, signed_in)
    stranger = User(auth_subject=f"stranger-{uuid.uuid4()}")
    db_session.add(stranger)
    await db_session.flush()
    conversation = Session(user_id=stranger.user_id)
    db_session.add(conversation)
    await db_session.flush()
    turn = Turn(
        session_id=conversation.session_id, user_id=stranger.user_id, ordinal=0
    )
    db_session.add(turn)
    await db_session.flush()

    response = await client.get(f"/api/turns/{turn.turn_id}/citations")

    assert response.status_code == 404


async def test_a_concept_with_no_evidence_is_still_listed(
    client: AsyncClient, db_session: AsyncSession, signed_in: str
):
    """The confidence floor gates what the system *claims*, not what it shows.

    Ingesting a paper canonicalizes its concepts before anything is known
    about the reader. Hiding those would make the graph look smaller than it
    is and give the reader nothing to correct.
    """
    user = await _principal(db_session, signed_in)
    await _concept(db_session, user, "Freshly Ingested Concept")

    body = (await client.get("/api/memory/concepts")).json()

    entry = next(
        c for c in body["concepts"] if c["canonical_name"] == "Freshly Ingested Concept"
    )
    assert entry["understanding_score"] is None
    assert entry["evidence_count"] == 0
    assert entry["is_weak"] is False, "no evidence is not the same as weak"
