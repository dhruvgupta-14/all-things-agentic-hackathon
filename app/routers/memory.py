"""Learner memory as a readable surface (ARCHITECTURE 15).

Everything the agent knows about a reader, shown to that reader. This is the
"why do you think that?" answer, and it is the reason `observations` carries
provenance at all — a score nobody can interrogate is a number, not a memory.

`user_id` appears in no path, query or body. Every query filters on the
verified principal, so there is no cross-user path to construct.

Scores are decayed at read time from `last_reinforced_at` (ARCHITECTURE 17),
so what the UI shows is what the callback gate would act on — never a stale
cached figure that disagrees with the system's own behaviour.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import Principal, get_current_user
from app.db.base import get_db
from app.db.models import Concept, ConceptRelationship, Paper, Turn
from app.services.learner_state import (
    CONFIDENCE_FLOOR,
    WEAK_SCORE_BELOW,
    is_callback_candidate,
    recompute,
)
from app.services.memory import MemoryService
from app.services.signals import SignalRejected, SignalService

router = APIRouter(prefix="/api/memory", tags=["memory"])

# The graph view is a picture, not a dump. Past a few hundred nodes it stops
# being readable and starts being a hairball, so the cap is a design decision
# rather than a pagination stopgap.
MAX_GRAPH_NODES = 300


class CorrectConceptRequest(BaseModel):
    """A reader telling us we were wrong about them."""

    understanding_score: float = Field(ge=0.0, le=1.0)
    note: str | None = Field(default=None, max_length=500)


async def _owned_concept(
    concept_id: uuid.UUID, principal: Principal, db: AsyncSession
) -> Concept:
    """404 rather than 403 — a 403 would confirm the id is real."""
    concept = await db.scalar(
        select(Concept).where(
            Concept.concept_id == concept_id,
            Concept.user_id == principal.user_id,
            Concept.merged_into_id.is_(None),
        )
    )
    if concept is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such concept.")
    return concept


def _serialize(record) -> dict:
    return {
        "concept_id": str(record.concept_id),
        "canonical_name": record.canonical_name,
        "understanding_score": record.understanding_score,
        "score_confidence": record.score_confidence,
        "effective_style": record.effective_style,
        "evidence_count": record.evidence_count,
        "last_reinforced_at": (
            record.last_reinforced_at.isoformat() if record.last_reinforced_at else None
        ),
        # Precomputed so the UI does not re-derive the weakness rule and drift
        # from the gate that actually fires callbacks.
        "is_weak": record.is_weak,
    }


@router.get("/concepts")
async def list_concepts(
    only_weak: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=MAX_GRAPH_NODES),
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Every concept this reader has met, best-evidenced first.

    Unfiltered by default. The confidence floor gates what the system will
    *claim* about a reader, not what it will show them — hiding a concept
    because we have no opinion about it yet would make the view look like the
    graph is smaller than it is.
    """
    records = await MemoryService(db).all_for_user(principal.user_id, limit=limit)
    if only_weak:
        records = [record for record in records if record.is_weak]
    return {
        "concepts": [_serialize(record) for record in records],
        # The thresholds travel with the payload so the UI can label a score
        # without hardcoding a number that lives in `learner_state.py`.
        "weak_below": WEAK_SCORE_BELOW,
        "confidence_floor": CONFIDENCE_FLOOR,
    }


@router.get("/concepts/{concept_id}")
async def get_concept(
    concept_id: uuid.UUID,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """One concept, with the evidence behind its score.

    The evidence list carries `turn_id`, which is what lets a reader click
    from "we think you found this hard" back to the exact exchange that said
    so. That link is the difference between a memory and an assertion.
    """
    concept = await _owned_concept(concept_id, principal, db)
    memory = MemoryService(db)

    records = await memory.lookup(
        principal.user_id, concept_name=concept.canonical_name, include_related=True
    )
    record = next(
        (item for item in records if item.concept_id == concept_id),
        None,
    )
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such concept.")

    observations = await memory.evidence_for(concept_id, limit=20)
    papers = await memory.visible_source_papers(
        principal.user_id, record.source_paper_ids
    )

    payload = _serialize(record)
    payload["user_override_score"] = concept.user_override_score
    payload["aliases"] = list(concept.aliases or [])
    payload["source_papers"] = [
        {"paper_id": str(paper_id), "title": title} for paper_id, title in papers
    ]
    payload["related"] = [
        {
            "concept_id": str(item.concept_id),
            "name": item.name,
            "relationship_type": item.relationship_type,
            "confidence": item.confidence,
        }
        for item in record.related
    ]
    payload["evidence"] = [
        {
            "observation_id": str(observation.observation_id),
            "signal_type": observation.signal_type,
            "signal_source": observation.signal_source,
            "weight": observation.weight,
            "style_in_play": observation.style_in_play,
            "note": observation.note,
            "turn_id": str(observation.turn_id) if observation.turn_id else None,
            "resolved_a_struggle": observation.resolves_observation_id is not None,
            "observed_at": (
                observation.observed_at.isoformat() if observation.observed_at else None
            ),
        }
        for observation in observations
    ]
    return payload


@router.patch("/concepts/{concept_id}")
async def correct_concept(
    concept_id: uuid.UUID,
    body: CorrectConceptRequest,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The reader correcting what we inferred about them.

    Two things happen, and both matter. `user_override_score` is set, and from
    then on it outranks inference and is never silently overwritten. And an
    observation is written with `signal_source = 'user_stated'`, so the
    correction joins the evidence trail rather than sitting outside it — the
    score stays reproducible from `observations` either way.
    """
    concept = await _owned_concept(concept_id, principal, db)

    concept.user_override_score = body.understanding_score
    await db.flush()

    # Which way the reader corrected us decides the signal. They are telling us
    # about themselves, so it is `user_stated` at that weight class.
    signal_type = (
        "user_stated_known"
        if body.understanding_score >= WEAK_SCORE_BELOW
        else "user_stated_unknown"
    )
    try:
        await SignalService(db).record(
            user_id=principal.user_id,
            concept_name=concept.canonical_name,
            signal_type=signal_type,
            note=body.note or "Corrected by the reader.",
        )
    except SignalRejected as exc:  # pragma: no cover - defensive
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    await recompute(db, concept_id)
    await db.commit()

    return await get_concept(concept_id, principal, db)


@router.get("/graph")
async def get_graph(
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The concept graph: typed, directed, confidence-weighted edges.

    Symmetric relationship types are stored once with a canonical orientation
    (ARCHITECTURE 4.10), so this returns each edge once and lets the client
    decide how to draw it. Emitting both directions would double the edge
    count and make the type distribution lie.
    """
    concepts = list(
        (
            await db.scalars(
                select(Concept)
                .where(
                    Concept.user_id == principal.user_id,
                    Concept.merged_into_id.is_(None),
                )
                .order_by(Concept.first_seen_at)
                .limit(MAX_GRAPH_NODES)
            )
        ).all()
    )
    if not concepts:
        return {"nodes": [], "edges": []}

    known = {concept.concept_id for concept in concepts}

    # Which paper introduced each concept, so the view can colour by source and
    # a cross-paper edge is visible as one.
    paper_titles = {
        paper_id: title
        for paper_id, title in (
            await db.execute(select(Paper.paper_id, Paper.title))
        ).all()
    }

    memory = MemoryService(db)
    now_records = await memory.lookup(
        principal.user_id, include_related=False, limit=MAX_GRAPH_NODES
    )
    decayed = {record.concept_id: record for record in now_records}

    nodes = []
    for concept in concepts:
        record = decayed.get(concept.concept_id)
        nodes.append(
            {
                "concept_id": str(concept.concept_id),
                "name": concept.canonical_name,
                "understanding_score": (
                    record.understanding_score if record else concept.understanding_score
                ),
                "score_confidence": concept.score_confidence,
                "evidence_count": concept.evidence_count,
                "is_weak": (
                    record.is_weak
                    if record
                    else is_callback_candidate(
                        concept.understanding_score, concept.score_confidence
                    )
                ),
                "papers": [
                    paper_titles.get(paper_id)
                    for paper_id in (concept.source_paper_ids or [])
                    if paper_id in paper_titles
                ],
            }
        )

    edges = (
        await db.scalars(
            select(ConceptRelationship).where(
                ConceptRelationship.user_id == principal.user_id
            )
        )
    ).all()

    return {
        "nodes": nodes,
        "edges": [
            {
                "source": str(edge.source_concept_id),
                "target": str(edge.target_concept_id),
                "type": edge.relationship_type,
                "confidence": edge.confidence,
                "discovery_method": edge.discovery_method,
            }
            for edge in edges
            # An edge to a concept beyond the node cap would render as a line
            # into nothing.
            if edge.source_concept_id in known and edge.target_concept_id in known
        ],
    }


# --------------------------------------------------------------------------
# A turn's citations, after the fact
# --------------------------------------------------------------------------
# HANDOFF 6.5: the transcript endpoint carries no citation payload, so a
# reloaded conversation had no `chunk_id` to click through to and the pills
# went inert. The SPA worked around it with a localStorage cache, which
# degrades to plain text on another device. This is the real fix, and it reads
# the same `turn_retrievals` rows the verifier wrote.
turns_router = APIRouter(prefix="/api/turns", tags=["citations"])


@turns_router.get("/{turn_id}/citations")
async def list_turn_citations(
    turn_id: uuid.UUID,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Every verified citation for one turn, in marker order.

    The payload matches the `citations` SSE event field for field, so the SPA
    can rehydrate a reloaded transcript through the same rendering path it
    uses live rather than a second, subtly different one.
    """
    from app.db.models import Chunk, Section, TurnRetrieval

    turn = await db.scalar(
        select(Turn).where(
            Turn.turn_id == turn_id, Turn.user_id == principal.user_id
        )
    )
    if turn is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such turn.")

    rows = (
        await db.execute(
            select(TurnRetrieval, Chunk, Section)
            .join(Chunk, Chunk.chunk_id == TurnRetrieval.chunk_id)
            .join(Section, Section.section_id == Chunk.section_id)
            .where(
                TurnRetrieval.turn_id == turn_id,
                # A citation *is* a `was_cited` row. Anything else was
                # retrieved and not used, and is not a citation.
                TurnRetrieval.was_cited.is_(True),
            )
            .order_by(TurnRetrieval.rank)
        )
    ).all()

    return {
        "turn_id": str(turn_id),
        "citations": [
            {
                "marker": retrieval.citation_marker,
                "chunk_id": str(chunk.chunk_id),
                "paper_id": str(chunk.paper_id),
                "section_path": section.section_path,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "similarity": retrieval.similarity,
            }
            for retrieval, chunk, section in rows
        ],
    }
