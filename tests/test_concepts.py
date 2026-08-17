"""Phase 6 — analysis and per-reader concept canonicalization."""

import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Concept, ConceptRelationship, Paper, User
from app.ingestion.concepts import ConceptService, normalize_name
from app.ingestion.pipeline import canonicalize_existing_paper, ingest_paper
from app.services.adjudication import Adjudication, ConservativeAdjudicator
from app.services.analysis import (
    ConceptCandidate,
    HeuristicAnalyzer,
    PaperAnalysis,
)
from app.services.embeddings import HashingEmbedder
from app.services.storage import LocalStorage
from tests.conftest import build_pdf

PAPER_PAGES = [
    "Attention Mechanisms\nAbstract\nScaled dot product attention improves translation.\n"
    "1 Introduction\nScaled dot product attention replaces recurrent processing here.",
    "2 Method\nScaled dot product attention weights token pairs across the sequence.\n"
    "3 Results\nScaled dot product attention beats the recurrent baseline soundly.",
]


@pytest.fixture
def storage(storage_dir) -> LocalStorage:
    return LocalStorage(storage_dir)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


async def _make_user(session: AsyncSession) -> User:
    user = User(auth_subject=f"concept-test-{uuid.uuid4()}")
    session.add(user)
    await session.flush()
    return user


async def _make_paper(session: AsyncSession) -> Paper:
    paper = Paper(
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex[:32],
        storage_uri=f"file://{uuid.uuid4()}.pdf",
        processing_status="ready",
    )
    session.add(paper)
    await session.flush()
    return paper


# --------------------------------------------------------------------------
# Validation and normalization
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Attention Mechanism", "attention mechanism"),
        ("  ATTENTION   mechanism ", "attention mechanism"),
        ("attention-mechanism", "attention mechanism"),
        ("Attention (Mechanism)!", "attention mechanism"),
    ],
)
def test_normalization_treats_case_and_punctuation_as_not_identity(raw, expected):
    assert normalize_name(raw) == expected


def test_analysis_drops_dangling_prerequisite_references():
    """Model output is untrusted: a prerequisite naming nothing is discarded."""
    analysis = PaperAnalysis(
        concepts=[
            ConceptCandidate(name="attention", prerequisites=["softmax", "ghost"]),
            ConceptCandidate(name="softmax"),
        ]
    ).resolved()

    assert analysis.concepts[0].prerequisites == ["softmax"]


def test_analysis_rejects_a_self_prerequisite():
    analysis = PaperAnalysis(
        concepts=[ConceptCandidate(name="attention", prerequisites=["attention"])]
    ).resolved()
    assert analysis.concepts[0].prerequisites == []


def test_implausible_year_is_dropped_not_stored():
    assert PaperAnalysis(year=9999).year is None
    assert PaperAnalysis(year=2017).year == 2017


def test_heuristic_analyzer_is_deterministic():
    analyzer = HeuristicAnalyzer()
    sections = [("method", "scaled dot product attention " * 5)]
    first = analyzer.analyze(None, sections)
    second = analyzer.analyze(None, sections)
    assert [c.name for c in first.concepts] == [c.name for c in second.concepts]


def test_heuristic_analyzer_skips_reference_sections():
    analyzer = HeuristicAnalyzer()
    result = analyzer.analyze(
        None, [("references", "zebra husbandry journal " * 10)]
    )
    assert result.concepts == []


# --------------------------------------------------------------------------
# Canonicalization
# --------------------------------------------------------------------------


async def test_new_concepts_are_created_for_the_user(
    db_session: AsyncSession, embedder
):
    user = await _make_user(db_session)
    paper = await _make_paper(db_session)
    service = ConceptService(db_session, embedder=embedder)

    result = await service.canonicalize(
        user.user_id,
        paper.paper_id,
        [ConceptCandidate(name="Attention Mechanism"), ConceptCandidate(name="Softmax")],
    )

    assert len(result.created) == 2
    stored = (
        await db_session.scalars(
            select(Concept).where(Concept.user_id == user.user_id)
        )
    ).all()
    assert {c.canonical_name for c in stored} == {"Attention Mechanism", "Softmax"}
    assert all(paper.paper_id in c.source_paper_ids for c in stored)


async def test_rerunning_canonicalization_is_idempotent(
    db_session: AsyncSession, embedder
):
    """A retry must not give the reader a second copy of every concept."""
    user = await _make_user(db_session)
    paper = await _make_paper(db_session)
    service = ConceptService(db_session, embedder=embedder)
    candidates = [
        ConceptCandidate(name="Attention Mechanism", prerequisites=["Softmax"]),
        ConceptCandidate(name="Softmax"),
    ]

    first = await service.canonicalize(user.user_id, paper.paper_id, candidates)
    second = await service.canonicalize(user.user_id, paper.paper_id, candidates)

    assert len(first.created) == 2
    assert second.created == []
    assert len(second.matched) == 2

    total = await db_session.scalar(
        select(func.count()).select_from(Concept).where(Concept.user_id == user.user_id)
    )
    assert total == 2

    edges = await db_session.scalar(
        select(func.count())
        .select_from(ConceptRelationship)
        .where(ConceptRelationship.user_id == user.user_id)
    )
    assert edges == 1, "the prerequisite edge must not be duplicated"


async def test_source_paper_ids_is_a_set_union_across_papers(
    db_session: AsyncSession, embedder
):
    user = await _make_user(db_session)
    first_paper = await _make_paper(db_session)
    second_paper = await _make_paper(db_session)
    service = ConceptService(db_session, embedder=embedder)
    candidate = [ConceptCandidate(name="Attention Mechanism")]

    await service.canonicalize(user.user_id, first_paper.paper_id, candidate)
    await service.canonicalize(user.user_id, second_paper.paper_id, candidate)
    await service.canonicalize(user.user_id, second_paper.paper_id, candidate)

    concept = await db_session.scalar(
        select(Concept).where(Concept.user_id == user.user_id)
    )
    assert sorted(concept.source_paper_ids) == sorted(
        [first_paper.paper_id, second_paper.paper_id]
    )


async def test_case_variant_matches_the_existing_concept(
    db_session: AsyncSession, embedder
):
    user = await _make_user(db_session)
    paper = await _make_paper(db_session)
    service = ConceptService(db_session, embedder=embedder)

    await service.canonicalize(
        user.user_id, paper.paper_id, [ConceptCandidate(name="Attention Mechanism")]
    )
    result = await service.canonicalize(
        user.user_id, paper.paper_id, [ConceptCandidate(name="attention mechanism")]
    )

    assert result.created == []
    total = await db_session.scalar(
        select(func.count()).select_from(Concept).where(Concept.user_id == user.user_id)
    )
    assert total == 1


async def test_alias_widens_rather_than_replacing(db_session: AsyncSession, embedder):
    user = await _make_user(db_session)
    paper = await _make_paper(db_session)
    service = ConceptService(db_session, embedder=embedder)

    await service.canonicalize(
        user.user_id,
        paper.paper_id,
        [ConceptCandidate(name="Evidence Lower Bound", aliases=["ELBO"])],
    )
    await service.canonicalize(
        user.user_id,
        paper.paper_id,
        [
            ConceptCandidate(
                name="Evidence Lower Bound", aliases=["variational lower bound"]
            )
        ],
    )

    concept = await db_session.scalar(
        select(Concept).where(Concept.user_id == user.user_id)
    )
    assert set(concept.aliases) == {"ELBO", "variational lower bound"}


async def test_concepts_are_scoped_per_user(db_session: AsyncSession, embedder):
    """Two readers of the same paper get separate learner models."""
    first = await _make_user(db_session)
    second = await _make_user(db_session)
    paper = await _make_paper(db_session)
    service = ConceptService(db_session, embedder=embedder)
    candidate = [ConceptCandidate(name="Attention Mechanism")]

    a = await service.canonicalize(first.user_id, paper.paper_id, candidate)
    b = await service.canonicalize(second.user_id, paper.paper_id, candidate)

    assert a.created and b.created
    assert a.created[0] != b.created[0]


async def test_prerequisite_edges_are_directed_and_typed(
    db_session: AsyncSession, embedder
):
    user = await _make_user(db_session)
    paper = await _make_paper(db_session)
    service = ConceptService(db_session, embedder=embedder)

    await service.canonicalize(
        user.user_id,
        paper.paper_id,
        [
            ConceptCandidate(name="Attention", prerequisites=["Softmax"]),
            ConceptCandidate(name="Softmax"),
        ],
    )

    edge = await db_session.scalar(
        select(ConceptRelationship).where(ConceptRelationship.user_id == user.user_id)
    )
    assert edge.relationship_type == "prerequisite_of"
    assert edge.discovery_method == "model"

    source = await db_session.get(Concept, edge.source_concept_id)
    target = await db_session.get(Concept, edge.target_concept_id)
    assert source.canonical_name == "Softmax"
    assert target.canonical_name == "Attention"


async def test_empty_candidate_list_is_a_no_op(db_session: AsyncSession, embedder):
    user = await _make_user(db_session)
    paper = await _make_paper(db_session)
    service = ConceptService(db_session, embedder=embedder)

    assert (await service.canonicalize(user.user_id, paper.paper_id, [])).total == 0


# --------------------------------------------------------------------------
# Pipeline integration
# --------------------------------------------------------------------------


async def test_ingestion_stores_candidates_and_links_the_uploader(
    db_session: AsyncSession, storage, embedder
):
    user = await _make_user(db_session)
    content_hash = uuid.uuid4().hex + uuid.uuid4().hex[:32]
    paper = Paper(
        content_hash=content_hash,
        storage_uri=storage.put(build_pdf(PAPER_PAGES), content_hash=content_hash),
        processing_status="queued",
    )
    db_session.add(paper)
    await db_session.flush()

    result = await ingest_paper(
        db_session,
        paper.paper_id,
        storage=storage,
        embedder=embedder,
        analyzer=HeuristicAnalyzer(),
        user_id=user.user_id,
    )

    assert paper.concept_candidates is not None
    assert paper.concept_candidates["concepts"], "analysis produced no candidates"
    assert result.concepts_linked > 0

    owned = await db_session.scalar(
        select(func.count()).select_from(Concept).where(Concept.user_id == user.user_id)
    )
    assert owned == result.concepts_linked


async def test_analysis_failure_does_not_fail_a_searchable_paper(
    db_session: AsyncSession, storage, embedder
):
    """Concepts enrich; they do not gate retrieval."""

    class BrokenAnalyzer(HeuristicAnalyzer):
        def analyze(self, title_hint, sections):
            raise RuntimeError("gemini is down")

    user = await _make_user(db_session)
    content_hash = uuid.uuid4().hex + uuid.uuid4().hex[:32]
    paper = Paper(
        content_hash=content_hash,
        storage_uri=storage.put(build_pdf(PAPER_PAGES), content_hash=content_hash),
        processing_status="queued",
    )
    db_session.add(paper)
    await db_session.flush()

    result = await ingest_paper(
        db_session,
        paper.paper_id,
        storage=storage,
        embedder=embedder,
        analyzer=BrokenAnalyzer(),
        user_id=user.user_id,
    )

    assert paper.processing_status == "ready"
    assert paper.concept_candidates is None
    assert result.concepts_linked == 0


async def test_second_reader_canonicalizes_without_reingesting(
    db_session: AsyncSession, storage, embedder
):
    """Phases 1-5 are shared; only phase 6b runs for the second reader."""
    first = await _make_user(db_session)
    second = await _make_user(db_session)

    content_hash = uuid.uuid4().hex + uuid.uuid4().hex[:32]
    paper = Paper(
        content_hash=content_hash,
        storage_uri=storage.put(build_pdf(PAPER_PAGES), content_hash=content_hash),
        processing_status="queued",
    )
    db_session.add(paper)
    await db_session.flush()

    await ingest_paper(
        db_session,
        paper.paper_id,
        storage=storage,
        embedder=embedder,
        analyzer=HeuristicAnalyzer(),
        user_id=first.user_id,
    )

    linked = await canonicalize_existing_paper(
        db_session, paper.paper_id, second.user_id, embedder=embedder
    )
    assert linked > 0

    for user in (first, second):
        owned = await db_session.scalar(
            select(func.count())
            .select_from(Concept)
            .where(Concept.user_id == user.user_id)
        )
        assert owned == linked


# --------------------------------------------------------------------------
# Adjudication band (ARCHITECTURE 16.3 step 4)
# --------------------------------------------------------------------------


class _FixedEmbedder(HashingEmbedder):
    """Every text embeds identically, so ANN similarity is always 1.0.

    That forces every candidate into the adjudication band, which is what
    these tests are about.
    """

    def _embed_one(self, text: str) -> list[float]:
        return [1.0] + [0.0] * 767


class _ScriptedAdjudicator:
    """Returns a fixed verdict, and records that it was consulted."""

    def __init__(self, verdict: str, confidence: float, rel: str | None = None) -> None:
        self._adjudication = Adjudication(
            verdict=verdict, confidence=confidence, relationship_type=rel
        )
        self.batches: list[int] = []
        self.calls: list[tuple[str, str]] = []

    @property
    def model_name(self) -> str:
        return "scripted"

    def adjudicate_batch(self, pairs):
        self.batches.append(len(pairs))
        self.calls.extend((p.candidate_name, p.existing_name) for p in pairs)
        return [self._adjudication for _ in pairs]


async def _two_rounds(session, user, paper, adjudicator):
    service = ConceptService(
        session, embedder=_FixedEmbedder(), adjudicator=adjudicator
    )
    await service.canonicalize(
        user.user_id, paper.paper_id, [ConceptCandidate(name="Variational Inference")]
    )
    result = await service.canonicalize(
        user.user_id, paper.paper_id, [ConceptCandidate(name="Variational Autoencoder")]
    )
    return result


async def test_similar_names_are_adjudicated_not_auto_merged(
    db_session: AsyncSession,
):
    """There is no "similar enough to merge without asking" band.

    Measured with real embeddings, `variational inference` and `variational
    autoencoder` score 0.9263 while `evidence lower bound` and `ELBO` score
    0.8595 — the pair that must stay separate scores *higher* than the pair
    that must merge. Any auto-merge threshold collapses adjacent concepts.
    """
    user = await _make_user(db_session)
    paper = await _make_paper(db_session)
    adjudicator = _ScriptedAdjudicator("related", 0.95, "component_of")

    await _two_rounds(db_session, user, paper, adjudicator)

    assert adjudicator.calls, "the model must be consulted above the ANN floor"


async def test_related_verdict_keeps_concepts_separate_and_adds_an_edge(
    db_session: AsyncSession,
):
    user = await _make_user(db_session)
    paper = await _make_paper(db_session)

    result = await _two_rounds(
        db_session, user, paper, _ScriptedAdjudicator("related", 0.95, "component_of")
    )

    assert len(result.created) == 1, "a related concept is a new node, not a merge"
    assert result.relationships_created == 1

    total = await db_session.scalar(
        select(func.count()).select_from(Concept).where(Concept.user_id == user.user_id)
    )
    assert total == 2

    edge = await db_session.scalar(
        select(ConceptRelationship).where(ConceptRelationship.user_id == user.user_id)
    )
    assert edge.relationship_type == "component_of"


async def test_same_verdict_merges(db_session: AsyncSession):
    user = await _make_user(db_session)
    paper = await _make_paper(db_session)

    result = await _two_rounds(
        db_session, user, paper, _ScriptedAdjudicator("same", 0.99)
    )

    assert result.created == []
    total = await db_session.scalar(
        select(func.count()).select_from(Concept).where(Concept.user_id == user.user_id)
    )
    assert total == 1


async def test_low_confidence_same_does_not_merge(db_session: AsyncSession):
    """A merge is destructive, so it needs conviction, not a bare majority."""
    user = await _make_user(db_session)
    paper = await _make_paper(db_session)

    result = await _two_rounds(
        db_session, user, paper, _ScriptedAdjudicator("same", 0.5)
    )

    assert len(result.created) == 1
    total = await db_session.scalar(
        select(func.count()).select_from(Concept).where(Concept.user_id == user.user_id)
    )
    assert total == 2


async def test_distinct_verdict_creates_no_edge(db_session: AsyncSession):
    user = await _make_user(db_session)
    paper = await _make_paper(db_session)

    result = await _two_rounds(
        db_session, user, paper, _ScriptedAdjudicator("distinct", 0.9)
    )

    assert len(result.created) == 1
    assert result.relationships_created == 0


async def test_offline_default_never_merges(db_session: AsyncSession):
    """With no model configured, two names are two concepts."""
    user = await _make_user(db_session)
    paper = await _make_paper(db_session)

    result = await _two_rounds(db_session, user, paper, ConservativeAdjudicator())

    assert len(result.created) == 1
    assert result.relationships_created == 0


async def test_a_paper_costs_one_adjudication_call(db_session: AsyncSession):
    """Per-concept calls exhaust a day's free-tier quota on a single paper.

    Measured against the live API: gemini-3.5-flash free tier allows 20
    generate_content requests per day, and a paper yields up to 20 concepts.
    One call per paper is what makes ingestion affordable at all.
    """
    user = await _make_user(db_session)
    paper = await _make_paper(db_session)
    adjudicator = _ScriptedAdjudicator("related", 0.95, "component_of")
    service = ConceptService(
        db_session, embedder=_FixedEmbedder(), adjudicator=adjudicator
    )

    await service.canonicalize(
        user.user_id, paper.paper_id, [ConceptCandidate(name="Seed Concept")]
    )
    await service.canonicalize(
        user.user_id,
        paper.paper_id,
        [ConceptCandidate(name=f"Concept {n}") for n in range(6)],
    )

    # The seed round has nothing to compare against, so it costs no call at
    # all — that is §16.3's zero-model-cost path. The second round compares
    # six candidates and spends exactly one request on all of them.
    assert adjudicator.batches == [6]


async def test_adjudication_failure_degrades_to_distinct(db_session: AsyncSession):
    """A model outage must not fail a paper that is otherwise searchable."""

    class _Broken:
        model_name = "broken"

        def adjudicate_batch(self, pairs):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    user = await _make_user(db_session)
    paper = await _make_paper(db_session)

    result = await _two_rounds(db_session, user, paper, _Broken())

    assert len(result.created) == 1, "the concept is still created, just unmerged"
    assert result.relationships_created == 0


# --------------------------------------------------------------------------
# Cross-paper precondition (ARCHITECTURE 12, step 0)
# --------------------------------------------------------------------------


async def test_cross_paper_edge_exists_before_any_question(db_session: AsyncSession):
    """The callback's precondition: the edge is written at ingest, not on demand.

    Paper A is canonicalized first, then Paper B. A concept that appears only
    in B must end up connected to a concept from A — otherwise the 1-hop
    expansion in §12 step 3 has nothing to find and the callback cannot fire.
    """
    user = await _make_user(db_session)
    paper_a = await _make_paper(db_session)
    paper_b = await _make_paper(db_session)
    adjudicator = _ScriptedAdjudicator("related", 0.90, "component_of")
    service = ConceptService(
        db_session, embedder=_FixedEmbedder(), adjudicator=adjudicator
    )

    await service.canonicalize(
        user.user_id, paper_a.paper_id, [ConceptCandidate(name="Variational Lower Bound")]
    )
    result = await service.canonicalize(
        user.user_id, paper_b.paper_id, [ConceptCandidate(name="Variance Schedule")]
    )

    assert result.relationships_created == 1

    edge = await db_session.scalar(
        select(ConceptRelationship).where(ConceptRelationship.user_id == user.user_id)
    )
    assert edge.relationship_type == "component_of"

    source = await db_session.get(Concept, edge.source_concept_id)
    target = await db_session.get(Concept, edge.target_concept_id)

    # The edge genuinely spans the two papers.
    assert paper_b.paper_id in source.source_paper_ids
    assert paper_a.paper_id in target.source_paper_ids
    assert paper_a.paper_id not in source.source_paper_ids


async def test_merged_concept_carries_both_papers(db_session: AsyncSession):
    """The other way the two papers connect: one shared concept.

    When B's name for a concept merges into A's, the surviving node lists both
    papers — which is what lets §12 step 7 expand scope to the prior paper.
    """
    user = await _make_user(db_session)
    paper_a = await _make_paper(db_session)
    paper_b = await _make_paper(db_session)
    service = ConceptService(
        db_session,
        embedder=_FixedEmbedder(),
        adjudicator=_ScriptedAdjudicator("same", 0.95),
    )

    await service.canonicalize(
        user.user_id, paper_a.paper_id, [ConceptCandidate(name="Variational Lower Bound")]
    )
    await service.canonicalize(
        user.user_id, paper_b.paper_id, [ConceptCandidate(name="Variational Bound Objective")]
    )

    concept = await db_session.scalar(
        select(Concept).where(Concept.user_id == user.user_id)
    )
    assert sorted(concept.source_paper_ids) == sorted([paper_a.paper_id, paper_b.paper_id])
