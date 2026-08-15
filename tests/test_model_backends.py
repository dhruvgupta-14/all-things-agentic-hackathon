"""Backend selection for the Gemini-backed services.

Two transports reach the same models: an AI Studio API key (development) and
Vertex AI (deployment). The selection rules and — more importantly — the model
identity across transports are what these pin down.

No network calls anywhere: clients are constructed lazily, so every rule here
is testable without credentials or quota.
"""

import pytest

from app.services.analysis import GeminiAnalyzer, HeuristicAnalyzer, get_analyzer
from app.services.embeddings import (
    EMBED_BATCH_SIZE,
    GeminiEmbedder,
    HashingEmbedder,
    get_embedder,
)


@pytest.fixture(autouse=True)
def _no_ambient_credentials(settings_env):
    """Start from "nothing configured", whatever the real .env holds."""
    settings_env(gemini_api_key=None, vertex_project=None)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def test_no_credentials_selects_the_local_stubs():
    assert isinstance(get_embedder(), HashingEmbedder)
    assert isinstance(get_analyzer(), HeuristicAnalyzer)


def test_api_key_selects_gemini_over_the_stub(settings_env):
    settings_env(gemini_api_key="test-key")

    embedder = get_embedder()
    analyzer = get_analyzer()

    assert isinstance(embedder, GeminiEmbedder)
    assert isinstance(analyzer, GeminiAnalyzer)
    assert embedder.transport == "ai-studio"
    assert analyzer.transport == "ai-studio"


def test_vertex_project_selects_vertex(settings_env):
    settings_env(vertex_project="some-project")

    assert get_embedder().transport == "vertex"
    assert get_analyzer().transport == "vertex"


def test_vertex_wins_when_both_are_configured(settings_env):
    """A deployment given a project must not fall back to a personal API key."""
    settings_env(gemini_api_key="test-key", vertex_project="some-project")

    assert get_embedder().transport == "vertex"
    assert get_analyzer().transport == "vertex"


# --------------------------------------------------------------------------
# Model identity
# --------------------------------------------------------------------------


def test_transport_does_not_change_the_embedding_model_name():
    """The property that makes the AI Studio -> Vertex switch free.

    Both transports serve the same `gemini-embedding-001`, so the vectors are
    interchangeable. If `model_name` differed by transport, every paper would
    be flagged stale on the switch and re-embedded for no reason.
    """
    by_key = GeminiEmbedder(api_key="test-key")
    by_vertex = GeminiEmbedder(project="p", location="us-central1")

    assert by_key.model_name == by_vertex.model_name == "gemini-embedding-001"
    assert by_key.transport != by_vertex.transport


def test_transport_does_not_change_the_analyzer_model_name():
    assert (
        GeminiAnalyzer(api_key="k").model_name
        == GeminiAnalyzer(project="p", location="l").model_name
    )


def test_switching_transport_does_not_make_papers_stale(settings_env):
    """Stated as the reindex staleness check sees it: same name, nothing to do."""
    settings_env(gemini_api_key="test-key")
    with_key = get_embedder().model_name

    settings_env(vertex_project="some-project")
    with_vertex = get_embedder().model_name

    assert with_key == with_vertex


def test_gemini_differs_from_the_local_stub():
    """Switching off the stub *must* invalidate its vectors."""
    assert GeminiEmbedder(api_key="k").model_name != HashingEmbedder().model_name


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [GeminiEmbedder, GeminiAnalyzer])
def test_construction_requires_a_transport(cls):
    """Neither credential given is a configuration error, not a silent default."""
    with pytest.raises(ValueError, match="api_key or a project"):
        cls()


def test_client_construction_is_lazy():
    """Selecting a backend must not need credentials or a network call."""
    embedder = GeminiEmbedder(api_key="definitely-not-a-real-key")
    assert embedder._client is None
    assert embedder.model_name  # reading identity does not build a client


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------


def test_each_vector_space_carries_its_own_floor():
    """Cosine scores are not comparable between models.

    0.25 is right for lexical overlap and badly wrong for gemini-embedding-001,
    where genuinely unrelated text still scores ~0.55.
    """
    assert HashingEmbedder().default_min_similarity == 0.25
    assert GeminiEmbedder(api_key="k").default_min_similarity == 0.58


def test_gemini_floor_sits_above_measured_unrelated_scores():
    """Unrelated queries topped out at 0.555 against a real corpus."""
    assert GeminiEmbedder(api_key="k").default_min_similarity > 0.555


def test_gemini_floor_sits_below_measured_relevant_scores():
    """The worst relevant query scored 0.626; the floor must not exclude it."""
    assert GeminiEmbedder(api_key="k").default_min_similarity < 0.626


def test_default_batch_size_is_under_the_api_cap():
    """Measured against the live API: 250 is refused, 100 is the documented cap."""
    assert 0 < EMBED_BATCH_SIZE <= 100


def test_large_input_is_split_into_windows(monkeypatch):
    """A 100-chunk paper must become several requests, not one rejected one."""
    embedder = GeminiEmbedder(api_key="k", batch_size=20)
    calls: list[int] = []

    def fake_call(texts, task_type):
        calls.append(len(texts))
        return [[0.0] * 768 for _ in texts]

    monkeypatch.setattr(embedder, "_embed_call", fake_call)

    vectors = embedder.embed_batch([f"chunk {i}" for i in range(95)])

    assert len(vectors) == 95
    assert calls == [20, 20, 20, 20, 15]


def test_batching_preserves_input_order(monkeypatch):
    """Vectors are zipped back onto chunks positionally; order is load-bearing."""
    embedder = GeminiEmbedder(api_key="k", batch_size=2)

    def fake_call(texts, task_type):
        # Encode each text's identity in its vector so order is checkable.
        return [[float(int(t.split()[-1]))] * 768 for t in texts]

    monkeypatch.setattr(embedder, "_embed_call", fake_call)

    vectors = embedder.embed_batch([f"chunk {i}" for i in range(5)])
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_a_short_input_is_a_single_call(monkeypatch):
    embedder = GeminiEmbedder(api_key="k", batch_size=20)
    calls: list[int] = []
    monkeypatch.setattr(
        embedder,
        "_embed_call",
        lambda texts, task_type: (calls.append(len(texts)), [[0.0] * 768] * len(texts))[1],
    )

    embedder.embed_batch(["one", "two"])
    assert calls == [2]
