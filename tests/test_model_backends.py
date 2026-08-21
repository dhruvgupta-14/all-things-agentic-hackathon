"""The Gemini-backed services: one backend, and the batching around it.

This module used to be about *selection* — an AI Studio API key for
development, Vertex AI for deployment, and which won when both were set. There
is nothing to select any more. The API-key transport is gone and so is the
stub fallback each factory used to return when nothing was configured, because
that fallback made an unconfigured deployment look healthy while embedding with
a hashing trick and answering from canned text.

What is left is worth pinning: that the factories build a Vertex client and
never anything else, that construction stays lazy, and that batching respects
the API's limits.

No network calls anywhere: clients are constructed lazily, so every rule here
is testable without credentials or quota.
"""

import pytest

from app.services import analysis, embeddings
from app.services.analysis import GeminiAnalyzer
from app.services.embeddings import EMBED_BATCH_SIZE, GeminiEmbedder
from tests.fakes import HashingEmbedder

# Captured at import, before conftest's autouse fixture can patch them.
_REAL_EMBEDDER_FACTORY = embeddings.get_embedder
_REAL_ANALYZER_FACTORY = analysis.get_analyzer


@pytest.fixture(autouse=True)
def _real_factories(monkeypatch):
    """Undo the harness's fakes: this module is about the real factories.

    `conftest.fake_backends` patches these everywhere else, which is what keeps
    the suite off the network. Here they are the subject. Restoring them is
    safe because nothing below builds a client — construction is lazy, and
    building one is what would cost a call.
    """
    monkeypatch.setattr(embeddings, "get_embedder", _REAL_EMBEDDER_FACTORY)
    monkeypatch.setattr(analysis, "get_analyzer", _REAL_ANALYZER_FACTORY)


# --------------------------------------------------------------------------
# There is one backend
# --------------------------------------------------------------------------


def test_the_factories_build_vertex_clients(settings_env):
    settings_env(vertex_project="some-project")

    embedder = embeddings.get_embedder()
    analyzer = analysis.get_analyzer()

    assert isinstance(embedder, GeminiEmbedder)
    assert isinstance(analyzer, GeminiAnalyzer)
    assert embedder.transport == "vertex"
    assert analyzer.transport == "vertex"


def test_there_is_no_stub_fallback_left_to_reach(settings_env):
    """The regression guard.

    An unconfigured process used to get a hashing embedder here — ingesting
    papers whose vectors could never match a real query, with no error at any
    point. The fallback is gone from the application entirely; the fake now
    lives in the test harness, where it cannot be selected by accident.
    """
    assert not hasattr(embeddings, "HashingEmbedder")
    assert not hasattr(analysis, "HeuristicAnalyzer")

    settings_env(vertex_project="some-project")
    assert not isinstance(embeddings.get_embedder(), HashingEmbedder)


def test_the_project_comes_from_settings(settings_env):
    settings_env(vertex_project="a-specific-project", vertex_location="global")

    embedder = embeddings.get_embedder()

    assert embedder._project == "a-specific-project"
    assert embedder._location == "global"


# --------------------------------------------------------------------------
# Model identity
# --------------------------------------------------------------------------


def test_the_embedding_model_name_is_stable():
    """Stored on every paper, and compared against on read.

    Changing this string marks every existing paper stale and triggers a
    re-index, so it is not a detail — it is a migration trigger.
    """
    assert GeminiEmbedder(project="p", location="global").model_name == "gemini-embedding-001"


def test_a_real_embedder_differs_from_the_test_fake():
    """The fake's vectors must never be mistaken for real ones: they live in a
    different space, and a paper embedded with one cannot be searched with the
    other."""
    real = GeminiEmbedder(project="p", location="global").model_name
    assert real != HashingEmbedder().model_name


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [GeminiEmbedder, GeminiAnalyzer])
def test_construction_requires_a_project_and_location(cls):
    """Both are keyword-only and required, so a caller cannot half-configure one
    and get a client pointed somewhere it did not intend."""
    with pytest.raises(TypeError):
        cls()


def test_client_construction_is_lazy():
    """Selecting a backend must not need credentials or a network call."""
    embedder = GeminiEmbedder(project="p", location="global")
    assert embedder._client is None
    assert embedder.model_name  # reading identity does not build a client


# --------------------------------------------------------------------------
# The relevance floor
# --------------------------------------------------------------------------


def test_each_vector_space_carries_its_own_floor():
    """Cosine scores are not comparable between models.

    0.25 is right for the fake's lexical overlap and badly wrong for
    gemini-embedding-001, where genuinely unrelated text still scores ~0.55.
    This is why RETRIEVAL_MIN_SIMILARITY stays unset.
    """
    assert HashingEmbedder().default_min_similarity == 0.25
    assert GeminiEmbedder(project="p", location="global").default_min_similarity == 0.58


def test_gemini_floor_sits_above_measured_unrelated_scores():
    """Unrelated queries topped out at 0.555 against a real corpus."""
    assert GeminiEmbedder(project="p", location="global").default_min_similarity > 0.555


def test_gemini_floor_sits_below_measured_relevant_scores():
    """The worst relevant query scored 0.626; the floor must not exclude it."""
    assert GeminiEmbedder(project="p", location="global").default_min_similarity < 0.626


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------


def test_default_batch_size_is_under_the_api_cap():
    """Measured against the live API: 250 is refused, 100 is the documented cap."""
    assert 0 < EMBED_BATCH_SIZE <= 100


def test_large_input_is_split_into_windows(monkeypatch):
    """A 100-chunk paper must become several requests, not one rejected one."""
    embedder = GeminiEmbedder(project="p", location="global", batch_size=20)
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
    embedder = GeminiEmbedder(project="p", location="global", batch_size=2)

    def fake_call(texts, task_type):
        # Encode each text's identity in its vector so order is checkable.
        return [[float(int(t.split()[-1]))] * 768 for t in texts]

    monkeypatch.setattr(embedder, "_embed_call", fake_call)

    vectors = embedder.embed_batch([f"chunk {i}" for i in range(5)])
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_a_short_input_is_a_single_call(monkeypatch):
    embedder = GeminiEmbedder(project="p", location="global", batch_size=20)
    calls: list[int] = []
    monkeypatch.setattr(
        embedder,
        "_embed_call",
        lambda texts, task_type: (calls.append(len(texts)), [[0.0] * 768] * len(texts))[1],
    )

    embedder.embed_batch(["one", "two"])
    assert calls == [2]
