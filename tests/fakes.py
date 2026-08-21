"""Deterministic test doubles for the model and storage backends.

These used to live in `app/services/` as fallbacks: `get_embedder()` returned
the hashing embedder when no project was configured, `get_storage()` wrote to a
directory, and so on. That made a misconfigured deployment look healthy — it
ingested papers and answered questions, with nothing real behind it.

They are still useful, just not as production code. The suite substitutes them
through the module-level factories (see `conftest.py`), which is why the
application calls `embeddings.get_embedder()` rather than importing the name.

The hashing embedder is a hashing-trick embedding, not random noise: tokens map
to fixed dimensions, so texts sharing vocabulary genuinely score closer
together. That makes retrieval assertions meaningful offline. It is emphatically
**not** semantic — it cannot match "car" to "automobile" — so a relevance floor
tuned against it does not transfer to `gemini-embedding-001`.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from app.db.models import EMBEDDING_DIM
from app.services.adjudication import Adjudication, ConceptPair
from app.services.analysis import MAX_CANDIDATES, ConceptCandidate, PaperAnalysis
from app.services.quizzes import AuthoredQuiz, Grading
from app.services.storage import ObjectNotFoundError

STUB_MODEL_NAME = "local-hashing-v1"
STUB_ANALYZER_NAME = "local-heuristic-v1"

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        # A zero vector makes cosine distance undefined. Anchor empty text to
        # one fixed direction so it is merely irrelevant, never NaN.
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


class HashingEmbedder:
    """Deterministic embedder. Same text always yields the same vector."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return STUB_MODEL_NAME

    @property
    def default_min_similarity(self) -> float:
        # Lexical overlap scores lower and flatter than semantic similarity.
        return 0.25

    @property
    def transport(self) -> str:
        return "fake"

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in _tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self._dim
            # A sign bit keeps unrelated collisions from always reinforcing.
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        return _normalise(vector)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class InMemoryStorage:
    """Object storage in a dict.

    Replaces the filesystem backend the application used to carry. Tests never
    needed the disk — they needed a place to put bytes and get them back.
    """

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def put(self, data: bytes, *, content_hash: str) -> str:
        uri = f"memory://{content_hash}.pdf"
        self._objects[uri] = data
        return uri

    def get(self, uri: str) -> bytes:
        try:
            return self._objects[uri]
        except KeyError:
            raise ObjectNotFoundError(uri) from None


_STOPWORDS = frozenset(
    """
    a about above after again all also an and any approach are as at
    based be been before being below between both but by can could did
    do does down during each few for from further had has have he her
    here his how i if in into is it its just may me method might model
    models more most must my no not of off on once only or other our out
    over own paper results same she should show shows so some such than
    that the their then there these they this those to too under up use
    used using very was we were what when where which while who whom why
    with without would you your
    """.split()
)

_PHRASE = re.compile(r"\b([A-Za-z][A-Za-z0-9-]{2,})\b")


class HeuristicAnalyzer:
    """Deterministic analyzer.

    Extracts repeated multi-word technical phrases. This is not semantic
    understanding — it is frequency counting — but it is stable, free, and
    exercises every downstream code path, which is what the suite needs.
    """

    @property
    def model_name(self) -> str:
        return STUB_ANALYZER_NAME

    def analyze(
        self, title_hint: str | None, sections: list[tuple[str, str]]
    ) -> PaperAnalysis:
        # Weight the sections that actually introduce concepts.
        weights = {"abstract": 3, "introduction": 2, "method": 3, "results": 1}
        counts: Counter[str] = Counter()

        for role, body in sections:
            weight = weights.get(role, 1)
            if role == "references":
                continue
            words = [w for w in _PHRASE.findall(body)]
            lowered = [w.lower() for w in words]

            for size in (2, 3):
                for index in range(len(words) - size + 1):
                    window = lowered[index : index + size]
                    if any(word in _STOPWORDS or len(word) < 3 for word in window):
                        continue
                    counts[" ".join(window)] += weight

        candidates: list[ConceptCandidate] = []
        for phrase, score in counts.most_common(MAX_CANDIDATES * 3):
            if score < 2:
                continue
            # Skip a phrase already contained in one we accepted, so
            # "dot product attention" does not also yield "product attention".
            if any(phrase in accepted.name for accepted in candidates):
                continue
            candidates.append(ConceptCandidate(name=phrase))
            if len(candidates) >= MAX_CANDIDATES:
                break

        return PaperAnalysis(
            title=title_hint, concepts=candidates, authors=[], year=None
        ).resolved()


class ConservativeAdjudicator:
    """Never merges.

    A wrong merge silently corrupts the graph and is awkward to unpick; a
    duplicate is untidy and mergeable later through `merged_into_id`. For a
    test double, "these are different concepts" is the answer that asserts
    nothing it has not been told.
    """

    @property
    def model_name(self) -> str:
        return "conservative-no-merge"

    def adjudicate_batch(self, pairs: list[ConceptPair]) -> list[Adjudication]:
        return [
            Adjudication(
                verdict="distinct",
                confidence=0.0,
                reason="no adjudicator configured; defaulting to distinct",
            )
            for _ in pairs
        ]


class StubQuizAuthor:
    """Deterministic. Keeps the suite from spending quota."""

    model_name = "stub-quiz-author"

    def write(
        self, *, concept: str, difficulty: str, passages: list[str]
    ) -> AuthoredQuiz:
        return AuthoredQuiz(
            question=f"In your own words, what is {concept} and why does it matter here?",
            must_mention=[concept, "why it matters"],
        )


class StubGrader:
    """Lexical overlap, not judgement. Enough to exercise every branch."""

    model_name = "stub-grader"

    def grade(self, *, question: str, rubric: list[str], answer: str) -> Grading:
        lowered = (answer or "").lower()
        missing = [point for point in rubric if point.lower() not in lowered]
        if not rubric:
            grade = "partial"
        elif not missing:
            grade = "correct"
        elif len(missing) == len(rubric):
            grade = "incorrect"
        else:
            grade = "partial"
        return Grading(grade=grade, missing_elements=missing, confidence=0.6)
