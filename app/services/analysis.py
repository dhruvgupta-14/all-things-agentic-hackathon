"""Phase 6a — structural analysis of a paper.

One model call per paper, producing the metadata and the concept candidates
that canonicalization later folds into each reader's own graph. The output is
paper-scoped and shared: two users uploading the same PDF pay for this once
(ARCHITECTURE 8.4).

The result is schema-validated before it is stored. Model output is untrusted
input — a name that arrives 4 000 characters long, or a prerequisite pointing
at a concept that was never proposed, is dropped here rather than becoming a
malformed row.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Protocol

from pydantic import BaseModel, Field, field_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings

logger = logging.getLogger(__name__)

# Pinned via settings; see Settings.gemini_model for why not an alias.
GEMINI_MODEL = "gemini-3.5-flash"
STUB_ANALYZER_NAME = "local-heuristic-v1"

MAX_CANDIDATES = 20


class AnalysisUnavailable(Exception):
    """The analyzer could not produce a result. Treated as transient."""


class ConceptCandidate(BaseModel):
    name: str = Field(max_length=200)
    aliases: list[str] = Field(default_factory=list, max_length=10)
    description: str | None = Field(default=None, max_length=2000)
    # Names of other candidates this one depends on. Resolved against the
    # candidate set after validation; dangling references are dropped.
    prerequisites: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip()
        if not cleaned:
            raise ValueError("empty concept name")
        return cleaned

    @field_validator("aliases")
    @classmethod
    def _clean_aliases(cls, values: list[str]) -> list[str]:
        seen: list[str] = []
        for value in values:
            cleaned = re.sub(r"\s+", " ", value).strip()[:200]
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        return seen


class PaperAnalysis(BaseModel):
    title: str | None = Field(default=None, max_length=1000)
    authors: list[str] = Field(default_factory=list, max_length=50)
    year: int | None = None
    concepts: list[ConceptCandidate] = Field(default_factory=list)

    @field_validator("year")
    @classmethod
    def _sane_year(cls, value: int | None) -> int | None:
        # The schema CHECK would reject anything outside this; fail soft here
        # rather than losing the whole analysis to one hallucinated digit.
        return value if value and 1900 <= value <= 2100 else None

    def resolved(self) -> PaperAnalysis:
        """Drop prerequisite references that name no proposed concept."""
        known = {concept.name for concept in self.concepts}
        for concept in self.concepts:
            concept.prerequisites = [
                name for name in concept.prerequisites if name in known and name != concept.name
            ]
        return self


class Analyzer(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def analyze(self, title_hint: str | None, sections: list[tuple[str, str]]) -> PaperAnalysis:
        """`sections` is a list of (section_role, text)."""
        ...


# --------------------------------------------------------------------------
# Local stub
# --------------------------------------------------------------------------

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
    """Deterministic local analyzer.

    Extracts repeated multi-word technical phrases. This is not semantic
    understanding — it is frequency counting — but it is stable, free, and
    exercises every downstream code path, which is what local development
    needs from it.
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


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------

_PROMPT = """You are analysing a research paper for a reading companion.

Extract:
- the paper's title, authors and publication year, if stated
- the key technical concepts a reader must understand, at most {limit}

For each concept give the clearest full name, any aliases or abbreviations the
paper uses for it, a one-sentence description, and which other concepts in your
own list are prerequisites for understanding it.

Only report concepts this paper actually discusses. Do not invent
prerequisites, and only reference concept names that appear in your own list.

PAPER
{body}
"""


class GeminiAnalyzer:
    """Reachable via an AI Studio API key or Vertex AI — the same model."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
        model: str | None = None,
    ) -> None:
        if not api_key and not project:
            raise ValueError("GeminiAnalyzer needs either an api_key or a project")
        self._model = model or GEMINI_MODEL
        self._api_key = api_key
        self._project = project
        self._location = location
        self._client = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def transport(self) -> str:
        return "vertex" if self._project else "ai-studio"

    def _get_client(self):
        if self._client is None:
            from google import genai

            if self._project:
                self._client = genai.Client(
                    vertexai=True, project=self._project, location=self._location
                )
            else:
                self._client = genai.Client(api_key=self._api_key)
        return self._client

    @retry(
        retry=retry_if_exception_type(AnalysisUnavailable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def analyze(
        self, title_hint: str | None, sections: list[tuple[str, str]]
    ) -> PaperAnalysis:
        from google.genai import types

        # Concepts are introduced early; sending the whole paper mostly buys
        # reference lists.
        body = "\n\n".join(
            f"## {role}\n{text[:4000]}"
            for role, text in sections
            if role != "references"
        )[:60000]

        try:
            response = self._get_client().models.generate_content(
                model=self._model,
                contents=_PROMPT.format(limit=MAX_CANDIDATES, body=body),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PaperAnalysis,
                    temperature=0.0,
                ),
            )
        except Exception as exc:
            raise AnalysisUnavailable(str(exc)) from exc

        parsed = response.parsed
        if parsed is None:
            raise AnalysisUnavailable("model returned no parseable analysis")

        analysis = PaperAnalysis.model_validate(parsed, from_attributes=True)
        if title_hint and not analysis.title:
            analysis.title = title_hint
        analysis.concepts = analysis.concepts[:MAX_CANDIDATES]
        return analysis.resolved()


def get_analyzer() -> Analyzer:
    settings = get_settings()
    # Vertex first, for the same reason as the embedder.
    if settings.vertex_project:
        return GeminiAnalyzer(
            project=settings.vertex_project,
            location=settings.vertex_location,
            model=settings.gemini_model,
        )
    if settings.gemini_api_key:
        return GeminiAnalyzer(
            api_key=settings.gemini_api_key, model=settings.gemini_model
        )
    logger.debug("using the local heuristic analyzer; concepts will be lexical")
    return HeuristicAnalyzer()
