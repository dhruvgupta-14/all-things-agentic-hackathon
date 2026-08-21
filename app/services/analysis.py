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
# Prompt
# --------------------------------------------------------------------------

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
    """The one structural call per paper, on Vertex AI."""

    def __init__(
        self,
        *,
        project: str,
        location: str,
        model: str | None = None,
    ) -> None:
        self._model = model or GEMINI_MODEL
        self._project = project
        self._location = location
        self._client = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def transport(self) -> str:
        return "vertex"

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(
                vertexai=True, project=self._project, location=self._location
            )
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
    """The application's analyzer. One backend, no branch.

    Called through the module rather than imported by name, so the test
    harness has one place to substitute a deterministic fake.
    """
    settings = get_settings()
    return GeminiAnalyzer(
        project=settings.vertex_project,
        location=settings.vertex_location,
        model=settings.gemini_model,
    )
