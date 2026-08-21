"""Concept identity adjudication — §16.3 step 4.

The ambiguous band only. Steps 1–3 of canonicalization resolve the common case
deterministically at zero model cost; this is the narrow slice where embedding
similarity genuinely cannot decide.

The failure this exists to prevent is specific and predictable: embeddings put
*variational inference* and *variational autoencoder* very close together.
Those are strongly **related** and definitely **not the same**, and cosine
similarity cannot tell "same" from "adjacent". Only a model can, and only here
— comparing every pair would be slow, expensive, and unnecessary.

Verdict vocabulary is closed and the output is schema-validated. The model
proposes; `ConceptService` commits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.db.models import RELATIONSHIP_TYPE

logger = logging.getLogger(__name__)

Verdict = Literal["same", "related", "distinct"]


class AdjudicationUnavailable(Exception):
    """The adjudicator could not decide. Callers fall back to `distinct`."""


class Adjudication(BaseModel):
    """A closed-vocabulary judgement about two concept names."""

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    # Only meaningful when verdict == "related". Validated against the closed
    # set before it can become an edge.
    relationship_type: str | None = None
    reason: str | None = Field(default=None, max_length=300)

    def typed_relationship(self) -> str | None:
        if self.verdict != "related":
            return None
        if self.relationship_type in RELATIONSHIP_TYPE:
            return self.relationship_type
        # An unrecognised type is not an error worth failing on — it degrades
        # to the honest weak fallback rather than inventing a stronger claim.
        return "co_occurs_with"


@dataclass(slots=True)
class ConceptPair:
    """One ambiguous-band comparison awaiting a verdict."""

    candidate_name: str
    existing_name: str
    candidate_description: str | None = None
    existing_description: str | None = None


class BatchAdjudication(BaseModel):
    """Verdicts for a batch, positionally aligned with the input pairs."""

    verdicts: list[Adjudication]


class Adjudicator(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def adjudicate_batch(self, pairs: list[ConceptPair]) -> list[Adjudication]:
        """Judge every pair in one call.

        Ingesting a paper produces up to `MAX_CANDIDATES` concepts, and asking
        about each separately costs one request each — which exhausts a modest
        quota on a single paper. The judgement is per-pair either way; only
        the number of round trips differs.
        """
        ...


_PROMPT = """You are deciding whether two technical concept names, taken from
research papers one reader has read, refer to the same idea.

Answer with exactly one verdict:

- "same": different surface forms of one idea. Abbreviations, expansions and
  notational variants are the same idea — "ELBO" and "evidence lower bound"
  are the same.
- "related": genuinely different ideas with a real connection. A method and
  the model it is used to train are related, not the same — "variational
  inference" and "variational autoencoder" are related, not the same. If the
  verdict is "related", give the relationship_type from exactly this set:
  {relationship_types}
- "distinct": no useful relationship.

Prefer "related" over "same" when uncertain. Merging two different ideas
destroys information; keeping two names for one idea is a minor untidiness.

Return one verdict per numbered pair, in the same order, and exactly as many
verdicts as there are pairs.

PAIRS
{pairs}
"""


class GeminiAdjudicator:
    """The concept-identity judgment, on Vertex AI."""

    def __init__(
        self,
        *,
        model: str,
        project: str,
        location: str,
    ) -> None:
        self._model = model
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
        # Shared per process: building one costs a ~12s credential and TLS
        # handshake on its first request (see app/services/genai_client.py).
        if self._client is None:
            from app.services.genai_client import get_genai_client

            self._client = get_genai_client(
                project=self._project,
                location=self._location,
            )
        return self._client

    @retry(
        retry=retry_if_exception_type(AdjudicationUnavailable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True,
    )
    def adjudicate_batch(self, pairs: list[ConceptPair]) -> list[Adjudication]:
        from google.genai import types

        if not pairs:
            return []

        rendered = "\n\n".join(
            f"{index}. A: {pair.candidate_name}"
            f"\n   A means: {pair.candidate_description or '(no description)'}"
            f"\n   B: {pair.existing_name}"
            f"\n   B means: {pair.existing_description or '(no description)'}"
            for index, pair in enumerate(pairs, start=1)
        )
        prompt = _PROMPT.format(
            relationship_types=", ".join(RELATIONSHIP_TYPE), pairs=rendered
        )

        try:
            response = self._get_client().models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchAdjudication,
                    # Identity decisions must not vary run to run: the same
                    # pair has to resolve the same way every time, or the
                    # graph is not reproducible.
                    temperature=0.0,
                ),
            )
        except Exception as exc:
            raise AdjudicationUnavailable(f"{type(exc).__name__}: {exc}") from exc

        parsed = response.parsed
        if parsed is None:
            raise AdjudicationUnavailable("model returned no parseable verdict")

        batch = BatchAdjudication.model_validate(parsed, from_attributes=True)
        if len(batch.verdicts) != len(pairs):
            # A misaligned batch would attach verdicts to the wrong pairs and
            # merge the wrong concepts. Refuse rather than guess the mapping.
            raise AdjudicationUnavailable(
                f"asked about {len(pairs)} pairs, received {len(batch.verdicts)} verdicts"
            )
        return batch.verdicts


def get_adjudicator() -> Adjudicator:
    """The application's adjudicator. One backend, no branch.

    The old fallback never merged anything, so an unconfigured deployment
    canonicalized every concept into a duplicate — the cross-paper edges the
    callback depends on were simply never written, silently.

    Called through the module rather than imported by name, so the test
    harness has one place to substitute a deterministic fake.
    """
    settings = get_settings()
    return GeminiAdjudicator(
        model=settings.gemini_model,
        project=settings.vertex_project,
        location=settings.vertex_location,
    )
