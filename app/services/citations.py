"""Deterministic citation verification (ARCHITECTURE 9.2 step 9).

A citation is not a claim the model makes — it is a `turn_retrievals` row that
got flagged. This module is the only thing that flags them, and it flags only
markers it can match to a passage actually retrieved during this turn.

Markers the model invented are removed from the answer before the reader sees
it. If removing them leaves the answer with no evidence at all, the turn is
downgraded to `no_evidence` rather than shipped as an ungrounded claim.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field

from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

# `[1]`, `[12]`. Deliberately narrow: bracketed numbers are what the tool hands
# the model, and anything else is not a citation.
_MARKER = re.compile(r"\[(\d{1,2})\]")


@dataclass(slots=True)
class VerifiedCitation:
    marker: str
    chunk: RetrievedChunk


@dataclass(slots=True)
class VerificationResult:
    """The verified answer, and what the turn may claim about its grounding."""

    text: str
    citations: list[VerifiedCitation] = field(default_factory=list)
    stripped_markers: list[str] = field(default_factory=list)
    grounding_status: str = "n/a"

    @property
    def cited_chunk_ids(self) -> set[uuid.UUID]:
        return {citation.chunk.chunk_id for citation in self.citations}


def _tidy(text: str) -> str:
    """Close the gaps a removed marker leaves behind."""
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +\n", "\n", text)
    return text.strip()


def verify(draft: str, retrieved: list[RetrievedChunk]) -> VerificationResult:
    """Match every marker in the draft against this turn's retrieval set.

    `retrieved` is positional: the passage handed to the model as `[1]` is
    `retrieved[0]`. That is the whole mapping — there is no other way for a
    marker to become a citation.
    """
    if not draft or not draft.strip():
        return VerificationResult(text="", grounding_status="no_evidence")

    # Nothing was retrieved, so no marker can possibly resolve.
    if not retrieved:
        cleaned = _tidy(_MARKER.sub("", draft))
        stripped = sorted(set(_MARKER.findall(draft)))
        if stripped:
            logger.warning(
                "citation markers present with an empty retrieval set",
                extra={"markers": stripped},
            )
        return VerificationResult(
            text=cleaned,
            stripped_markers=[f"[{m}]" for m in stripped],
            grounding_status="no_evidence",
        )

    verified: dict[str, RetrievedChunk] = {}
    invented: list[str] = []

    def resolve(match: re.Match[str]) -> str:
        marker = match.group(0)
        index = int(match.group(1))
        # Markers are 1-based over the retrieval set, in the order the tool
        # handed them out.
        if 1 <= index <= len(retrieved):
            verified.setdefault(marker, retrieved[index - 1])
            return marker
        invented.append(marker)
        return ""

    cleaned = _tidy(_MARKER.sub(resolve, draft))

    if invented:
        logger.warning(
            "stripped citation markers that match no retrieved passage",
            extra={"markers": sorted(set(invented)), "retrieved": len(retrieved)},
        )

    citations = [
        VerifiedCitation(marker=marker, chunk=chunk)
        for marker, chunk in sorted(verified.items(), key=lambda kv: kv[0])
    ]

    if citations:
        status = "grounded"
    elif invented:
        # Every marker was invented. Stripping them left prose that reads as
        # grounded but is not, so the turn must not claim to be.
        status = "no_evidence"
    else:
        # The model cited nothing, though passages were available — an answer
        # that may be true but is not evidenced here.
        status = "degraded"

    return VerificationResult(
        text=cleaned,
        citations=citations,
        stripped_markers=sorted(set(invented)),
        grounding_status=status,
    )
