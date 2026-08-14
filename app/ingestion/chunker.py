"""Phase 3 — sections to chunks.

Two rules are structural rather than stylistic:

* A chunk never crosses a section boundary. The schema enforces this with an
  FK to exactly one section (ARCHITECTURE 4.5); this module is what makes the
  data satisfy it.
* Every chunk carries a real page span, because a chunk without a page cannot
  be cited.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.ingestion.sectioner import DetectedSection

TARGET_CHARS = 1400
MAX_CHARS = 8000  # the hard schema limit on chunks.content
MIN_CHARS = 120  # below this a chunk is merged rather than stored alone
OVERLAP_CHARS = 180

# Reference lists retrieve badly and cite worse: kept for completeness, hidden
# from the ANN index.
NON_INDEXABLE_ROLES = frozenset({"references"})


@dataclass(slots=True)
class DetectedChunk:
    ordinal: int
    section_ordinal: int
    content: str
    content_hash: str
    token_count: int
    page_start: int
    page_end: int
    is_indexable: bool


def estimate_tokens(text: str) -> int:
    """Approximate token count.

    A real tokenizer is not worth a dependency here: this feeds context
    budgeting, where being 10% out is harmless. English prose runs about four
    characters per token.
    """
    return max(1, round(len(text) / 4))


def _flow(text: str) -> str:
    """Rejoin the per-line text the parser preserved for the sectioner.

    Hard line breaks inside a paragraph are a layout artefact of the PDF, not
    meaning. They are kept through section detection because headings are
    identified by occupying their own line, and undone here so stored chunks
    read as prose and embed cleanly.
    """
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text).strip()


def _split_paragraphs(text: str) -> list[str]:
    parts = [_flow(part) for part in re.split(r"\n\s*\n", text)]
    parts = [part for part in parts if part]
    return parts or ([_flow(text)] if text.strip() else [])


def _split_sentences(text: str) -> list[str]:
    """Sentence split that does not break on common academic abbreviations."""
    protected = re.sub(
        r"\b(et al|e\.g|i\.e|cf|Fig|Eq|Sec|Tab|vs|approx|Dr|Prof)\.",
        lambda m: m.group(0).replace(".", "\x00"),
        text,
    )
    pieces = re.split(r"(?<=[.!?])\s+", protected)
    return [piece.replace("\x00", ".").strip() for piece in pieces if piece.strip()]


def _hard_split(text: str, size: int) -> list[str]:
    """Last-resort split for text with no usable boundary.

    A single unbroken run longer than the target — a table dump, a base64
    blob, a language this sentence splitter does not segment — has to be cut
    somewhere. Prefer the nearest preceding space so words survive.
    """
    pieces: list[str] = []
    remaining = text
    while len(remaining) > size:
        window = remaining[:size]
        cut = window.rfind(" ")
        if cut < size // 2:  # no sensible boundary; cut at the limit
            cut = size
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return [piece for piece in pieces if piece]


def _pack(units: list[str]) -> list[str]:
    """Greedily pack units up to TARGET_CHARS, with a little overlap."""
    chunks: list[str] = []
    current = ""

    for unit in units:
        # An oversized unit is split on sentences rather than emitted whole.
        # The comparison is against TARGET_CHARS, not MAX_CHARS: a 7 000-char
        # paragraph is under the schema limit but far too coarse to retrieve.
        if len(unit) > TARGET_CHARS:
            if current:
                chunks.append(current)
                current = ""
            sentences = _split_sentences(unit)
            # Guard against unbounded recursion: if splitting did not actually
            # break the unit down, fall back to a hard split.
            if len(sentences) <= 1:
                chunks.extend(_hard_split(unit, TARGET_CHARS))
            else:
                chunks.extend(_pack(sentences))
            continue

        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= TARGET_CHARS or not current:
            current = candidate
        else:
            chunks.append(current)
            tail = current[-OVERLAP_CHARS:] if len(current) > OVERLAP_CHARS else ""
            # Overlap starts at a word boundary so the carried text reads.
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            current = f"{tail}\n\n{unit}" if tail else unit

    if current:
        chunks.append(current)

    return chunks


def _merge_runts(chunks: list[str]) -> list[str]:
    """Fold anything too short to stand alone into its neighbour."""
    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < MIN_CHARS and len(merged[-1]) + len(chunk) <= MAX_CHARS:
            merged[-1] = f"{merged[-1]}\n\n{chunk}"
        else:
            merged.append(chunk)
    return merged


def chunk_sections(sections: list[DetectedSection]) -> list[DetectedChunk]:
    chunks: list[DetectedChunk] = []

    for section in sections:
        units = _split_paragraphs(section.text)
        if not units:
            continue

        for body in _merge_runts(_pack(units)):
            content = body.strip()
            if not content:
                continue
            content = content[:MAX_CHARS]

            chunks.append(
                DetectedChunk(
                    ordinal=len(chunks),
                    section_ordinal=section.ordinal,
                    content=content,
                    content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    token_count=min(estimate_tokens(content), 32767),
                    # Page attribution is inherited from the section. Finer
                    # per-chunk pages need character offsets the parser does
                    # not currently preserve; the span is always correct, just
                    # wider than ideal on multi-page sections.
                    page_start=section.page_start,
                    page_end=section.page_end,
                    is_indexable=section.section_role not in NON_INDEXABLE_ROLES,
                )
            )

    return chunks
