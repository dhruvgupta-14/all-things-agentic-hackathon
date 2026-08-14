"""Phase 2 — pages to sections.

Section boundaries are what make chunking precise and citations human-usable:
"section 3.2, p.5" is actionable where "chunk 47" is not. Detection is regex
over heading conventions, which academic papers follow closely. Where it fails,
the role is `unknown` — the honest fallback, never a fabricated guess
(ARCHITECTURE 4.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.parser import Page

# Numbered headings: "3 Method", "3.2 Training", "IV. Results", "A.1 Proofs".
_NUMBERED = re.compile(
    r"^\s{0,3}(?P<path>(?:\d+|[A-Z])(?:\.\d+){0,3})\.?\s+(?P<title>[^\n]{2,120})$"
)
_ROMAN = re.compile(
    r"^\s{0,3}(?P<path>[IVXLC]{1,6})\.\s+(?P<title>[^\n]{2,120})$"
)
# Unnumbered but well-known: "Abstract", "References", "Acknowledgements".
_BARE = re.compile(r"^\s{0,3}(?P<title>[A-Z][A-Za-z ]{2,40})\s*$")

# Longest match wins, so "related work" is not swallowed by "work".
ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"related\s+work|prior\s+work|background", "related_work"),
    (r"experimental\s+setup|experiments?|evaluation|setup", "experiments"),
    (r"results?|findings", "results"),
    (r"discussion|analysis|ablation", "discussion"),
    (r"conclusions?|future\s+work|summary", "conclusion"),
    (r"references?|bibliography", "references"),
    (r"appendix|supplementary|supplemental", "appendix"),
    (r"introduction", "introduction"),
    (r"abstract", "abstract"),
    (r"methods?|methodology|approach|model|architecture|our\s+\w+", "method"),
)

_BARE_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experimental setup",
    "evaluation",
    "results",
    "discussion",
    "analysis",
    "conclusion",
    "conclusions",
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "appendix",
}

MAX_HEADING_WORDS = 14


@dataclass(slots=True)
class DetectedSection:
    ordinal: int
    heading: str | None
    section_path: str
    section_role: str
    page_start: int
    page_end: int
    text: str


def classify_role(heading: str | None) -> str:
    if not heading:
        return "unknown"
    lowered = heading.lower()
    for pattern, role in ROLE_PATTERNS:
        if re.search(pattern, lowered):
            return role
    return "unknown"


def _match_heading(line: str) -> tuple[str, str] | None:
    """Return (section_path, heading) when this line looks like a heading."""
    stripped = line.strip()
    if not stripped or len(stripped.split()) > MAX_HEADING_WORDS:
        return None
    # A sentence, not a heading.
    if stripped.endswith((".", ",", ";", ":")) and not re.match(r"^\d", stripped):
        return None

    for pattern in (_NUMBERED, _ROMAN):
        match = pattern.match(stripped)
        if match:
            title = match.group("title").strip()
            # Reject prose that merely opens with a number ("2019 saw ...").
            if title and title[0].isupper() and not title.endswith("."):
                return match.group("path"), title

    bare = _BARE.match(stripped)
    if bare and bare.group("title").strip().lower() in _BARE_HEADINGS:
        title = bare.group("title").strip()
        return "", title

    return None


def detect_sections(pages: list[Page]) -> list[DetectedSection]:
    """Split the document at detected headings.

    Text before the first heading becomes a leading `unknown` section rather
    than being discarded — on many papers that region is the title block and
    abstract.
    """
    readable = [page for page in pages if page.text.strip()]
    if not readable:
        return []

    # (page_number, line)
    lines: list[tuple[int, str]] = [
        (page.number, line)
        for page in readable
        for line in page.text.split("\n")
    ]

    boundaries: list[tuple[int, str, str]] = []  # (line index, path, heading)
    for index, (_, line) in enumerate(lines):
        match = _match_heading(line)
        if match:
            boundaries.append((index, match[0], match[1]))

    if not boundaries or boundaries[0][0] > 0:
        boundaries.insert(0, (0, "", ""))

    sections: list[DetectedSection] = []
    for ordinal, (start, path, heading) in enumerate(boundaries):
        end = boundaries[ordinal + 1][0] if ordinal + 1 < len(boundaries) else len(lines)
        body = lines[start:end]
        if not body:
            continue

        text = "\n".join(line for _, line in body).strip()
        if not text:
            continue

        page_numbers = [page for page, _ in body]
        sections.append(
            DetectedSection(
                ordinal=len(sections),
                heading=heading or None,
                # A section with no detected number still needs a stable,
                # citable path, so fall back to its position. ASCII-only, and
                # shaped so it cannot be mistaken for a printed section number.
                section_path=path or f"sec-{len(sections)}",
                section_role=classify_role(heading),
                page_start=min(page_numbers),
                page_end=max(page_numbers),
                text=text,
            )
        )

    return sections
