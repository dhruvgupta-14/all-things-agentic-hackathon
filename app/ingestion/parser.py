"""Phase 1 — PDF bytes to per-page text, in reading order, with hostile
content removed.

PyMuPDF is used rather than a pure-text extractor because three CORE
requirements need span-level geometry that text-only libraries do not expose:

* **Reading order.** Two-column papers interleave into nonsense when read by
  vertical position alone. Column detection needs each span's x-position.
* **Invisible-text stripping.** The prompt-injection vector for a PDF is text
  the human reader cannot see — white on white, zero-sized, or positioned off
  the page. Detecting it needs colour, size and bounding box.
* **Heading detection.** Font size and weight identify a heading that carries
  no section number, which regex alone cannot.

PyMuPDF is AGPL-3.0. That is a deliberate, recorded project decision.

Everything downstream depends on page numbers being real, because a chunk
without a page is uncitable (ARCHITECTURE 4.5). Pages that yield no text are
recorded rather than dropped, so the honest "these pages could not be read"
message has something to say.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pymupdf

logger = logging.getLogger(__name__)

# A page with fewer than this many characters is treated as unreadable — a
# scanned page usually yields a handful of stray ligatures, not nothing.
MIN_CHARS_FOR_READABLE_PAGE = 40

# Ratio below which the document is considered a scan we cannot serve.
MIN_EXTRACTABLE_RATIO = 0.10

# Text lighter than this on white is invisible to a reader. Expressed as the
# minimum channel distance from pure white, per RGB channel.
WHITE_PROXIMITY = 24

# Below this point size text is decorative at best and hidden at worst.
MIN_VISIBLE_FONT_SIZE = 4.0

# PDF text render mode 3 = "neither fill nor stroke", i.e. drawn invisibly.
# It is the cleanest way to hide instructions in a PDF.
RENDER_MODE_INVISIBLE = 3

# Phrases that have no business in a research paper and are the signature of
# an injection attempt. Recorded, never acted on.
_INJECTION_PATTERNS = (
    r"ignore (?:all |the )?(?:previous|prior|above) instructions",
    r"disregard (?:all |the )?(?:previous|prior|above)",
    r"you are now",
    r"system prompt",
    r"reveal (?:your|the) (?:prompt|instructions)",
    r"grant (?:me |the user )?access",
    r"</?(?:system|assistant|user)>",
)
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


class PdfEncryptedError(Exception):
    """Password-protected. Permanent: retrying will not help."""


class PdfCorruptError(Exception):
    """Unparseable as a PDF. Permanent."""


class PdfNotExtractableError(Exception):
    """Parseable, but effectively a scan. Permanent."""


@dataclass(slots=True)
class Page:
    number: int  # 1-based, as printed in a citation
    text: str
    # Lines whose typography marks them as headings. The sectioner trusts
    # these in addition to its own numbering regexes.
    headings: list[str] = field(default_factory=list)

    @property
    def is_readable(self) -> bool:
        return len(self.text.strip()) >= MIN_CHARS_FOR_READABLE_PAGE


@dataclass(slots=True)
class ParsedDocument:
    pages: list[Page]
    page_count: int
    extractable_text_ratio: float
    unreadable_pages: list[int] = field(default_factory=list)
    title: str | None = None
    # Goes to papers.security_findings — what was stripped and what was seen.
    security_findings: dict = field(default_factory=dict)

    @property
    def is_partial(self) -> bool:
        return bool(self.unreadable_pages)


def _open(data: bytes) -> pymupdf.Document:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PdfCorruptError(str(exc)) from exc

    # An empty owner password is common and harmless; anything else is a
    # document we were not given the means to read.
    if document.needs_pass and not document.authenticate(""):
        raise PdfEncryptedError("PDF is password protected.")

    if document.page_count == 0:
        raise PdfCorruptError("PDF contains no pages.")
    return document


def _is_invisible(span: dict, page_rect: pymupdf.Rect) -> str | None:
    """Return why this span is invisible to a human reader, or None."""
    if span.get("size", 0) < MIN_VISIBLE_FONT_SIZE:
        return "tiny_font"

    # PyMuPDF exposes the text render mode when it is not the default.
    if span.get("render_mode", 0) == RENDER_MODE_INVISIBLE:
        return "render_mode_3"

    colour = span.get("color", 0)
    red, green, blue = (colour >> 16) & 255, (colour >> 8) & 255, colour & 255
    if min(255 - red, 255 - green, 255 - blue) < WHITE_PROXIMITY:
        return "white_on_white"

    bbox = pymupdf.Rect(span.get("bbox", (0, 0, 0, 0)))
    if bbox.is_empty or not bbox.intersects(page_rect):
        return "off_page"

    return None


def _columns(blocks: list[dict], page_rect: pymupdf.Rect) -> int:
    """Detect a two-column layout from the distribution of block left edges.

    A two-column paper has blocks clustered at two distinct x positions with a
    gutter between them. One column has a single cluster. Reading a two-column
    page top-to-bottom interleaves the columns into nonsense, which is the
    single worst thing that can happen to retrieval quality.
    """
    lefts = [
        block["bbox"][0]
        for block in blocks
        if block.get("type") == 0 and block["bbox"][2] - block["bbox"][0] > 20
    ]
    if len(lefts) < 6:
        return 1

    midline = (page_rect.x0 + page_rect.x1) / 2
    left_side = [x for x in lefts if x < midline]
    right_side = [x for x in lefts if x >= midline]

    # Both sides must be substantially populated, and a block that straddles
    # the midline (a full-width heading or figure) is not evidence either way.
    if len(left_side) < 3 or len(right_side) < 3:
        return 1

    spanning = sum(
        1
        for block in blocks
        if block.get("type") == 0
        and block["bbox"][0] < midline - 20
        and block["bbox"][2] > midline + 20
    )
    # A page of full-width paragraphs is single-column regardless of where
    # blocks happen to start.
    if spanning > len(lefts) / 3:
        return 1

    return 2


def _sorted_blocks(blocks: list[dict], page_rect: pymupdf.Rect) -> list[dict]:
    text_blocks = [block for block in blocks if block.get("type") == 0]
    midline = (page_rect.x0 + page_rect.x1) / 2

    if _columns(blocks, page_rect) == 1:
        return sorted(text_blocks, key=lambda b: (round(b["bbox"][1], 1), b["bbox"][0]))

    # Left column top-to-bottom, then right column top-to-bottom. A block that
    # spans the gutter is full-width and sorts by its vertical position within
    # the left column, which keeps titles and section headings in place.
    left, right = [], []
    for block in text_blocks:
        x0, _, x2, _ = block["bbox"][0], 0, block["bbox"][2], 0
        if x0 < midline and x2 > midline:
            left.append(block)  # spans the gutter: treat as left/full width
        elif x0 >= midline:
            right.append(block)
        else:
            left.append(block)

    by_position = lambda b: (round(b["bbox"][1], 1), b["bbox"][0])  # noqa: E731
    return sorted(left, key=by_position) + sorted(right, key=by_position)


def _normalise(raw: str) -> str:
    """Tidy extracted text while preserving its line structure.

    Line breaks are deliberately kept: a heading occupies its own line, and
    that is the only signal section detection has to work from. Reflowing
    lines into readable paragraphs is the chunker's job, done once the
    sectioner has taken what it needs.

    De-hyphenation runs before anything else touches newlines, since it is the
    line break itself that marks the split word.
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_page(
    page: pymupdf.Page, findings: dict
) -> tuple[str, list[str]]:
    """One page to (text in reading order, heading candidates)."""
    try:
        raw = page.get_text("dict")
    except Exception as exc:
        logger.warning("page %d failed extraction: %s", page.number + 1, exc)
        return "", []

    blocks = _sorted_blocks(raw.get("blocks", []), page.rect)

    sizes: list[float] = []
    lines_out: list[tuple[str, float, bool]] = []

    for block in blocks:
        for line in block.get("lines", []):
            parts: list[str] = []
            line_sizes: list[float] = []
            bold = False
            for span in line.get("spans", []):
                reason = _is_invisible(span, page.rect)
                if reason is not None:
                    # Stripped, not rendered. Counted so the finding is
                    # reportable rather than silent.
                    findings["invisible_spans"] = findings.get("invisible_spans", 0) + 1
                    findings.setdefault("invisible_reasons", {})
                    findings["invisible_reasons"][reason] = (
                        findings["invisible_reasons"].get(reason, 0) + 1
                    )
                    hidden = span.get("text", "").strip()
                    if hidden and _INJECTION_RE.search(hidden):
                        findings.setdefault("injection_hits", []).append(hidden[:200])
                    continue

                text = span.get("text", "")
                if not text.strip():
                    continue
                parts.append(text)
                line_sizes.append(float(span.get("size", 0)))
                # Bit 4 of the span flags marks a bold face.
                bold = bold or bool(int(span.get("flags", 0)) & 2**4)

            if not parts:
                continue
            joined = "".join(parts).strip()
            size = max(line_sizes) if line_sizes else 0.0
            sizes.append(size)
            lines_out.append((joined, size, bold))

    if not lines_out:
        return "", []

    # A heading is set larger than the body, or is bold and short. The body
    # size is the median, which is robust to a large title and to footnotes.
    ordered = sorted(sizes)
    body_size = ordered[len(ordered) // 2]

    text_lines: list[str] = []
    headings: list[str] = []
    for content, size, bold in lines_out:
        text_lines.append(content)
        looks_like_heading = size > body_size + 0.6 or (bold and size >= body_size)
        if len(content.split()) <= 12 and not content.endswith(".") and looks_like_heading:
            headings.append(content)

    return _normalise("\n".join(text_lines)), headings


def parse_pdf(data: bytes) -> ParsedDocument:
    """Extract text per page, in reading order, with invisible spans removed.

    Raises PdfEncryptedError, PdfCorruptError or PdfNotExtractableError — all
    permanent, none worth a retry.
    """
    document = _open(data)
    findings: dict = {}

    try:
        pages: list[Page] = []
        for index in range(document.page_count):
            try:
                page = document.load_page(index)
                text, headings = _extract_page(page, findings)
            except Exception as exc:
                # One broken page must not lose the other ninety-nine.
                logger.warning("page %d failed extraction: %s", index + 1, exc)
                text, headings = "", []
            pages.append(Page(number=index + 1, text=text, headings=headings))

        title = _guess_title(document, pages)
    finally:
        document.close()

    unreadable = [page.number for page in pages if not page.is_readable]
    ratio = (len(pages) - len(unreadable)) / len(pages)

    if ratio < MIN_EXTRACTABLE_RATIO:
        raise PdfNotExtractableError(
            f"Only {ratio:.0%} of pages yielded text; this looks like a scan."
        )

    return ParsedDocument(
        pages=pages,
        page_count=len(pages),
        extractable_text_ratio=round(ratio, 4),
        unreadable_pages=unreadable,
        title=title,
        security_findings=findings,
    )


def probe_page_count(data: bytes) -> int:
    """Page count without extracting any text.

    Cheap enough to run at the upload boundary, so an over-long document is
    refused with a useful message instead of failing 40 seconds into a job.
    """
    document = _open(data)
    try:
        return document.page_count
    finally:
        document.close()


def _guess_title(document: pymupdf.Document, pages: list[Page]) -> str | None:
    """Best-effort title, later refined by the model during analysis.

    PDF metadata is frequently the LaTeX job name rather than the paper title,
    so a metadata title is only trusted when it looks like prose.
    """
    try:
        meta_title = (document.metadata or {}).get("title")
    except Exception:
        meta_title = None

    if isinstance(meta_title, str):
        candidate = meta_title.strip()
        if 10 <= len(candidate) <= 300 and " " in candidate:
            return candidate

    # The largest text on page one is almost always the title.
    if pages and pages[0].headings:
        for heading in pages[0].headings:
            if 10 <= len(heading) <= 300:
                return heading

    if pages and pages[0].text:
        first_line = pages[0].text.split("\n", 1)[0].strip()
        if 10 <= len(first_line) <= 300:
            return first_line

    return None
