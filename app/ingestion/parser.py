"""Phase 1 — PDF bytes to per-page text.

Everything downstream depends on page numbers being real, because a chunk
without a page is uncitable (ARCHITECTURE 4.5). Pages that yield no text are
recorded rather than dropped, so the honest "these pages could not be read"
message in the partial-ready state has something to say.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

# A page with fewer than this many characters is treated as unreadable — a
# scanned page usually yields a handful of stray ligatures, not nothing.
MIN_CHARS_FOR_READABLE_PAGE = 40

# Ratio below which the document is considered a scan we cannot serve.
MIN_EXTRACTABLE_RATIO = 0.10


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

    @property
    def is_partial(self) -> bool:
        return bool(self.unreadable_pages)


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
    text = re.sub(r"[ \t ]+", " ", text)
    # Words split across a line break: "gradi-\nent" -> "gradient".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(data: bytes) -> ParsedDocument:
    """Extract text per page.

    Raises PdfEncryptedError, PdfCorruptError or PdfNotExtractableError — all
    permanent, none worth a retry.
    """
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
    except PdfReadError as exc:
        raise PdfCorruptError(str(exc)) from exc
    except Exception as exc:  # pypdf raises assorted types on malformed input
        raise PdfCorruptError(str(exc)) from exc

    if reader.is_encrypted:
        # An empty user password is common and harmless; anything else is a
        # document we were not given the means to read.
        try:
            if reader.decrypt("") == 0:
                raise PdfEncryptedError("PDF is password protected.")
        except PdfEncryptedError:
            raise
        except Exception as exc:
            raise PdfEncryptedError(str(exc)) from exc

    try:
        raw_pages = list(reader.pages)
    except Exception as exc:
        raise PdfCorruptError(str(exc)) from exc

    if not raw_pages:
        raise PdfCorruptError("PDF contains no pages.")

    pages: list[Page] = []
    for index, raw_page in enumerate(raw_pages, start=1):
        try:
            text = _normalise(raw_page.extract_text() or "")
        except Exception as exc:
            # One broken page must not lose the other ninety-nine.
            logger.warning("page %d failed extraction: %s", index, exc)
            text = ""
        pages.append(Page(number=index, text=text))

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
        title=_guess_title(reader, pages),
    )


def probe_page_count(data: bytes) -> int:
    """Page count without extracting any text.

    Cheap enough to run at the upload boundary, so an over-long document is
    refused with a useful message instead of failing 40 seconds into a job.
    """
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise PdfEncryptedError("PDF is password protected.")
        return len(reader.pages)
    except (PdfEncryptedError, PdfCorruptError):
        raise
    except Exception as exc:
        raise PdfCorruptError(str(exc)) from exc


def _guess_title(reader: PdfReader, pages: list[Page]) -> str | None:
    """Best-effort title, later refined by the model during analysis.

    PDF metadata is frequently the LaTeX job name rather than the paper title,
    so a metadata title is only trusted when it looks like prose.
    """
    try:
        meta_title = (reader.metadata or {}).get("/Title")
    except Exception:
        meta_title = None

    if isinstance(meta_title, str):
        candidate = meta_title.strip()
        if 10 <= len(candidate) <= 300 and " " in candidate:
            return candidate

    if pages and pages[0].text:
        first_line = pages[0].text.split("\n", 1)[0].strip()
        if 10 <= len(first_line) <= 300:
            return first_line

    return None
