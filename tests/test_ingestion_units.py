"""Unit coverage for the three ingestion phases.

These are pure functions over text, so they need neither a database nor an
event loop.
"""

import pytest

from app.ingestion.chunker import MAX_CHARS, TARGET_CHARS, chunk_sections, estimate_tokens
from app.ingestion.parser import (
    Page,
    PdfCorruptError,
    parse_pdf,
    probe_page_count,
)
from app.ingestion.sectioner import classify_role, detect_sections
from tests.conftest import build_pdf

# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def test_parses_pages_and_keeps_page_numbers():
    document = parse_pdf(
        build_pdf(
            [
                "First page about gradients and the loss surface of networks.",
                "Second page discussing attention mechanisms in some detail.",
            ]
        )
    )

    assert document.page_count == 2
    assert [page.number for page in document.pages] == [1, 2]
    assert "gradients" in document.pages[0].text
    assert document.extractable_text_ratio == 1.0
    assert document.unreadable_pages == []


def test_blank_pages_are_recorded_not_dropped():
    """The partial-ready message needs to know which pages failed."""
    document = parse_pdf(
        build_pdf(["A page with a good deal of readable text written on it.", "", ""])
    )

    assert document.page_count == 3
    assert document.unreadable_pages == [2, 3]
    assert document.is_partial
    assert document.extractable_text_ratio == pytest.approx(1 / 3, abs=0.01)


def test_scanned_document_is_rejected_as_permanent():
    from app.ingestion.parser import PdfNotExtractableError

    with pytest.raises(PdfNotExtractableError):
        parse_pdf(build_pdf([""] * 12))


def test_corrupt_input_raises_permanent_error():
    with pytest.raises(PdfCorruptError):
        parse_pdf(b"%PDF-1.4 this is not really a pdf")


def test_probe_page_count_matches_parse():
    data = build_pdf(["one", "two", "three"])
    assert probe_page_count(data) == 3


def test_hyphenated_words_are_rejoined():
    """De-hyphenation has to happen before newlines are collapsed."""
    document = parse_pdf(
        build_pdf(["The gradi-\nent descends the loss surface to a minimum here."])
    )
    assert "gradient" in document.pages[0].text


# --------------------------------------------------------------------------
# Sectioner
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("Abstract", "abstract"),
        ("1 Introduction", "introduction"),
        ("Related Work", "related_work"),
        ("3 Method", "method"),
        ("4 Experiments", "experiments"),
        ("5 Results", "results"),
        ("6 Discussion", "discussion"),
        ("7 Conclusion", "conclusion"),
        ("References", "references"),
        ("A Appendix", "appendix"),
        ("Zebra Husbandry", "unknown"),
        (None, "unknown"),
    ],
)
def test_role_classification(heading, expected):
    assert classify_role(heading) == expected


def test_related_work_is_not_swallowed_by_a_shorter_pattern():
    """Ordering matters: 'work' must not win over 'related work'."""
    assert classify_role("2 Related Work") == "related_work"


def test_sections_split_on_numbered_headings():
    pages = [
        Page(
            number=1,
            text=(
                "1 Introduction\nWe study attention.\n"
                "2 Method\nWe propose a model.\n"
                "3 Results\nIt works well."
            ),
        )
    ]
    sections = detect_sections(pages)

    assert [s.section_path for s in sections] == ["1", "2", "3"]
    assert [s.section_role for s in sections] == ["introduction", "method", "results"]
    assert "attention" in sections[0].text


def test_text_before_the_first_heading_is_kept():
    """On most papers that region is the title block and abstract."""
    pages = [Page(number=1, text="Some Paper Title\nAuthors here\n1 Introduction\nBody.")]
    sections = detect_sections(pages)

    assert len(sections) == 2
    assert sections[0].section_role == "unknown"
    assert "Some Paper Title" in sections[0].text


def test_section_page_spans_come_from_the_pages_they_cover():
    pages = [
        Page(number=3, text="1 Introduction\nStart of intro."),
        Page(number=4, text="Continues here.\n2 Method\nMethod body."),
    ]
    sections = detect_sections(pages)

    assert sections[0].page_start == 3
    assert sections[0].page_end == 4
    assert sections[1].page_start == 4


def test_prose_beginning_with_a_number_is_not_a_heading():
    pages = [Page(number=1, text="1 Introduction\n2019 saw a surge in transformers.")]
    sections = detect_sections(pages)
    assert len(sections) == 1


def test_no_readable_pages_yields_no_sections():
    assert detect_sections([Page(number=1, text="   ")]) == []


# --------------------------------------------------------------------------
# Chunker
# --------------------------------------------------------------------------


def _section(ordinal, role, text, page_start=1, page_end=1):
    from app.ingestion.sectioner import DetectedSection

    return DetectedSection(
        ordinal=ordinal,
        heading=role.title(),
        section_path=str(ordinal),
        section_role=role,
        page_start=page_start,
        page_end=page_end,
        text=text,
    )


def test_chunks_never_cross_a_section_boundary():
    """The rule the schema enforces with an FK; this is what satisfies it."""
    sections = [
        _section(0, "method", "Alpha. " * 400),
        _section(1, "results", "Beta. " * 400),
    ]
    chunks = chunk_sections(sections)

    assert len(chunks) > 2
    for chunk in chunks:
        owner = sections[chunk.section_ordinal]
        marker = "Alpha" if owner.section_role == "method" else "Beta"
        other = "Beta" if marker == "Alpha" else "Alpha"
        assert other not in chunk.content


def test_every_chunk_carries_a_page_span():
    """A chunk without a page is uncitable."""
    chunks = chunk_sections([_section(0, "method", "Body. " * 300, 4, 6)])

    assert chunks
    for chunk in chunks:
        assert chunk.page_start == 4
        assert chunk.page_end == 6
        assert chunk.page_start > 0


def test_chunks_respect_the_schema_length_limit():
    chunks = chunk_sections([_section(0, "method", "word " * 20000)])
    assert chunks
    for chunk in chunks:
        assert 0 < len(chunk.content) <= MAX_CHARS


def test_ordinals_are_contiguous_across_the_document():
    chunks = chunk_sections(
        [_section(0, "method", "A. " * 300), _section(1, "results", "B. " * 300)]
    )
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_reference_sections_are_stored_but_not_indexed():
    chunks = chunk_sections(
        [_section(0, "references", "[1] Someone. A paper. 2019. " * 40)]
    )
    assert chunks
    assert all(chunk.is_indexable is False for chunk in chunks)


def test_body_sections_are_indexable():
    chunks = chunk_sections([_section(0, "method", "Real content. " * 60)])
    assert all(chunk.is_indexable for chunk in chunks)


def test_short_section_becomes_one_chunk():
    chunks = chunk_sections([_section(0, "abstract", "A short abstract about models.")])
    assert len(chunks) == 1
    assert chunks[0].content == "A short abstract about models."


def test_content_hash_is_stable_and_distinct():
    chunks = chunk_sections([_section(0, "method", "Alpha content here.")])
    again = chunk_sections([_section(0, "method", "Alpha content here.")])
    other = chunk_sections([_section(0, "method", "Different content here.")])

    assert chunks[0].content_hash == again[0].content_hash
    assert chunks[0].content_hash != other[0].content_hash


def test_packing_targets_the_intended_size():
    chunks = chunk_sections([_section(0, "method", "Sentence here. " * 500)])
    # Overlap can push a chunk slightly past target; it must not run away.
    assert all(len(c.content) <= TARGET_CHARS * 2 for c in chunks)


def test_token_estimate_scales_with_length():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 400) == 100
