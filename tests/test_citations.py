"""Deterministic citation verification (ARCHITECTURE 9.2 step 9).

A citation is a retrieval row that got flagged, never a claim the model made.
These tests pin the consequences of that: a marker the model invented is
removed from the answer, and an answer left with no evidence is not allowed to
present itself as grounded.
"""

import uuid

from app.services.citations import verify
from app.services.retrieval import RetrievedChunk


def _chunk(index: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        paper_id=uuid.uuid4(),
        content=f"passage {index}",
        similarity=0.9 - index / 100,
        rank=index,
        page_start=index,
        page_end=index,
        section_path=f"{index}.0",
        section_heading=f"Section {index}",
        section_role="method",
    )


def test_a_marker_matching_a_retrieved_passage_is_kept():
    retrieved = [_chunk(1), _chunk(2)]

    result = verify("The bound contains a KL term [1].", retrieved)

    assert result.text == "The bound contains a KL term [1]."
    assert [c.marker for c in result.citations] == ["[1]"]
    assert result.citations[0].chunk is retrieved[0]
    assert result.grounding_status == "grounded"


def test_markers_map_positionally_to_the_retrieval_set():
    retrieved = [_chunk(1), _chunk(2), _chunk(3)]

    result = verify("First [1], second [2], third [3].", retrieved)

    assert [c.chunk for c in result.citations] == retrieved


def test_an_invented_marker_is_stripped_from_the_answer():
    """The model cannot cite something it was never given."""
    retrieved = [_chunk(1)]

    result = verify("Grounded [1]. Invented [7].", retrieved)

    assert "[7]" not in result.text
    assert "[1]" in result.text
    assert result.stripped_markers == ["[7]"]
    assert result.grounding_status == "grounded"


def test_stripping_every_marker_downgrades_to_no_evidence():
    """Prose that reads as grounded but is not must not claim to be."""
    retrieved = [_chunk(1)]

    result = verify("According to the paper [4], the result holds [9].", retrieved)

    assert "[4]" not in result.text and "[9]" not in result.text
    assert result.citations == []
    assert result.grounding_status == "no_evidence"


def test_citing_nothing_when_passages_existed_is_degraded():
    """Possibly true, but not evidenced here — and that distinction is kept."""
    result = verify("Attention weights every token pair.", [_chunk(1)])

    assert result.grounding_status == "degraded"
    assert result.citations == []


def test_markers_with_an_empty_retrieval_set_cannot_resolve():
    result = verify("The paper says so [1].", [])

    assert "[1]" not in result.text
    assert result.grounding_status == "no_evidence"


def test_an_empty_draft_is_no_evidence():
    assert verify("", [_chunk(1)]).grounding_status == "no_evidence"
    assert verify("   ", [_chunk(1)]).grounding_status == "no_evidence"


def test_the_same_marker_used_twice_is_one_citation():
    retrieved = [_chunk(1)]

    result = verify("Here [1], and again [1].", retrieved)

    assert len(result.citations) == 1
    assert result.text.count("[1]") == 2


def test_removing_a_marker_does_not_leave_a_gap_before_punctuation():
    result = verify("The result holds [5].", [_chunk(1)])

    assert result.text == "The result holds."


def test_cited_chunk_ids_are_what_gets_flagged():
    retrieved = [_chunk(1), _chunk(2)]

    result = verify("Only the first [1].", retrieved)

    assert result.cited_chunk_ids == {retrieved[0].chunk_id}
    assert retrieved[1].chunk_id not in result.cited_chunk_ids
