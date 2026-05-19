"""
Retrieval regression tests for the Railway RAG system
"""
from unittest.mock import Mock

from utils import (
    apply_source_diversity,
    deduplicate_chunks,
    extract_technical_identifiers,
    parse_citations,
    TOKEN_BUDGET,
)


def apply_token_budget(context_parts: list[str]) -> list[str]:
    max_chars = TOKEN_BUDGET * 4
    joined = "\n\n".join(context_parts)

    if len(joined) <= max_chars:
        return context_parts

    truncated = joined[:max_chars]
    last_double_newline = truncated.rfind("\n\n")
    if last_double_newline > max_chars * 0.8:
        truncated = truncated[:last_double_newline]

    parts = truncated.split("\n\n")
    return [part for part in parts if part.strip()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_deduplication_removes_duplicates():
    """Test that deduplication removes duplicates while keeping first occurrence."""
    doc1 = Mock()
    doc1.page_content = "Identical content"
    doc1.metadata = {"source": "test.pdf", "page_number": 1}

    doc2 = Mock()
    doc2.page_content = "Unique content 1"
    doc2.metadata = {"source": "test.pdf", "page_number": 2}

    doc3 = Mock()
    doc3.page_content = "Identical content"
    doc3.metadata = {"source": "test.pdf", "page_number": 1}

    doc4 = Mock()
    doc4.page_content = "Unique content 2"
    doc4.metadata = {"source": "test.pdf", "page_number": 3}

    doc5 = Mock()
    doc5.page_content = "Another unique content"
    doc5.metadata = {"source": "test.pdf", "page_number": 4}

    documents = [doc1, doc2, doc3, doc4, doc5]
    unique_docs = deduplicate_chunks(documents)

    assert len(unique_docs) == 4
    assert unique_docs[0].page_content == "Identical content"
    assert unique_docs[1].page_content == "Unique content 1"
    assert unique_docs[2].page_content == "Unique content 2"
    assert unique_docs[3].page_content == "Another unique content"


def test_deduplication_preserves_order():
    """Test that deduplication preserves the original order."""
    doc1 = Mock()
    doc1.page_content = "First"
    doc1.metadata = {"source": "test.pdf", "page_number": 1}

    doc2 = Mock()
    doc2.page_content = "Second"
    doc2.metadata = {"source": "test.pdf", "page_number": 2}

    doc3 = Mock()
    doc3.page_content = "Third"
    doc3.metadata = {"source": "test.pdf", "page_number": 3}

    documents = [doc1, doc2, doc3]
    unique_docs = deduplicate_chunks(documents)

    assert len(unique_docs) == 3
    assert unique_docs[0].page_content == "First"
    assert unique_docs[1].page_content == "Second"
    assert unique_docs[2].page_content == "Third"


def test_source_diversity_limits_per_page():
    """Test that apply_source_diversity limits chunks per source-page pair."""
    docs = []
    for i in range(5):
        doc = Mock()
        doc.page_content = f"Content {i}"
        doc.metadata = {"source": "test.pdf", "page_number": 1}
        docs.append(doc)
    for i in range(3):
        doc = Mock()
        doc.page_content = f"Content B{i}"
        doc.metadata = {"source": "test.pdf", "page_number": 2}
        docs.append(doc)

    result = apply_source_diversity(docs, max_per_page=2)

    assert len(result) == 4
    assert sum(1 for d in result if d.metadata["page_number"] == 1) == 2
    assert sum(1 for d in result if d.metadata["page_number"] == 2) == 2


def test_source_diversity_preserves_under_limit():
    """Test that apply_source_diversity keeps all when under max."""
    docs = []
    for i in range(2):
        doc = Mock()
        doc.page_content = f"Content {i}"
        doc.metadata = {"source": "test.pdf", "page_number": 1}
        docs.append(doc)

    result = apply_source_diversity(docs, max_per_page=3)
    assert len(result) == 2


def test_token_budget_truncation():
    """Test that token budget logic properly truncates oversized context."""
    excess_chars = 500 * 4
    base_content = "x" * (TOKEN_BUDGET * 4)
    excess_content = "y" * excess_chars
    large_content = base_content + excess_content

    context_parts = [
        "First paragraph of content.",
        large_content,
        "Last paragraph of content.",
    ]

    truncated_parts = apply_token_budget(context_parts)
    joined = "\n\n".join(truncated_parts)

    assert len(joined) <= TOKEN_BUDGET * 4
    assert len(truncated_parts) > 0
    assert any(part.strip() for part in truncated_parts)


def test_identifier_extraction_finds_codes():
    """Test that technical_codes regex finds expected patterns."""
    text = "Check regulation 45/2019 and refer to EN-302 and SPAD standards."

    technical_codes = extract_technical_identifiers(text)

    assert "45/2019" in technical_codes
    assert any(code in ["EN", "SPAD"] for code in technical_codes)


def test_identifier_extraction_empty_on_plain_text():
    """Test that plain text doesn't produce false positives."""
    text = "What is the maximum permitted speed on this line?"

    technical_codes = extract_technical_identifiers(text)

    assert "What" not in technical_codes
    assert len(technical_codes) == 0


def test_citation_parse_valid_json_list():
    """Test parsing of valid JSON list in citation tags."""
    text = "Some answer.\n<used_chunks>[0, 2, 4]</used_chunks>"

    display_response, chunk_indices = parse_citations(text)

    assert chunk_indices == [0, 2, 4]
    assert display_response.strip() == "Some answer."


def test_citation_parse_malformed_gracefully():
    """Test that malformed citation content is handled gracefully."""
    text = "Some answer.\n<used_chunks>[0, 'two', 4]</used_chunks>"

    display_response, chunk_indices = parse_citations(text)

    assert all(isinstance(i, int) for i in chunk_indices)
    assert 0 in chunk_indices
    assert 4 in chunk_indices
    assert len(chunk_indices) == 2
    assert display_response.strip() == "Some answer."


def test_citation_strip_from_display():
    """Test that <used_chunks> tag is stripped from display text."""
    text = "This is the answer.\n<used_chunks>[1, 3, 5]</used_chunks>\nEnd of answer."

    display_response, _ = parse_citations(text)

    assert "<used_chunks>" not in display_response
    assert "</used_chunks>" not in display_response
    assert "This is the answer." in display_response
    assert "End of answer." in display_response
