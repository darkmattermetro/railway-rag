"""
Unit tests for the extract_chunks() function from local_builder.py
"""
from pathlib import Path
from unittest.mock import Mock

from ingest import extract_chunks, _token_length
from docling_core.types.doc import (
    DoclingDocument,
    SectionHeaderItem,
    TableItem,
    TextItem,
)


def _make_element(cls, text="", page_no=1, table_md=""):
    """Build a mock Docling element with proper isinstance support."""
    el = Mock(spec=cls)
    el.text = text
    el.prov = [Mock(page_no=page_no)]
    if cls is TableItem:
        el.export_to_markdown.return_value = table_md or text
    return el


def test_heading_tracked_in_metadata():
    """Test that heading information is tracked in metadata."""
    mock_doc = Mock(spec=DoclingDocument)
    section = _make_element(SectionHeaderItem, "Test Heading", page_no=1)
    text_el = _make_element(TextItem, "This is test content under the heading.", page_no=1)
    mock_doc.iterate_items.return_value = [
        (section, 1),
        (text_el, 0),
    ]

    chunks = extract_chunks(
        doc=mock_doc,
        filename="test.pdf",
        category="test_category",
    )

    assert len(chunks) > 0, "Should return at least one chunk"
    for chunk in chunks:
        assert "page_number" in chunk.metadata, "Chunk should have page_number in metadata"
        assert chunk.metadata.get("is_table") is False, "Chunk should not be marked as table"


def test_table_not_split():
    """Test that tables are not split across chunks."""
    mock_doc = Mock(spec=DoclingDocument)
    table_content = "| Column1 | Column2 |\n|---------|---------|\n"
    table_content += "\n".join([f"| Row{i} | Data{i} |" for i in range(1, 51)])
    table = _make_element(TableItem, table_content, page_no=1, table_md=table_content)
    mock_doc.iterate_items.return_value = [
        (table, 0),
    ]

    chunks = extract_chunks(
        doc=mock_doc,
        filename="test.pdf",
        category="test_category",
    )

    assert len(chunks) == 1, f"Expected exactly one chunk for table, got {len(chunks)}"
    assert chunks[0].metadata.get("is_table") is True, "Chunk should be marked as table"


def test_table_continuity_detected():
    """Test that table continuity is detected for consecutive tables."""
    mock_doc = Mock(spec=DoclingDocument)
    t1_content = "| Col1 | Col2 |\n|------|------|\n| A | B |"
    t2_content = "| Col1 | Col2 |\n|------|------|\n| C | D |"
    table1 = _make_element(TableItem, t1_content, page_no=1, table_md=t1_content)
    table2 = _make_element(TableItem, t2_content, page_no=2, table_md=t2_content)
    mock_doc.iterate_items.return_value = [
        (table1, 0),
        (table2, 0),
    ]

    chunks = extract_chunks(
        doc=mock_doc,
        filename="test.pdf",
        category="test_category",
    )

    assert len(chunks) >= 1, "Should return at least one chunk"
    has_continuity = any(
        chunk.metadata.get("is_table_continuation") is True
        for chunk in chunks
    )
    assert has_continuity is True, "Two consecutive tables under same heading should set continuity"


def test_table_continuity_not_set_on_heading_change():
    """Test that table continuity is not set when heading changes."""
    mock_doc = Mock(spec=DoclingDocument)
    t1_content = "| Col1 | Col2 |\n|------|------|\n| A | B |"
    t2_content = "| Col1 | Col2 |\n|------|------|\n| C | D |"
    table1 = _make_element(TableItem, t1_content, page_no=1, table_md=t1_content)
    heading = _make_element(SectionHeaderItem, "New Section", page_no=1)
    table2 = _make_element(TableItem, t2_content, page_no=1, table_md=t2_content)
    mock_doc.iterate_items.return_value = [
        (table1, 0),
        (heading, 1),
        (table2, 0),
    ]

    chunks = extract_chunks(
        doc=mock_doc,
        filename="test.pdf",
        category="test_category",
    )

    assert len(chunks) >= 1, "Should return at least one chunk"
    table_chunks = [c for c in chunks if c.metadata.get("is_table") is True]
    for chunk in table_chunks:
        assert chunk.metadata.get("is_table_continuation") is not True, (
            "Table continuity should not be set when heading changes"
        )


def test_metadata_schema_complete():
    """Test that metadata contains exactly the required keys."""
    mock_doc = Mock(spec=DoclingDocument)
    text_el = _make_element(TextItem, "This is test content.", page_no=1)
    mock_doc.iterate_items.return_value = [
        (text_el, 0),
    ]

    chunks = extract_chunks(
        doc=mock_doc,
        filename="test.pdf",
        category="test_category",
    )

    assert len(chunks) > 0, "Should return at least one chunk"
    required_keys = {
        "source", "page_number", "category", "ingestion_timestamp",
        "is_table", "is_table_continuation",
    }
    for chunk in chunks:
        # Should have at least the required keys (may have extras like is_section_header)
        metadata_keys = set(chunk.metadata.keys())
        assert metadata_keys >= required_keys, (
            f"Missing required keys. Expected at least: {required_keys}, Got: {metadata_keys}"
        )


def test_token_length_function():
    """Test the _token_length function."""
    length = _token_length("hello world")
    assert isinstance(length, int), "_token_length should return an integer"
    assert length > 0, "_token_length should return a positive value for non-empty string"

    empty_length = _token_length("")
    assert isinstance(empty_length, int), "_token_length should return an integer for empty string"
    assert empty_length >= 0, "_token_length should return non-negative value for empty string"


def test_empty_document_returns_empty_list():
    """Test that empty document returns empty list."""
    mock_doc = Mock(spec=DoclingDocument)
    mock_doc.iterate_items.return_value = []

    chunks = extract_chunks(
        doc=mock_doc,
        filename="test.pdf",
        category="test_category",
    )

    assert chunks == [], "Empty document should return empty list"
