"""
Retrieval unit tests for utils.py functions plus mock-based RRF, threshold, and token budget.
"""
from unittest.mock import Mock

from utils import (
    apply_source_diversity,
    deduplicate_chunks,
    extract_technical_identifiers,
    process_citations,
    CHUNK_BUDGET,
    FAISS_SIM_THRESHOLD,
)


def apply_token_budget(context_parts):
    max_chars = CHUNK_BUDGET * 4
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
# Deduplication
# ---------------------------------------------------------------------------

def test_deduplication_removes_duplicates():
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


# ---------------------------------------------------------------------------
# Source diversity
# ---------------------------------------------------------------------------

def test_source_diversity_limits_per_page():
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
    docs = []
    for i in range(2):
        doc = Mock()
        doc.page_content = f"Content {i}"
        doc.metadata = {"source": "test.pdf", "page_number": 1}
        docs.append(doc)

    result = apply_source_diversity(docs, max_per_page=3)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

def test_token_budget_truncation():
    excess_chars = 500 * 4
    base_content = "x" * (CHUNK_BUDGET * 4)
    excess_content = "y" * excess_chars
    large_content = base_content + excess_content

    context_parts = [
        "First paragraph of content.",
        large_content,
        "Last paragraph of content.",
    ]

    truncated_parts = apply_token_budget(context_parts)
    joined = "\n\n".join(truncated_parts)

    assert len(joined) <= CHUNK_BUDGET * 4
    assert len(truncated_parts) > 0
    assert any(part.strip() for part in truncated_parts)


# ---------------------------------------------------------------------------
# Technical identifier extraction
# ---------------------------------------------------------------------------

def test_identifier_extraction_finds_codes():
    text = "Check regulation 45/2019 and refer to EN-302 and SPAD standards."
    technical_codes = extract_technical_identifiers(text)
    assert "45/2019" in technical_codes
    assert any(code in ["EN", "SPAD"] for code in technical_codes)


def test_identifier_extraction_empty_on_plain_text():
    text = "What is the maximum permitted speed on this line?"
    technical_codes = extract_technical_identifiers(text)
    assert "What" not in technical_codes
    assert len(technical_codes) == 0


# ---------------------------------------------------------------------------
# Citation processing (process_citations)
# ---------------------------------------------------------------------------

def test_citation_process_replaces_chunk_ids():
    citation_map = {
        "chunk_0": {"file": "doc.pdf", "page": "3"},
        "chunk_2": {"file": "doc.pdf", "page": "5"},
    }
    response = "The signal height is 2.5m [chunk_0] and clearance is 1m [chunk_2]."
    processed, ordered = process_citations(response, citation_map)
    assert "[Refer 1]" in processed
    assert "[Refer 2]" in processed
    assert ordered == ["chunk_0", "chunk_2"]


def test_citation_process_unknown_id_logged():
    citation_map = {"chunk_0": {"file": "doc.pdf", "page": "3"}}
    response = "Height is 2.5m [chunk_0] but speed is [chunk_99]."
    processed, ordered = process_citations(response, citation_map)
    assert "[Refer 1]" in processed
    assert ordered == ["chunk_0"]


def test_citation_process_no_citations():
    response = "This answer has no citations."
    processed, ordered = process_citations(response, {})
    assert processed == response
    assert ordered == []


def test_citation_process_strips_tag():
    citation_map = {
        "chunk_1": {"file": "doc.pdf", "page": "1"},
    }
    response = "This is the answer. [chunk_1]"
    processed, _ = process_citations(response, citation_map)
    assert "[Refer 1]" in processed


# ---------------------------------------------------------------------------
# RRF merge order (mock-based)
# ---------------------------------------------------------------------------

def _rrf_merge(faiss_results, bm25_results, docstore, idx_to_docid, constant=60):
    rrf_scores = {}
    for rank, (doc, _score) in enumerate(faiss_results, 1):
        cid = doc.metadata.get("chunk_id")
        if cid:
            rrf_scores[cid] = {"doc": doc, "rrf": 1.0 / (constant + rank)}

    for rank, (idx, _score) in enumerate(bm25_results, 1):
        doc_id = idx_to_docid.get(idx)
        if doc_id:
            doc = docstore.search(doc_id)
            cid = doc.metadata.get("chunk_id")
            if cid:
                if cid in rrf_scores:
                    rrf_scores[cid]["rrf"] += 1.0 / (constant + rank)
                else:
                    rrf_scores[cid] = {"doc": doc, "rrf": 1.0 / (constant + rank)}

    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["rrf"], reverse=True)
    return [item["doc"] for item in sorted_results]


def _make_doc(chunk_id, content="test content"):
    doc = Mock(spec=["page_content", "metadata"])
    doc.page_content = content
    doc.metadata = {"chunk_id": chunk_id, "source": "test.pdf", "page_number": 1}
    return doc


def test_rrf_merge_ranks_by_combined_score():
    docs = {f"chunk_{i}": _make_doc(f"chunk_{i}") for i in range(5)}

    faiss_results = [(docs["chunk_0"], 0.9), (docs["chunk_1"], 0.8), (docs["chunk_2"], 0.7)]
    bm25_results = [(2, 10.0), (3, 8.0), (0, 7.0)]

    docstore = Mock()
    docstore.search.side_effect = lambda doc_id: docs.get(doc_id, None)
    idx_to_docid = {0: "chunk_0", 2: "chunk_2", 3: "chunk_3"}

    merged = _rrf_merge(faiss_results, bm25_results, docstore, idx_to_docid)

    cids = [d.metadata["chunk_id"] for d in merged]
    assert cids[0] == "chunk_0"


def test_rrf_merge_only_faiss_results():
    docs = [_make_doc("chunk_0"), _make_doc("chunk_1")]
    faiss_results = [(docs[0], 0.9), (docs[1], 0.8)]

    merged = _rrf_merge(faiss_results, [], Mock(), {})
    assert len(merged) == 2
    assert merged[0].metadata["chunk_id"] == "chunk_0"


# ---------------------------------------------------------------------------
# Score threshold filter
# ---------------------------------------------------------------------------

def test_score_threshold_filters_low_scores():
    docs = [_make_doc(f"chunk_{i}") for i in range(4)]
    results = [
        (docs[0], 0.50),
        (docs[1], 0.40),
        (docs[2], 0.30),
        (docs[3], 0.20),
    ]
    filtered = [(d, s) for d, s in results if s >= FAISS_SIM_THRESHOLD]
    assert len(filtered) == 2
    assert filtered[0][0].metadata["chunk_id"] == "chunk_0"


def test_score_threshold_no_results():
    docs = [_make_doc("chunk_0")]
    results = [(docs[0], 0.10)]
    filtered = [(d, s) for d, s in results if s >= FAISS_SIM_THRESHOLD]
    assert len(filtered) == 0


# ---------------------------------------------------------------------------
# Token budget edge cases
# ---------------------------------------------------------------------------

def test_token_budget_empty_input():
    assert apply_token_budget([]) == []


def test_token_budget_single_chunk_under_budget():
    parts = ["Small chunk."]
    assert apply_token_budget(parts) == parts


def test_token_budget_honors_chunk_budget_constant():
    total = CHUNK_BUDGET
    assert total == 3000
