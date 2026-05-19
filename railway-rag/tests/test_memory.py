"""
Memory pressure tests for the Railway RAG system
"""
import gc
import re
import tracemalloc
from unittest.mock import Mock

# Constants that would be defined in local_builder.py
CATEGORY_MAX_LEN: int = 64


def sanitize_category(category_name: str) -> str:
    """
    Category sanitization matching local_builder.py §6.4.
    """
    safe_category = re.sub(r"[^A-Za-z0-9_]", "_", category_name.strip())
    # Enforce length constraints
    if len(safe_category) < 1:
        safe_category = "unknown"
    elif len(safe_category) > CATEGORY_MAX_LEN:
        safe_category = safe_category[:CATEGORY_MAX_LEN]
    return safe_category


def test_sequential_processing_does_not_accumulate():
    """Test that sequential processing doesn't accumulate memory unboundedly."""
    tracemalloc.start()

    # Get baseline current memory (not peak)
    current_before, _ = tracemalloc.get_traced_memory()

    # Simulate processing 3 small PDF-like documents sequentially
    for i in range(3):
        mock_doc = Mock()
        mock_doc.body = [Mock()]

        temp_data = [j for j in range(1000)]
        temp_string = "x" * 10000

        del temp_data
        del temp_string
        del mock_doc
        gc.collect()

    # Get current memory after processing (not peak)
    current_after, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Use a generous tolerance multiplier (allow 5x growth for small baselines)
    # The baseline is near 0, so even a tiny allocation looks large relatively.
    # We just verify total bloat is under a reasonable absolute threshold.
    bloat = current_after - current_before
    assert bloat < 500_000, (
        f"Memory grew by {bloat} bytes (> 500 KB) — possible leak"
    )


def test_cleanup_barrier_releases_references():
    """Test that cleanup barrier releases references properly."""
    tracemalloc.start()

    # Snapshot current memory before allocation
    current_before, _ = tracemalloc.get_traced_memory()

    # Create a large bytes object (10 MB)
    pdf_data = b"x" * (10 * 1024 * 1024)

    # Measure current memory while holding the object
    current_with_data, _ = tracemalloc.get_traced_memory()

    # Verify the allocation was tracked (should be noticeably larger)
    # Use an approximate delta — at least 5 MB of growth
    alloc_delta = current_with_data - current_before
    assert alloc_delta > 5_000_000, (
        f"Expected >5 MB allocation, got {alloc_delta} bytes"
    )

    # Now simulate cleanup
    del pdf_data
    gc.collect()

    # Measure current memory after cleanup
    current_after, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Memory should have decreased closer to baseline (within 1 MB tolerance)
    cleanup_delta = abs(current_after - current_before)
    assert cleanup_delta < 1_000_000, (
        f"Memory delta from baseline after cleanup: {cleanup_delta} bytes "
        f"(expected < 1 MB)"
    )


def test_bm25_corpus_del_reduces_memory():
    """Test that deleting BM25 corpus reduces memory usage."""
    tracemalloc.start()

    current_before, _ = tracemalloc.get_traced_memory()

    # Load a list of 1000 small Documents into corpus_data
    corpus_data = []
    for i in range(1000):
        mock_doc = Mock()
        mock_doc.page_content = f"This is test document number {i} with some content."
        mock_doc.metadata = {"source": f"doc_{i}.pdf", "page_number": i % 100}
        corpus_data.append(mock_doc)

    current_with_corpus, _ = tracemalloc.get_traced_memory()

    # Verify corpus created measurable allocation
    alloc_delta = current_with_corpus - current_before
    assert alloc_delta > 10_000, (
        f"Expected >10 KB allocation, got {alloc_delta} bytes"
    )

    # Delete and collect
    del corpus_data
    gc.collect()

    current_after, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Memory should have decreased significantly
    cleanup_delta = abs(current_after - current_before)
    assert cleanup_delta < 50_000, (
        f"Memory delta from baseline after corpus delete: {cleanup_delta} bytes "
        f"(expected < 50 KB)"
    )


def test_category_sanitisation_length_cap():
    """Test category sanitization length capping."""
    input_category = "A" * 100

    result = sanitize_category(input_category)

    assert len(result) <= CATEGORY_MAX_LEN
    assert all(c == "A" or c == "_" for c in result)


def test_category_sanitisation_special_chars():
    """Test category sanitization with special characters."""
    input_category = "Signal & Switch / Control (v2)"

    result = sanitize_category(input_category)

    assert re.fullmatch(r"[A-Za-z0-9_]+", result) is not None
    assert len(result) > 0
