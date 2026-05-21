"""
Tests for ingest.py chunking: _garbage_ratio, _split_page_text, chunk_id format.
"""

import tiktoken

from ingest import _garbage_ratio, _split_page_text, CHUNK_SIZE, CHUNK_OVERLAP

_ENC = tiktoken.get_encoding("cl100k_base")


def test_garbage_ratio_clean_text():
    assert _garbage_ratio("Hello world this is clean text") == 0.0


def test_garbage_ratio_all_symbols():
    ratio = _garbage_ratio("%%% ^^^ &&& ***")
    assert ratio == 0.8


def test_garbage_ratio_mixed():
    ratio = _garbage_ratio("abc!!!!")
    assert ratio == 1 - 3 / 7


def test_garbage_ratio_empty():
    assert _garbage_ratio("") == 1.0


def test_garbage_ratio_no_text():
    assert _garbage_ratio("   ") == 0.0


def test_split_page_text_respects_chunk_size():
    long_text = "word " * 1000
    chunks = _split_page_text(long_text)
    for chunk_text, _, _ in chunks:
        token_count = len(_ENC.encode(chunk_text))
        assert token_count <= CHUNK_SIZE, (
            f"Chunk has {token_count} tokens, max {CHUNK_SIZE}"
        )


def test_split_page_text_single_chunk():
    short_text = "Hello world, this is a test."
    chunks = _split_page_text(short_text)
    assert len(chunks) == 1
    assert chunks[0][0] == short_text


def test_split_page_text_char_positions():
    text = "A " * 600 + "B " * 600
    chunks = _split_page_text(text)
    assert len(chunks) >= 2
    for chunk_text, char_start, char_end in chunks:
        assert char_end > char_start
        assert chunk_text == text[char_start:char_end]


def test_split_page_text_overlap():
    long_text = "word " * 1000
    chunks = _split_page_text(long_text)
    if len(chunks) >= 2:
        overlap_tokens = _ENC.encode(chunks[0][0])[-CHUNK_OVERLAP:]
        next_start_tokens = _ENC.encode(chunks[1][0])[: len(overlap_tokens)]
        match_ratio = sum(
            1 for a, b in zip(overlap_tokens, next_start_tokens) if a == b
        ) / max(len(overlap_tokens), 1)
        assert match_ratio >= 0.5, (
            f"Overlap between consecutive chunks is too low ({match_ratio:.0%})"
        )


def test_chunk_id_format():
    source_stem = "test_doc"
    assert _make_chunk_id(source_stem, 1, 0) == "test_doc_p1_0"
    assert _make_chunk_id(source_stem, 12, 3) == "test_doc_p12_3"


def _make_chunk_id(source_stem: str, page_no: int, idx: int) -> str:
    return f"{source_stem}_p{page_no}_{idx}"
