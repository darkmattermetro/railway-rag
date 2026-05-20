"""Shared utility functions for the Railway RAG pipeline."""

import logging
import re

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

RETRIEVAL_K: int = 20
TOKEN_BUDGET: int = 80_000


def _word_overlap_ratio(text1: str, text2: str) -> float:
    words1 = set(re.findall(r"\b[a-z0-9]+\b", text1.lower()))
    words2 = set(re.findall(r"\b[a-z0-9]+\b", text2.lower()))
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    return len(intersection) / max(len(words1), len(words2))


def extract_technical_identifiers(query: str) -> list[str]:
    codes: list[str] = re.findall(
        r"\b\d+/\d+\b|\b[A-Z]{2,5}\b|\b[A-Za-z]+-\d+[A-Za-z]?\b",
        query,
    )
    if codes:
        logger.info(
            "event=identifiers_extracted count=%d codes=%s",
            len(codes),
            codes,
        )
    return codes


def deduplicate_chunks(documents: list[Document], overlap_threshold: float = 0.7) -> list[Document]:
    unique_docs: list[Document] = []
    for doc in documents:
        is_duplicate = False
        for existing in unique_docs:
            if _word_overlap_ratio(doc.page_content, existing.page_content) > overlap_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_docs.append(doc)
    return unique_docs


def apply_source_diversity(documents: list[Document], max_per_page: int = 2) -> list[Document]:
    counts: dict[tuple[str, int], int] = {}
    result: list[Document] = []
    for doc in documents:
        key = (doc.metadata.get("source", ""), doc.metadata.get("page_number", 0))
        if counts.get(key, 0) < max_per_page:
            counts[key] = counts.get(key, 0) + 1
            result.append(doc)
    return result


def parse_citations(full_response: str) -> tuple[str, list[int]]:
    match = re.search(
        r"<used_chunks>\s*(\[.*?\])\s*</used_chunks>",
        full_response,
        re.DOTALL,
    )
    chunk_indices: list[int] = []
    if match:
        inner = match.group(1).strip()
        if inner.startswith("[") and inner.endswith("]"):
            inner = inner[1:-1]
            for part in inner.split(","):
                part = part.strip()
                if part and (part.isdigit() or (part.startswith("-") and part[1:].isdigit())):
                    chunk_indices.append(int(part))

    display_response = re.sub(
        r"<used_chunks>.*?</used_chunks>",
        "",
        full_response,
        flags=re.DOTALL,
    ).strip()

    return display_response, chunk_indices
