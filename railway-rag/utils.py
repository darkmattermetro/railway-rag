"""Shared utility functions for the Railway RAG pipeline."""

import json
import logging
import re
import sys
from dataclasses import dataclass

import tiktoken
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

RETRIEVAL_K: int = 20
MAX_CONTEXT_TOKENS: int = 3500
RESERVE_TOKENS: int = 500
CHUNK_BUDGET: int = MAX_CONTEXT_TOKENS - RESERVE_TOKENS  # 3000

FAISS_SIM_THRESHOLD: float = 0.35
FAISS_CONFIDENCE_THRESHOLD: float = 0.60
GARBAGE_RATIO_THRESHOLD: float = 0.40
EMBEDDING_RETRY_WAIT_SECS: int = 60

_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    source_file: str
    page_number: int
    chunk_id: str
    char_start: int
    char_end: int
    token_count: int


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def log_event(event: str, **kwargs) -> None:
    record = {"timestamp": __import__("time").time(), "event": event}
    record.update(kwargs)
    print(json.dumps(record, ensure_ascii=False), file=sys.stderr)


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


def process_citations(response: str, citation_map: dict) -> tuple[str, list[str]]:
    pattern = re.compile(r'\[(chunk_\d+)\]')
    matches = pattern.findall(response)

    ordered_ids: list[str] = []
    seen: set[str] = set()
    for m in matches:
        if m in citation_map:
            if m not in seen:
                seen.add(m)
                ordered_ids.append(m)
        else:
            logger.warning("event=hallucinated_citation chunk_id=%s", m)

    replacement = {cid: f"[Refer {i + 1}]" for i, cid in enumerate(ordered_ids)}

    def _replacer(m: re.Match) -> str:
        cid = m.group(1)
        return replacement.get(cid, "")

    processed = pattern.sub(_replacer, response).strip()
    return processed, ordered_ids
