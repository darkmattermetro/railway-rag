import json
import logging
import re
import sys

import tiktoken
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

RETRIEVAL_K: int = 10
FAISS_SIM_THRESHOLD: float = 0.35
MAX_CONTEXT_TOKENS: int = 3500
RESERVE_TOKENS: int = 500
CHUNK_BUDGET: int = MAX_CONTEXT_TOKENS - RESERVE_TOKENS  # 3000
TOKEN_BUDGET: int = CHUNK_BUDGET  # Alias for fallback compatibility

_ENC = tiktoken.get_encoding("cl100k_base")

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

def mmr_diversity(documents: list[Document], lambda_: float = 0.5, max_docs: int = 8) -> list[Document]:
    selected: list[Document] = []
    candidates = list(documents)
    while len(selected) < max_docs and candidates:
        best_idx = -1
        best_score = -1.0
        for i, cand in enumerate(candidates):
            rel = 1.0 / (candidates.index(cand) + 1)
            if selected:
                max_sim = max(_word_overlap_ratio(cand.page_content, s.page_content) for s in selected)
            else:
                max_sim = 0.0
            mmr = lambda_ * rel - (1 - lambda_) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        if best_idx >= 0:
            selected.append(candidates.pop(best_idx))
    return selected


def parse_citations(text: str) -> tuple[str, list[int]]:
    """
    Parses <used_chunks>[0, 2]</used_chunks> from the text.
    Returns the cleaned text and the list of integer indices.
    """
    indices = []
    
    # Try looking for exactly <used_chunks>[X, Y]</used_chunks>
    match = re.search(r'<used_chunks>\s*\[(.*?)\]\s*</used_chunks>', text)
    if match:
        content = match.group(1)
        # Parse the comma-separated strings inside the brackets safely without try-except blocks
        parts = content.split(',')
        for p in parts:
            p = p.strip()
            if p.isdigit():
                indices.append(int(p))
                
    # Remove the XML tag block from final display
    clean_text = re.sub(r'<used_chunks>.*?</used_chunks>', '', text, flags=re.DOTALL)
    
    return clean_text.strip(), indices


RRF_CONSTANT: int = 60


def rrf_merge(faiss_results: list, bm25_results: list) -> list:
    merged: dict[str, dict] = {}
    for rank, (doc, score) in enumerate(faiss_results, 1):
        cid = doc.metadata.get("chunk_id") or str(id(doc))
        merged.setdefault(cid, {"doc": doc, "rrf": 0.0})
        merged[cid]["rrf"] += 1.0 / (RRF_CONSTANT + rank)
    for rank, (doc, score) in enumerate(bm25_results, 1):
        cid = doc.metadata.get("chunk_id") or str(id(doc))
        merged.setdefault(cid, {"doc": doc, "rrf": 0.0})
        merged[cid]["rrf"] += 1.0 / (RRF_CONSTANT + rank)
    sorted_docs = sorted(merged.values(), key=lambda x: x["rrf"], reverse=True)
    return [item["doc"] for item in sorted_docs]


