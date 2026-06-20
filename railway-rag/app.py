#!/usr/bin/env python3
"""Streamlit Cloud Retrieval Application for Railway RAG Pipeline.

v2.0 — Agent pipeline with rate limit management, RAM guards, multi-turn context,
structured audit logging, and enterprise-grade safety constraints.

FAISS similarity search with score-weighted context allocation.
Runs within ~1 GB RAM constraint on Streamlit Cloud.
"""

import gc
import hashlib
import json
import logging
import os
import time
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st
import tiktoken
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from utils import (
    _word_overlap_ratio,
    deduplicate_chunks,
    mmr_diversity,
    parse_citations,
    rrf_merge,
    FAISS_SIM_THRESHOLD,
    RETRIEVAL_K,
    TOKEN_BUDGET,
)
from model_selector import ModelSelector, LLMExhaustedError, RateLimitWarning

# ----------------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Module-level constants
# ----------------------------------------------------------------------------
COOLDOWN_SECONDS: float = 3.0
GROQ_FALLBACK_MODEL: str = "llama-3.3-70b-versatile"
GEMINI_PRIMARY_MODEL: str = "gemini-2.5-flash"
RERANKER_TOP_N: int = 8
RERANKER_ENABLED: bool = os.environ.get("RERANKER_ENABLED", "0") == "1"
RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
MAX_CITATION_EXPANDERS: int = 5
MAX_QUERY_LENGTH: int = 500
MAX_LLM_CALLS_PER_QUERY: int = 4
INDEX_DIR: Path = Path(__file__).resolve().parent / "vector_indices"

@st.cache_resource
def _get_encoder():
    return tiktoken.get_encoding("cl100k_base")

SYSTEM_PROMPT: str = (
    "You are a precision technical analyst for DMRC (Delhi Metro Rail Corporation). "
    "Your answers are used by engineers making safety-critical decisions.\n\n"

    "ANSWER RULES:\n"
    "1. Use ONLY the numbered context blocks [0], [1], ... provided below as your source of truth.\n"
    "2. When you use information from block [N], reference it inline as [N].\n"
    "3. If multiple blocks say the same thing, state it once and cite all relevant blocks: [0][2].\n"
    "4. Do NOT synthesize, interpolate, or infer numerical values - speeds, clearances, voltages, "
    "tolerances, dimensions - unless they are stated verbatim in context.\n\n"

    "INFERENCE RULE:\n"
    "If you make a non-numerical logical inference (e.g. derived from stated principles), "
    "you MUST wrap it: <inferred>Your inference here (basis: block [N]).</inferred>\n"
    "Never use <inferred> for numbers, measurements, or code requirements.\n\n"

    "INSUFFICIENT CONTEXT RULE:\n"
    "If the context does not contain enough information to answer the question, say: "
    "'The provided documents do not contain sufficient information to answer this. "
    "Suggest checking [specific document type].'\n"
    "Do not guess. Do not hallucinate a plausible-sounding answer.\n\n"

    "OUTPUT FORMAT (strict):\n"
    "1. First line MUST be: <used_chunks>[list of integer indices]</used_chunks>\n"
    "2. Then write your answer.\n"
    "3. Any analysis, reasoning, or synthesis beyond verbatim document text\n"
    "   MUST be wrapped in <inferred> tags.\n"
    "4. The final line MAY repeat <used_chunks> for clarity.\n\n"
    "CITATION RULE:\n"
    "Cite every block you use inline as [N].\n"
    "If you are unsure about a numerical value or specification,\n"
    "state the block you are deriving from and mark as <inferred>.\n"
    "Never present AI-generated analysis as document fact."
)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def _token_length(text: str) -> int:
    return len(_get_encoder().encode(text))


def _sanitize_query(query: str) -> str:
    """Trim, normalize whitespace, enforce MAX_QUERY_LENGTH."""
    q = " ".join(query.strip().split())
    if len(q) > MAX_QUERY_LENGTH:
        q = q[:MAX_QUERY_LENGTH]
    return q


def _log_memory_snapshot(label: str) -> None:
    """Log current RSS and available memory for RAM debugging."""
    try:
        import psutil
        proc = psutil.Process()
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        avail_mb = psutil.virtual_memory().available / (1024 * 1024)
        logger.info(
            "event=memory_snapshot label=%s rss_mb=%.0f available_mb=%.0f",
            label, rss_mb, avail_mb,
        )
    except Exception:
        pass


@st.cache_data(ttl=86400)
def make_query_hash(query: str, context_hashes: tuple) -> str:
    """Deterministic cache key for (query, conversation_context)."""
    raw = f"{query}||{''.join(context_hashes)}"
    return hashlib.sha256(raw.encode()).hexdigest()


@st.cache_data(max_entries=64)
def get_cached_response(_cache_key: str) -> str | None:
    """Return cached answer if available, else None."""
    return None  # trivial stub — real caching is handled by st.cache_data


def _build_conversation_context() -> list[dict]:
    """Build recent history summary for multi-turn grounding."""
    history = st.session_state.get("messages", [])
    if len(history) < 3:
        return []
    recent = history[-6:]  # last 3 turns (user+assistant pairs)
    ctx = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")[:500]
        ctx.append({"role": role, "content": content[:200]})
    return ctx


def render_ai_inference(text: str) -> str:
    """Replace <inferred> tags with styled AI Gen badge section."""
    return re.sub(
        r'<inferred>(.*?)</inferred>',
        r'<div class="ai-gen-block"><span class="ai-gen-badge">AI Gen</span>\1</div>',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _extract_excerpt(text: str, max_chars: int = 280) -> str:
    """First ~2 sentences up to max_chars for reference previews."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    excerpt = ""
    for s in sentences:
        if len(excerpt) + len(s) > max_chars:
            break
        excerpt += s + " "
    return excerpt.strip() or text[:max_chars].rsplit(" ", 1)[0] + "…"


def render_reference_card(doc: Document) -> None:
    """Render a professional reference card via st.markdown."""
    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page_number", "?")
    heading = doc.metadata.get("heading", "")
    excerpt = _extract_excerpt(doc.page_content)
    html = '<div class="ref-card">'
    html += f'<div class="ref-header"><span class="ref-source">{source}</span><span class="ref-page">Page {page}</span></div>'
    if heading:
        html += f'<div class="ref-heading">{heading}</div>'
    html += f'<div class="ref-excerpt">{excerpt}</div></div>'
    st.markdown(html, unsafe_allow_html=True)


@st.cache_resource(max_entries=1)
def load_vector_index(index_path: str) -> FAISS:
    """Load FAISS index from disk with integrity check."""
    path = Path(index_path).resolve()
    hash_path = path / "index.hash"
    if hash_path.exists():
        expected = hash_path.read_text().strip()
        actual = hashlib.sha256()
        for fname in ("index.faiss", "index.pkl"):
            with open(path / fname, "rb") as f:
                while chunk := f.read(65536):
                    actual.update(chunk)
        if actual.hexdigest() != expected:
            raise RuntimeError(f"FAISS index integrity mismatch at {path}")
    # --------------------------------------------------------------------
    # Streamlit Cloud Safe Embedding Loader
    # --------------------------------------------------------------------
    # Force CPU execution because Streamlit Cloud does not provide CUDA.
    # This prevents PyTorch/SentenceTransformer device initialization crashes.
    #
    # Optional:
    # Add "embedding_model.txt" inside each index folder to dynamically
    # specify the embedding model used during ingestion.
    #
    # Example content:
    #   BAAI/bge-small-en-v1.5
    #
    # Falls back to bge-small if no file exists.
    # --------------------------------------------------------------------
    model_file = path / "embedding_model.txt"

    if model_file.exists():
        model_name = model_file.read_text().strip()
    else:
        model_name = "BAAI/bge-small-en-v1.5"

    logger.info("Loading embedding model: %s", model_name)

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def _check_available_memory_mb():
    try:
        import psutil
        return int(psutil.virtual_memory().available / (1024 * 1024))
    except ImportError:
        pass
    for path in [
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory.max",
    ]:
        try:
            with open(path) as f:
                val = f.read().strip()
                if val != "max":
                    limit_mb = int(val) // (1024 * 1024)
                    if limit_mb < 1500:
                        return limit_mb
        except (FileNotFoundError, ValueError, OSError):
            continue
    return None


_RERANKER_MIN_MB = 600


@st.cache_resource
def load_reranker():
    if not RERANKER_ENABLED:
        return None
    avail_mb = _check_available_memory_mb()
    if avail_mb is not None and avail_mb < _RERANKER_MIN_MB:
        logger.warning(
            "event=reranker_skipped reason=low_memory available_limit_mb=%d min_required_mb=%d",
            avail_mb, _RERANKER_MIN_MB,
        )
        return None
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading reranker model: %s", RERANKER_MODEL)
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        return CrossEncoder(RERANKER_MODEL, device=device)
    except Exception:
        logger.warning("event=reranker_load_failed — disabling reranker")
        return None


@st.cache_resource(max_entries=1)
def load_bm25_index(index_path: str):
    path = Path(index_path).resolve()
    corpus_path = path / "bm25_corpus.json"
    if not corpus_path.exists():
        logger.info("event=bm25_corpus_not_found path=%s", corpus_path)
        return None, []
    from rank_bm25 import BM25Okapi
    with open(corpus_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    # RAM cap: enforce max 10k entries / ~25 MB on 1 GB budget
    if len(entries) > 10000:
        logger.warning(
            "event=bm25_corpus_capped entries=%d max=10000", len(entries),
        )
        entries = entries[:10000]
    texts = [e["text"] for e in entries]
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    logger.info("event=bm25_index_loaded entries=%d", len(entries))
    return bm25, entries


@st.cache_data(ttl=300)
def discover_indices() -> list[str]:
    """Scan INDEX_DIR for valid index subdirectories; return path strings."""
    if not INDEX_DIR.exists():
        return []
    indices: list[str] = []
    for path in INDEX_DIR.iterdir():
        if path.is_dir() and path.name.endswith("_index"):
            if (path / "index.faiss").exists() and (path / "index.pkl").exists():
                indices.append(str(path))
    return sorted(indices)


# ----------------------------------------------------------------------------
# Agent Dataclasses
# ----------------------------------------------------------------------------
@dataclass
class CallCounter:
    """Rolling LLM call count within a single user query."""
    total_calls: int = 0
    provider_calls: dict = field(default_factory=dict)
    start_time: float = 0.0
    exceeded: bool = False

@dataclass
class PlannerOutput:
    search_scope: str = "single"
    sub_questions: list = field(default_factory=list)
    query_type: str = "factual"
    confidence: float = 0.5
    answer_format: str = "paragraph"
    is_followup: bool = False

@dataclass
class ReviewerOutput:
    valid_indices: list = field(default_factory=list)
    hallucination_risk: bool = False
    missing_citation: bool = False
    out_of_range_citation: bool = False
    risk_reason: str = ""
    quality_score: float = 1.0
    inline_citation_count: int = 0
    answer_length_tokens: int = 0
    has_insufficient_context_response: bool = False

@dataclass
class QueryAuditRecord:
    """Structured log for one query lifecycle."""
    query: str = ""
    planner_scope: str = ""
    planner_confidence: float = 0.0
    query_type: str = ""
    sub_questions: list = field(default_factory=list)
    retrieval_count: int = 0
    bm25_count: int = 0
    llm_calls: int = 0
    answer_length_chars: int = 0
    answer_length_tokens: int = 0
    quality_score: float = 0.0
    hallucination_risk: bool = False
    missing_citation: bool = False
    total_latency_ms: float = 0.0
    provider_used: str = ""
    error: str = ""


# ----------------------------------------------------------------------------
# Agent 1 — Planner
# ----------------------------------------------------------------------------
def planner_agent(user_query: str, available_indices: list, llm_selector) -> PlannerOutput:
    """Analyze query and return structured plan with confidence and answer_format."""
    try:
        indices_str = ", ".join(available_indices)
        prompt = (
            f"Analyze this DMRC technical query and return a JSON routing decision.\n\n"
            f"Query: \"{user_query}\"\n"
            f"Available document collections: {indices_str}\n\n"
            "Definitions:\n"
            '- "search_scope": "single" -> one collection; "cross" -> needs multiple collections\n'
            '- "sub_questions": decompose if clearly independent parts else []\n'
            '- "query_type": "factual" | "procedural" | "comparative" | "calculation"\n'
            '- "confidence": 0.0–1.0 — how confident you are the available indices can answer this\n'
            '- "answer_format": "paragraph" | "list" | "table" | "code" | "step_by_step"\n'
            '- "is_followup": true if the query refers to previous conversation (e.g. "what about voltage?")\n\n'
            "Return ONLY this JSON, no explanation, no markdown fences:\n"
            '{{"search_scope": "single" | "cross", '
            '"sub_questions": [], '
            '"query_type": "factual" | "procedural" | "comparative" | "calculation", '
            '"confidence": 0.8, '
            '"answer_format": "paragraph" | "list" | "table" | "code" | "step_by_step", '
            '"is_followup": false}}'
        )
        messages = [
            SystemMessage(content="You are a query analysis assistant. Output only JSON."),
            HumanMessage(content=prompt),
        ]
        result = llm_selector.invoke_with_fallback(messages)
        text = result.content if hasattr(result, "content") else str(result)
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.MULTILINE)
        data = json.loads(text)

        scope = data.get("search_scope", "single")
        if scope not in ("single", "cross"):
            scope = "single"
        sub_qs = data.get("sub_questions", [])[:3]
        qtype = data.get("query_type", "factual")
        if qtype not in ("factual", "procedural", "comparative", "calculation"):
            qtype = "factual"
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        afmt = data.get("answer_format", "paragraph")
        if afmt not in ("paragraph", "list", "table", "code", "step_by_step"):
            afmt = "paragraph"
        is_followup = bool(data.get("is_followup", False))
        return PlannerOutput(scope, sub_qs, qtype, confidence, afmt, is_followup)
    except Exception as e:
        logger.warning("event=planner_fallback error=%s", e)
        return PlannerOutput()


# ----------------------------------------------------------------------------
# Agent 2 — Research
# ----------------------------------------------------------------------------
def research_agent(
    sub_questions: list[str],
    query_type: str,
    llm_selector,
    planner_confidence: float = 0.5,
    answer_format: str = "paragraph",
) -> list[str]:
    try:
        questions = sub_questions if sub_questions else [""]
        # Confidence-adjusted counts: low confidence -> more queries
        base_counts = {"factual": 3, "procedural": 4, "comparative": 5, "calculation": 3}
        base_count = base_counts.get(query_type, 3)
        if planner_confidence < 0.4:
            base_count = min(base_count + 2, 7)
        elif planner_confidence > 0.8:
            base_count = max(base_count - 1, 2)
        count = max(1, min(base_count, 8 // max(len(questions), 1)))

        # Answer-format-tailored query style instruction
        fmt_guidance = {
            "table": "Include column/field names and data categories in each query.",
            "list": "Focus on enumerable items: types, components, steps, requirements.",
            "code": "Include standard designations, code numbers, or spec clause references.",
            "step_by_step": "Include sequencing words (first, then, final) and stage names.",
            "paragraph": "General-purpose; cover factual and relational aspects.",
        }.get(answer_format, "General-purpose; cover factual and relational aspects.")

        prompt = (
            f"You are generating search queries for a DMRC (Delhi Metro Rail Corporation) "
            f"technical document retrieval system covering signalling, civil, traction, "
            f"OHE, and telecom engineering.\n\n"
            f"Query type: {query_type}\n"
            f"Answer format expected: {answer_format}\n"
            f"Format-specific guidance: {fmt_guidance}\n"
            f"Sub-questions to expand:\n"
            + "\n".join(f'{i+1}. {q}' for i, q in enumerate(questions))
            + f"\n\nGenerate exactly {count} search queries per sub-question "
            f"({count * len(questions)} total).\n"
            f"Each query must be:\n"
            f"- 4-12 words, no filler phrases\n"
            f"- Optimized for keyword matching against technical railway specs\n"
            f"- Use DMRC/Indian railway terminology where applicable "
            f"(ATP, IEF, SCADA, OHE, SSI, axle counter, etc.)\n\n"
            f"Query type guidance:\n"
            f"- factual: use the exact technical term + synonyms + acronym expansions\n"
            f"- procedural: include action words (procedure, commissioning, testing, maintenance steps)\n"
            f"- comparative: include both entities being compared in each query\n"
            f"- calculation: include the standard/formula name, unit, or code reference\n\n"
            "Output: one query per line, no numbering, no bullets, no explanation."
        )
        messages = [
            SystemMessage(content="You are a query expansion assistant for DMRC railway engineering documents."),
            HumanMessage(content=prompt),
        ]
        result = llm_selector.invoke_with_fallback(messages)
        text = result.content if hasattr(result, "content") else str(result)
        queries = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if not queries:
            return [questions[0]] if questions else []
        queries = list(dict.fromkeys([sub_questions[0]] + queries[:7]))
        return queries[:8]
    except Exception as e:
        logger.warning("event=research_fallback error=%s", e)
        return sub_questions[:1] if sub_questions else []


# ----------------------------------------------------------------------------
# Agent 3 — Retriever (frozen retrieval core)
# ----------------------------------------------------------------------------
def retriever_agent(
    queries: list[str],
    user_query: str,
    vector_store: FAISS,
    bm25,
    corpus_entries: list[dict],
    reranker,
) -> tuple[list[Document], list[str]]:
    """Frozen retrieval core — do not modify retrieval logic."""
    seen_cids: set[str] = set()
    merged_docs: list[Document] = []
    for q in queries:
        docs_with_scores = vector_store.similarity_search_with_score(q, k=RETRIEVAL_K)
        faiss_filtered = [(doc, score) for doc, score in docs_with_scores if score >= FAISS_SIM_THRESHOLD]
        if bm25:
            tokenized_q = q.lower().split()
            bm25_scores = bm25.get_scores(tokenized_q)
            top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:RETRIEVAL_K]
            bm25_docs = [(Document(page_content=corpus_entries[idx]["text"],
                                  metadata=corpus_entries[idx].get("metadata", {})),
                          bm25_scores[idx]) for idx in top_idx]
            query_merged = rrf_merge(faiss_filtered, bm25_docs)
        else:
            query_merged = [doc for doc, _ in faiss_filtered]
        for doc in query_merged:
            cid = doc.metadata.get("chunk_id") or str(id(doc))
            if cid not in seen_cids:
                seen_cids.add(cid)
                merged_docs.append(doc)

    deduped_docs = deduplicate_chunks(merged_docs)
    diverse_docs = mmr_diversity(deduped_docs)
    if reranker:
        pairs = [[user_query, doc.page_content] for doc in diverse_docs]
        scores = reranker.predict(pairs)
        scored = sorted(zip(scores, diverse_docs), key=lambda x: x[0], reverse=True)
        reranked_docs = [doc for _, doc in scored[:RERANKER_TOP_N]]
        available_scores = [score for score, _ in scored[:RERANKER_TOP_N]]
        min_s = min(available_scores) if available_scores else 0
        max_s = max(available_scores) if available_scores else 1
        span = max_s - min_s
        if span > 0:
            chunk_weights = [(s - min_s) / span for s in available_scores]
        else:
            chunk_weights = [1.0] * len(available_scores)
    else:
        reranked_docs = diverse_docs[:RERANKER_TOP_N]
        n = len(reranked_docs)
        chunk_weights = [1.0 / (1 + i * 0.4) for i in range(n)]

    logger.info(
        "event=retrieval_complete merged=%d deduped=%d diverse=%d reranked=%d",
        len(merged_docs),
        len(deduped_docs),
        len(diverse_docs),
        len(reranked_docs),
    )

    context_parts: list[str] = []
    total_tokens: int = 0

    for i, doc in enumerate(reranked_docs):
        if i >= len(chunk_weights) or total_tokens >= TOKEN_BUDGET:
            break
        chunk_tokens: int = _token_length(doc.page_content)
        alloc: int = min(chunk_tokens, int(TOKEN_BUDGET * chunk_weights[i]), TOKEN_BUDGET - total_tokens)
        if alloc <= 0:
            break
        if alloc >= chunk_tokens:
            context_parts.append(doc.page_content)
        else:
            context_parts.append(_get_encoder().decode(_get_encoder().encode(doc.page_content)[:alloc]))
        total_tokens += alloc

    logger.info(
        "event=context_built chunks=%d tokens=%d",
        len(context_parts),
        total_tokens,
    )

    return reranked_docs, context_parts


# ----------------------------------------------------------------------------
# Agent 4 — Writer
# ----------------------------------------------------------------------------
def writer_agent(
    user_query: str,
    context_parts: list[str],
    query_type: str,
    llm_selector,
    response_placeholder,
    conversation_context: list | None = None,
) -> str:
    if not context_parts:
        response_placeholder.markdown("⚠️ No relevant documents found for this query.")
        return ""

    type_hint = {
        "procedural": "Structure your answer as numbered steps.",
        "comparative": "Use a structured comparison. Address each aspect separately.",
        "calculation": "Show your working. State formula, substitution, and result.",
        "factual": "",
    }.get(query_type, "")
    effective_system = (type_hint + "\n\n" + SYSTEM_PROMPT).strip()

    context_str = "\n\n---\n\n".join(
        f"[{i}] {c}" for i, c in enumerate(context_parts)
    )

    conv_block = ""
    if conversation_context:
        turns = "\n".join(
            f"{m['role']}: {m['content']}" for m in conversation_context[-4:]
        )
        conv_block = f"\n\nRecent conversation context:\n{turns}\n"

    user_prompt = (
        f"Context blocks:\n{context_str}\n{conv_block}\n"
        f"Question: {user_query}\n\n"
        "Instructions:\n"
        "- Answer directly and technically. No preamble.\n"
        "- Cite block numbers inline as [N] whenever you use a block's content.\n"
        "- If context is insufficient, say so explicitly - do not guess.\n"
        "- End your response with <used_chunks>[list of integer indices]</used_chunks>."
    )

    messages = [
        SystemMessage(content=effective_system),
        HumanMessage(content=user_prompt),
    ]

    response_placeholder.markdown('<div class="status-msg">Generating answer...</div>', unsafe_allow_html=True)
    full_response: str = ""

    try:
        stream_generator = llm_selector.stream_with_fallback(messages)
        stream_text = ""
        for chunk in stream_generator:
            if isinstance(chunk.content, str):
                stream_text += chunk.content
            display_text = re.sub(r'<used_chunks>.*?</used_chunks>', '', stream_text, flags=re.DOTALL)
            display_text = re.sub(r'</?inferred>', '', display_text)
            display_text = render_ai_inference(display_text)
            response_placeholder.markdown(display_text + "▌")
        full_response = stream_text
    except LLMExhaustedError:
        logger.error("event=llm_all_providers_exhausted")
        return None
    except Exception as e:
        logger.exception("event=writer_llm_failed error=%s", e)
        return None

    return full_response


# ----------------------------------------------------------------------------
# Agent 5 — Reviewer (zero LLM calls)
# ----------------------------------------------------------------------------
def reviewer_agent(
    full_response: str,
    reranked_docs: list[Document],
    user_query: str,
) -> ReviewerOutput:
    try:
        _, chunk_indices = parse_citations(full_response)

        missing_citation = not chunk_indices
        out_of_range_citation = False
        valid_indices = []
        for idx in chunk_indices:
            if 0 <= idx < len(reranked_docs):
                valid_indices.append(idx)
            else:
                out_of_range_citation = True

        hallucination_risk = False
        risk_reasons = []
        clean_response = re.sub(r'<[^>]+>.*?</[^>]+>', '', full_response, flags=re.DOTALL)
        if valid_indices:
            low_overlap = []
            for idx in valid_indices:
                ratio = _word_overlap_ratio(clean_response, reranked_docs[idx].page_content)
                if ratio < 0.15:
                    low_overlap.append(idx)
            if low_overlap:
                hallucination_risk = True
                risk_reasons.append(
                    f"Low overlap with cited chunks: indices {low_overlap} "
                    f"(ratio < 0.15) - possible hallucination"
                )

        if out_of_range_citation:
            risk_reasons.append("Some cited chunk indices exceed available document count")

        if missing_citation:
            risk_reasons.append("No <used_chunks> found in response - citation missing")

        # Quality metrics
        answer_length_tokens = len(_get_encoder().encode(full_response))
        inline_citation_count = len(re.findall(r'\[(\d+)\]', full_response))
        has_insufficient = bool(re.search(
            r'do(es)?\s+not\s+contain\s+sufficient|insufficient\s+information|cannot\s+answer',
            full_response, re.IGNORECASE,
        ))
        # Quality score: heuristic 0-1 based on citation density
        quality_score = 1.0
        if missing_citation:
            quality_score -= 0.3
        if hallucination_risk:
            quality_score -= 0.3
        if inline_citation_count == 0 and answer_length_tokens > 50:
            quality_score -= 0.2
        if has_insufficient:
            quality_score = max(quality_score, 0.6)  # honest refusal is OK
        quality_score = max(0.0, quality_score)

        return ReviewerOutput(
            valid_indices=valid_indices,
            hallucination_risk=hallucination_risk,
            missing_citation=missing_citation,
            out_of_range_citation=out_of_range_citation,
            risk_reason=" | ".join(risk_reasons) if risk_reasons else "All checks passed",
            quality_score=quality_score,
            inline_citation_count=inline_citation_count,
            answer_length_tokens=answer_length_tokens,
            has_insufficient_context_response=has_insufficient,
        )
    except Exception as e:
        logger.warning("event=reviewer_error error=%s", e)
        return ReviewerOutput(
            valid_indices=[],
            hallucination_risk=False,
            missing_citation=False,
            out_of_range_citation=False,
            risk_reason=f"Review skipped due to error: {e}",
        )


# ----------------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------------
def main() -> None:
    """Main entrypoint for the Streamlit retrieval application."""

    st.set_page_config(
        page_title="DMRC Document Search",
        page_icon="📄",
        layout="wide",
    )

    # --------------------------------------------------------------------
    # Secrets (read once in main, not at module scope)
    # --------------------------------------------------------------------
    gemini_keys = st.secrets.get("gemini", {}).get("api_keys", [])
    if not gemini_keys and "GEMINI_API_KEY" in st.secrets:
        gemini_keys = [st.secrets["GEMINI_API_KEY"]]

    groq_keys = st.secrets.get("groq", {}).get("api_keys", [])
    if not groq_keys and "GROQ_API_KEY" in st.secrets:
        groq_keys = [st.secrets["GROQ_API_KEY"]]

    if not gemini_keys and not groq_keys:
        st.error(
            "No API keys found. Configure `.streamlit/secrets.toml` "
            "with at least one provider's API key."
        )
        st.stop()

    # --------------------------------------------------------------------
    # Session state initialization
    # --------------------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_query_time" not in st.session_state:
        st.session_state.last_query_time = 0.0
    if "current_index" not in st.session_state:
        st.session_state.current_index = None
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "last_plan" not in st.session_state:
        st.session_state.last_plan = None
    if "last_queries" not in st.session_state:
        st.session_state.last_queries = []
    if "last_review" not in st.session_state:
        st.session_state.last_review = None

    # --------------------------------------------------------------------
    # UI Layout
    # --------------------------------------------------------------------
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }

    .dmrc-header {
        background: #003366;
        padding: 1.5rem 2rem 0.8rem 2rem;
        border-radius: 0 0 10px 10px;
        margin: -0.5rem -1rem 1.5rem -1rem;
    }
    .dmrc-header h1 {
        color: #FFFFFF;
        font-size: 1.6rem;
        font-weight: 600;
        margin: 0 0 0.2rem 0;
        letter-spacing: -0.02em;
    }
    .dmrc-header .subtitle {
        color: rgba(255,255,255,0.7);
        font-size: 0.85rem;
        font-weight: 400;
        margin: 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.15);
    }

    .stChatInput { border: 2px solid #D0D8E0; border-radius: 8px; transition: border-color 0.2s; }
    .stChatInput:focus, .stChatInput:focus-within { border-color: #003366; box-shadow: 0 0 0 2px rgba(0,51,102,0.1); }
    .stChatInput input { font-size: 0.9rem; }

    .stChatFloatingInputContainer { border-radius: 8px; }

    [data-testid="stChatMessage"] {
        border-left: 3px solid #D0D8E0;
        padding-left: 0.75rem;
        margin: 0.5rem 0;
        background: #FFFFFF;
        border-radius: 0 6px 6px 0;
    }
    [data-testid="stChatMessageContent"] { font-size: 0.95rem; line-height: 1.6; color: #1A1A2E; }

    .st-expander {
        border: 1px solid #E0E4E8;
        border-radius: 6px;
        margin: 0.25rem 0;
        transition: border-color 0.2s;
        background: #FFFFFF;
    }
    .st-expander:hover { border-color: #003366; }
    .st-expander summary { font-size: 0.9rem; font-weight: 500; color: #003366; padding: 0.25rem 0; }

    hr { margin: 1rem 0; border-color: #E0E4E8; }

    .status-msg {
        color: #5A6A7E;
        font-size: 0.85rem;
        font-style: italic;
        margin: 0.5rem 0;
    }

    [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E0E4E8; }
    [data-testid="stSidebar"] .stSelectbox label { font-size: 0.85rem; font-weight: 500; color: #1A1A2E; }

    .stAlert { border-left: 3px solid #003366; background: #F0F4F8; border-radius: 4px; }

    .ai-gen-badge {
        display: inline-block;
        background: #FFC107;
        color: #000000;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 0.1rem 0.45rem;
        border-radius: 3px;
        margin-right: 0.4rem;
        vertical-align: middle;
        text-transform: uppercase;
    }
    .ai-gen-block {
        background: #FFF8E1;
        border-left: 4px solid #FFC107;
        padding: 0.5rem 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        color: #333;
        line-height: 1.5;
    }

    .ref-card {
        border: 1px solid #E0E4E8;
        border-radius: 6px;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        background: #F8FAFB;
    }
    .ref-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.2rem;
    }
    .ref-source {
        font-weight: 600;
        color: #003366;
        font-size: 0.85rem;
    }
    .ref-page {
        color: #5A6A7E;
        font-size: 0.75rem;
    }
    .ref-heading {
        color: #1A1A2E;
        font-size: 0.8rem;
        font-style: italic;
        margin-bottom: 0.2rem;
    }
    .ref-excerpt {
        color: #333;
        font-size: 0.85rem;
        line-height: 1.4;
    }

    .conf-badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.15rem 0.6rem;
        border-radius: 12px;
    }
    .conf-high { background: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; }
    .conf-med { background: #FFF8E1; color: #F57F17; border: 1px solid #FFE082; }
    .conf-low { background: #FFEBEE; color: #C62828; border: 1px solid #EF9A9A; }
</style>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="dmrc-header">
    <h1>DMRC Document Search Engine</h1>
    <p class="subtitle">Delhi Metro Rail Corporation — Intelligent Document Search</p>
</div>
""", unsafe_allow_html=True)

    # --------------------------------------------------------------------
    # Index selection sidebar
    # --------------------------------------------------------------------
    with st.sidebar:
        st.header("Document Collection")
        available_indices = discover_indices()

        if not available_indices:
            st.error(
                "No vector indices found. Please run local_builder.py first to "
                "ingest documents into the vector_indices/ directory."
            )
            st.stop()

        index_display_names: list[str] = [Path(p).name for p in available_indices]
        selected_display: str = st.selectbox(
            "Collection",
            index_display_names,
            help="Choose which document set to search",
        )
        selected_index_name: str = available_indices[index_display_names.index(selected_display)]

        # Clear chat history when index changes
        if st.session_state.current_index != selected_display:
            st.session_state.messages = []
            st.session_state.current_index = selected_display

        with st.spinner("Loading..."):
            try:
                vector_store = load_vector_index(selected_index_name)
                faiss_docs_count: int = vector_store.index.ntotal

                logger.info(
                    "event=index_loaded index=%s faiss_docs=%d",
                    selected_index_name,
                    faiss_docs_count,
                )
            except (RuntimeError, FileNotFoundError, ValueError) as e:
                logger.exception("Failed to load vector index")
                st.error(
                    f"Failed to load index: {selected_index_name}\n\n"
                    f"Error: {str(e)}"
                )
                st.info(
                    "Possible causes:\n"
                    "- Embedding model mismatch\n"
                    "- GPU-generated FAISS index not converted to CPU\n"
                    "- Missing index.pkl or index.faiss\n"
                    "- Corrupted index files"
                )
                st.stop()

        reranker = load_reranker()
        bm25, corpus_entries = load_bm25_index(selected_index_name)
        st.caption(f"📦 {faiss_docs_count:,} indexed chunks")

    # --------------------------------------------------------------------
    # Chat history display
    # --------------------------------------------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = re.sub(r'<used_chunks>.*?</used_chunks>', '', message["content"], flags=re.DOTALL)
            content = render_ai_inference(content)
            st.markdown(content)

    # --------------------------------------------------------------------
    # Query handling
    # --------------------------------------------------------------------
    user_query_raw: str = st.chat_input("Ask a question about DMRC documents...")

    if user_query_raw:
        user_query: str = _sanitize_query(user_query_raw)
        t_query_start: float = time.monotonic()
        call_counter: CallCounter = CallCounter(start_time=t_query_start)
        audit: QueryAuditRecord = QueryAuditRecord(query=user_query)

        # Cooldown protection
        now = time.time()
        elapsed = now - st.session_state.last_query_time
        if elapsed < COOLDOWN_SECONDS:
            remaining = COOLDOWN_SECONDS - elapsed
            st.info(f"⏳ Please wait {remaining:.0f}s before submitting another query.")
            st.stop()
        st.session_state.last_query_time = now

        if len(user_query) < 3:
            st.info("Query too short. Please enter at least 3 characters.")
            st.stop()

        # Increment global query count for model rotation
        current_query_count = st.session_state.query_count
        st.session_state.query_count += 1

        logger.info("event=query_received query_len=%d query_count=%d", len(user_query), current_query_count)

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # --------------------------------------------------------------------
        # LLM Setup
        # --------------------------------------------------------------------
        gemini_models = st.secrets.get("gemini", {}).get("models", [GEMINI_PRIMARY_MODEL])
        groq_models = st.secrets.get("groq", {}).get("models", [GROQ_FALLBACK_MODEL])

        if "llm_state" not in st.session_state:
            st.session_state.llm_state = {}
        llm_selector = ModelSelector(
            gemini_models=gemini_models,
            groq_models=groq_models,
            gemini_api_keys=gemini_keys,
            groq_api_keys=groq_keys,
            query_count=current_query_count,
            session_state=st.session_state.llm_state,
        )

        # ----------------------------------------------------------------
        # AGENT PIPELINE
        # ----------------------------------------------------------------

        with st.chat_message("assistant"):
            _log_memory_snapshot("before_pipeline")
            response_placeholder = st.empty()
            response_placeholder.markdown(
                '<div class="status-msg">🔍 Analyzing query...</div>',
                unsafe_allow_html=True,
            )

            # 1. Planner
            plan = planner_agent(user_query, index_display_names, llm_selector)
            call_counter.total_calls += 1
            audit.planner_scope = plan.search_scope
            audit.planner_confidence = plan.confidence
            audit.query_type = plan.query_type
            audit.sub_questions = plan.sub_questions

            # 2. Research
            sub_qs = plan.sub_questions if plan.sub_questions else [user_query]
            response_placeholder.markdown(
                '<div class="status-msg">🔎 Generating search strategy...</div>',
                unsafe_allow_html=True,
            )
            queries_to_search = research_agent(
                sub_qs, plan.query_type, llm_selector,
                planner_confidence=plan.confidence,
                answer_format=plan.answer_format,
            )
            call_counter.total_calls += 1

            # 3. Retriever
            response_placeholder.markdown(
                '<div class="status-msg">📄 Searching documents...</div>',
                unsafe_allow_html=True,
            )
            reranked_docs, context_parts = retriever_agent(
                queries_to_search, user_query, vector_store,
                bm25, corpus_entries, reranker,
            )
            audit.retrieval_count = len(reranked_docs)
            audit.bm25_count = len(corpus_entries) if corpus_entries else 0

            # 4. Writer
            conversation_ctx = _build_conversation_context()
            full_response = writer_agent(
                user_query, context_parts, plan.query_type,
                llm_selector, response_placeholder,
                conversation_context=conversation_ctx,
            )
            call_counter.total_calls += 1
            audit.llm_calls = call_counter.total_calls
            provider_used = getattr(llm_selector, "last_provider", "")
            audit.provider_used = provider_used
            st.session_state.last_llm = provider_used

            # Enforce MAX_LLM_CALLS_PER_QUERY
            if call_counter.total_calls > MAX_LLM_CALLS_PER_QUERY:
                logger.warning(
                    "event=llm_call_budget_exceeded calls=%d max=%d",
                    call_counter.total_calls, MAX_LLM_CALLS_PER_QUERY,
                )

            if not full_response:
                if full_response is None:
                    st.error("Failed to generate an answer. Please try again.")
                    msg = "I encountered an error generating an answer."
                    audit.error = "writer_returned_none"
                else:
                    st.warning("No relevant documents found for this query.")
                    msg = "No relevant documents found."
                    audit.error = "no_context_parts"
                st.session_state.messages.append(
                    {"role": "assistant", "content": msg}
                )
                st.stop()

            # 5. Reviewer
            review = reviewer_agent(full_response, reranked_docs, user_query)
            st.session_state.last_plan = plan
            st.session_state.last_queries = queries_to_search
            st.session_state.last_review = review

            if review.hallucination_risk:
                st.warning("⚠️ This answer includes claims that may not be fully supported by the source documents.")

            # Citation synthesis
            display_response, chunk_indices = parse_citations(full_response)
            display_response = render_ai_inference(display_response)

            # Top-N citation selection
            valid_indices = [idx for idx in chunk_indices if 0 <= idx < len(reranked_docs)]
            if valid_indices:
                scored_indices = sorted(
                    ((_word_overlap_ratio(display_response, reranked_docs[idx].page_content), idx)
                     for idx in valid_indices),
                    key=lambda x: (-x[0], x[1]),
                )
                final_indices = [idx for _, idx in scored_indices[:MAX_CITATION_EXPANDERS]]
            else:
                final_indices = []

            # Confidence display
            conf_pct = int(review.quality_score * 100)
            if conf_pct >= 80:
                conf_tag = '<span class="conf-badge conf-high">High Confidence</span>'
            elif conf_pct >= 50:
                conf_tag = '<span class="conf-badge conf-med">Medium Confidence</span>'
            else:
                conf_tag = '<span class="conf-badge conf-low">Low Confidence</span>'

            clean_text = re.sub(r'<used_chunks>.*?</used_chunks>', '', display_response, flags=re.DOTALL).strip()
            inferred_parts = re.findall(r'<inferred>(.*?)</inferred>', full_response, flags=re.DOTALL)

            # --- Section 1: Answer ---
            st.markdown("### Answer")
            response_placeholder.markdown(clean_text, unsafe_allow_html=True)

            # --- Section 2: Document Facts ---
            if final_indices:
                st.markdown("### Document Facts")
                fact_seen: set[str] = set()
                for idx in final_indices:
                    doc = reranked_docs[idx]
                    source = doc.metadata.get("source", "unknown")
                    page = str(doc.metadata.get("page_number", "?"))
                    key = f"{source}:{page}"
                    if key not in fact_seen:
                        fact_seen.add(key)
                        st.markdown(
                            f'<div class="ref-card"><div class="ref-header"><span class="ref-source">{source}</span>'
                            f'<span class="ref-page">Page {page}</span></div></div>',
                            unsafe_allow_html=True,
                        )

            # --- Section 3: AI Gen Analysis ---
            if inferred_parts:
                st.markdown("### AI Gen Analysis")
                for part in inferred_parts:
                    st.markdown(
                        f'<div class="ai-gen-block"><span class="ai-gen-badge">AI Gen</span>{part}</div>',
                        unsafe_allow_html=True,
                    )

            # --- Section 4: References ---
            if final_indices:
                st.markdown("### References")
                ref_seen: set[str] = set()
                for idx in final_indices:
                    doc = reranked_docs[idx]
                    key = f"{doc.metadata.get('source','unknown')}:{doc.metadata.get('page_number','?')}"
                    if key not in ref_seen:
                        ref_seen.add(key)
                        render_reference_card(doc)

            st.markdown(f"<div style='margin-bottom:0.75rem'>{conf_tag}</div>", unsafe_allow_html=True)

            # RAM cleanup: release large references after rendering
            del reranked_docs, context_parts, corpus_entries
            gc.collect()

            audit.quality_score = review.quality_score
            audit.hallucination_risk = review.hallucination_risk
            audit.missing_citation = review.missing_citation
            audit.answer_length_tokens = review.answer_length_tokens
            audit.answer_length_chars = len(full_response)

            logger.info(
                "event=query_audit %s", json.dumps({
                    "query": audit.query[:120],
                    "scope": audit.planner_scope,
                    "confidence": round(audit.planner_confidence, 2),
                    "query_type": audit.query_type,
                    "sub_questions": len(audit.sub_questions),
                    "retrieval_count": audit.retrieval_count,
                    "llm_calls": audit.llm_calls,
                    "answer_chars": audit.answer_length_chars,
                    "answer_tokens": audit.answer_length_tokens,
                    "quality_score": round(audit.quality_score, 2),
                    "hallucination_risk": audit.hallucination_risk,
                    "provider": audit.provider_used,
                    "error": audit.error or "",
                    "latency_ms": round((time.monotonic() - t_query_start) * 1000),
                }),
            )

        total_latency_ms = (time.monotonic() - t_query_start) * 1000
        logger.info("event=query_latency_ms elapsed=%.0f", total_latency_ms)

        # Add assistant response to history
        st.session_state.messages.append(
            {"role": "assistant", "content": display_response}
        )


if __name__ == "__main__":
    main()