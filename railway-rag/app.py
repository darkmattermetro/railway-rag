#!/usr/bin/env python3
"""Streamlit Cloud Retrieval Application for Railway RAG Pipeline.

Hybrid retrieval with FAISS + BM25 (RRF merge), FlashRank reranking, and LLM generation.
Runs within ~1 GB RAM constraint on Streamlit Cloud.
"""

import gc
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tiktoken
from flashrank import Ranker, RerankRequest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from model_selector import ModelSelector, LLMExhaustedError
from utils import (
    apply_source_diversity,
    count_tokens,
    deduplicate_chunks,
    process_citations,
    CHUNK_BUDGET,
    FAISS_SIM_THRESHOLD,
    FAISS_CONFIDENCE_THRESHOLD,
    RETRIEVAL_K,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

COOLDOWN_SECONDS: float = 1.0
GROQ_MODEL: str = "llama-3.3-70b-versatile"
GROQ_TIMEOUT: int = 45
GEMINI_FALLBACK_MODEL: str = "gemini-1.5-flash"
RERANKER_MODEL: str = "ms-marco-MiniLM-L-12-v2"
RERANKER_TOP_N: int = 20
RRF_CONSTANT: int = 60
INDEX_DIR: Path = Path(__file__).resolve().parent / "vector_indices"
_ENC = tiktoken.get_encoding("cl100k_base")


@st.cache_resource
def load_indices(category: str) -> tuple:
    """Load FAISS index and BM25 model for a category. Cached."""
    index_path = INDEX_DIR / category

    # FAISS
    hash_path = index_path / "index.hash"
    if hash_path.exists():
        expected = hash_path.read_text().strip()
        actual = hashlib.sha256()
        actual.update((index_path / "index.faiss").read_bytes())
        actual.update((index_path / "index.pkl").read_bytes())
        if actual.hexdigest() != expected:
            raise RuntimeError(f"FAISS index integrity mismatch at {index_path}")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vector_store = FAISS.load_local(
        str(index_path), embeddings, allow_dangerous_deserialization=True
    )

    # BM25
    corpus_path = index_path / "bm25_corpus.json"
    bm25 = None
    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            tokenised_corpus: list[list[str]] = json.load(f)
        bm25 = BM25Okapi(tokenised_corpus)
        logger.info("event=bm25_loaded index=%s docs=%d", category, len(tokenised_corpus))
    else:
        logger.warning("event=bm25_missing index=%s", category)

    return vector_store, bm25


def _rrf_merge(
    faiss_results: list[tuple[Document, float]],
    bm25_results: list[tuple[int, float]],
    bm25_docstore,
    bm25_idx_to_docid: dict[int, str],
    constant: int = RRF_CONSTANT,
) -> list[Document]:
    """Merge two ranked lists using Reciprocal Rank Fusion."""
    rrf_scores: dict[str, dict] = {}

    for rank, (doc, _score) in enumerate(faiss_results, 1):
        cid = doc.metadata.get("chunk_id")
        if cid:
            rrf_scores[cid] = {"doc": doc, "rrf": 1.0 / (constant + rank)}

    for rank, (idx, _score) in enumerate(bm25_results, 1):
        doc_id = bm25_idx_to_docid.get(idx)
        if doc_id:
            doc = bm25_docstore.search(doc_id)
            cid = doc.metadata.get("chunk_id")
            if cid:
                if cid in rrf_scores:
                    rrf_scores[cid]["rrf"] += 1.0 / (constant + rank)
                else:
                    rrf_scores[cid] = {"doc": doc, "rrf": 1.0 / (constant + rank)}

    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["rrf"], reverse=True)
    return [item["doc"] for item in sorted_results[:RETRIEVAL_K]]


def retrieve(query: str, vector_store: FAISS, bm25: BM25Okapi | None) -> tuple[list[Document], dict]:
    """Retrieve documents using hybrid FAISS + BM25 with RRF merge."""
    # 1. FAISS search with scores
    faiss_results: list[tuple[Document, float]] = (
        vector_store.similarity_search_with_score(query, k=RETRIEVAL_K)
    )

    # 2. Filter by similarity threshold
    faiss_results = [(d, s) for d, s in faiss_results if s >= FAISS_SIM_THRESHOLD]

    # 3. Decide whether to use BM25
    top_score = faiss_results[0][1] if faiss_results else 0.0
    use_bm25 = bm25 is not None and (not faiss_results or top_score < FAISS_CONFIDENCE_THRESHOLD)

    if use_bm25:
        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[-RETRIEVAL_K:][::-1]
        bm25_results = [(int(idx), float(bm25_scores[idx])) for idx in top_bm25_indices]

        merged_docs = _rrf_merge(
            faiss_results,
            bm25_results,
            vector_store.docstore,
            vector_store.index_to_docstore_id,
        )
    else:
        merged_docs = [d for d, _ in faiss_results[:RETRIEVAL_K]]

    if not merged_docs:
        return [], {}

    # 4. FlashRank rerank
    ranker = Ranker(model_name=RERANKER_MODEL)
    passages = [
        {"id": i, "text": doc.page_content, "meta": dict(doc.metadata)}
        for i, doc in enumerate(merged_docs)
    ]
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked = ranker.rerank(rerank_request)
    reranked_docs = [merged_docs[r["id"]] for r in reranked]

    # 5. Dedup and source diversity
    deduped = deduplicate_chunks(reranked_docs)
    diverse = apply_source_diversity(deduped, max_per_page=2)

    # 6. Token budget
    budget_docs: list[Document] = []
    total_tokens = 0
    for doc in diverse:
        t = count_tokens(doc.page_content)
        if total_tokens + t > CHUNK_BUDGET:
            if not budget_docs:
                budget_docs.append(doc)
                total_tokens += t
            break
        budget_docs.append(doc)
        total_tokens += t

    logger.info("event=retrieval complete docs=%d tokens=%d", len(budget_docs), total_tokens)

    # 7. Citation map
    citation_map: dict[str, dict[str, str]] = {}
    for doc in budget_docs:
        cid = doc.metadata.get("chunk_id")
        if cid:
            citation_map[cid] = {
                "file": doc.metadata.get("source", "unknown"),
                "page": str(doc.metadata.get("page_number", "?")),
            }

    return budget_docs, citation_map


def main() -> None:
    groq_api_key: str = st.secrets["GROQ_API_KEY"]
    gemini_api_key: str = st.secrets["GEMINI_API_KEY"]

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_query_time" not in st.session_state:
        st.session_state.last_query_time = 0.0
    if "current_index" not in st.session_state:
        st.session_state.current_index = None

    st.set_page_config(page_title="DMRC Document Search", page_icon="📄", layout="wide")

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
    .status-msg { color: #5A6A7E; font-size: 0.85rem; font-style: italic; margin: 0.5rem 0; }
    [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E0E4E8; }
    [data-testid="stSidebar"] .stSelectbox label { font-size: 0.85rem; font-weight: 500; color: #1A1A2E; }
    .stAlert { border-left: 3px solid #003366; background: #F0F4F8; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="dmrc-header">
    <h1>DMRC Document Search Engine</h1>
    <p class="subtitle">Delhi Metro Rail Corporation — Intelligent Document Search</p>
</div>
""", unsafe_allow_html=True)

    with st.sidebar:
        st.header("Document Collection")
        available_indices = sorted(
            p.name for p in INDEX_DIR.iterdir()
            if p.is_dir() and p.name.endswith("_index")
            and (p / "index.faiss").exists() and (p / "index.pkl").exists()
        )
        if not available_indices:
            st.error("No vector indices found. Run ingest.py first.")
            st.stop()

        selected_index: str = st.selectbox("Collection", available_indices)

        if st.session_state.current_index != selected_index:
            st.session_state.messages = []
            st.session_state.current_index = selected_index

        with st.spinner("Loading..."):
            vector_store, bm25 = load_indices(selected_index)
            faiss_docs = vector_store.index.ntotal
            bm25_status = "✓ BM25" if bm25 else "✗ BM25 unavailable"
            st.caption(f"{faiss_docs} docs | {bm25_status}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query: str = st.chat_input("Ask a question about DMRC documents...")

    if user_query:
        if time.time() - st.session_state.last_query_time < COOLDOWN_SECONDS:
            st.warning("Please wait a moment before submitting another query.")
            st.stop()
        st.session_state.last_query_time = time.time()

        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        groq_model = st.secrets.get("GROQ_MODEL", GROQ_MODEL)
        gemini_model = st.secrets.get("GEMINI_FALLBACK_MODEL", GEMINI_FALLBACK_MODEL)
        llm_selector = ModelSelector(
            groq_api_key=groq_api_key,
            gemini_api_key=gemini_api_key,
            groq_model=groq_model,
            gemini_model=gemini_model,
            groq_timeout=GROQ_TIMEOUT,
        )

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown('<div class="status-msg">Searching documents...</div>', unsafe_allow_html=True)

            retrieved_docs, citation_map = retrieve(user_query, vector_store, bm25)

            if not retrieved_docs:
                response_placeholder.markdown("No relevant documents found.")
                st.session_state.messages.append({"role": "assistant", "content": "No relevant documents found."})
                st.stop()

            # --- Build prompt ---
            valid_ids = list(citation_map.keys())
            SYSTEM_PROMPT: str = (
                "You are a technical rail analyst for DMRC.\n"
                "Answer using the provided context blocks as your primary source.\n"
                "Search across all available context blocks — they may come from "
                "multiple documents. Synthesize information from all relevant sources.\n\n"
                "If information is not directly in context, you may make reasonable "
                "inferences based on related content. Indicate when you are inferring.\n\n"
                "Never fabricate metrics, speeds, telemetry, or calculations.\n\n"
                "Examples:\n"
                "- Acceptable: 'The signal head is approximately 2.5m high (inferred "
                "from mast dimensions on page 8).'\n"
                "- NOT acceptable: Making up a speed value, clearance point, or "
                "measurement not found anywhere in the provided documents.\n\n"
                "Synthesize the information. Do NOT repeat the same point "
                "multiple times — if multiple chunks contain the same "
                "information, state it once.\n\n"
                "When you use information from a chunk, cite it inline as [chunk_id].\n"
                f"Use only chunk_ids present in: {valid_ids}\n"
                "For example: 'The signal height is 2.5m [chunk_0]'\n"
                "If no specific chunk supports a statement, do not add any citation."
            )

            context_str: str = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)
            user_prompt: str = (
                f"Context:\n{context_str}\n\n"
                f"Question: {user_query}\n\n"
                "Provide a precise answer based on the context above. "
                "Use all relevant sources. Where you infer, note it."
            )

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            # --- LLM Generation ---
            response_placeholder.markdown('<div class="status-msg">Generating answer...</div>', unsafe_allow_html=True)
            t_llm_start = time.monotonic()

            try:
                result = llm_selector.invoke_with_fallback(messages)
                full_response = result.content
            except LLMExhaustedError:
                logger.error("event=llm_all_providers_exhausted")
                st.error("All LLM providers are temporarily unavailable. Please try again later.")
                st.stop()
                return
            except Exception as e:
                logger.error("event=llm_unexpected_error error=%s", str(e)[:200])
                st.error("An unexpected error occurred. Please try again later.")
                st.stop()
                return

            # --- Citation synthesis ---
            display_response, cited_ids = process_citations(full_response, citation_map)
            response_placeholder.markdown(display_response)

            # Render expanders
            ref_num = 1
            seen_citations: set[str] = set()
            for cid in cited_ids:
                ref = citation_map.get(cid)
                if not ref:
                    continue
                citation_key = f"{ref['file']}:{ref['page']}"
                if citation_key in seen_citations:
                    continue
                seen_citations.add(citation_key)
                with st.expander(f"**Refer {ref_num}**", expanded=False):
                    st.caption(f"{ref['file']} — Page {ref['page']}")
                    for doc in retrieved_docs:
                        if doc.metadata.get("chunk_id") == cid:
                            st.text(doc.page_content)
                            break
                ref_num += 1

            logger.info("event=query_complete tokens=%d citations=%d", len(context_str), len(seen_citations))
            elapsed = time.monotonic() - t_llm_start
            logger.info("event=llm_done elapsed_s=%.2f model=%s", elapsed, "groq/gemini")

        st.session_state.messages.append({"role": "assistant", "content": display_response})


if __name__ == "__main__":
    main()
