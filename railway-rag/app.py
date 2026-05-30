#!/usr/bin/env python3
"""Streamlit Cloud Retrieval Application for Railway RAG Pipeline.

FAISS similarity search with score-weighted context allocation.
Runs within ~1 GB RAM constraint on Streamlit Cloud.
"""

import hashlib
import json
import logging
import os
import random
import time
import re
from pathlib import Path

import streamlit as st
import tiktoken
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from utils import (
    _word_overlap_ratio,
    apply_source_diversity,
    deduplicate_chunks,
    mmr_diversity,
    parse_citations,
    rrf_merge,
    FAISS_SIM_THRESHOLD,
    RETRIEVAL_K,
    TOKEN_BUDGET,
)
from model_selector import ModelSelector, LLMExhaustedError

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
COOLDOWN_SECONDS: float = 1.0
GROQ_FALLBACK_MODEL: str = "llama-3.3-70b-versatile"
GEMINI_PRIMARY_MODEL: str = "gemini-2.5-flash"
RERANKER_TOP_N: int = 8
RERANKER_ENABLED: bool = os.environ.get("RERANKER_ENABLED", "0") == "1"
RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
MAX_CITATION_EXPANDERS: int = 5
INDEX_DIR: Path = Path(__file__).resolve().parent / "vector_indices"
_ENC = tiktoken.get_encoding("cl100k_base")

SYSTEM_PROMPT: str = (
    "You are a technical rail analyst for DMRC.\n"
    "Answer using the provided context blocks as your primary source.\n"
    "Search across all available context blocks — they may come from "
    "multiple documents. Synthesize information from all relevant sources.\n\n"
    "If information is not directly in context, you may make reasonable "
    "inferences based on related content. You MUST wrap any inferred information inside <inferred> and </inferred> tags.\n\n"
    "Never fabricate metrics, speeds, telemetry, or calculations.\n\n"
    "Examples:\n"
    "- Acceptable: '<inferred>The signal head is approximately 2.5m high (based on mast dimensions on page 8).</inferred>'\n"
    "- NOT acceptable: Making up a speed value, clearance point, or "
    "measurement not found anywhere in the provided documents.\n\n"
    "Synthesize the information. Do NOT repeat the same point "
    "multiple times — if multiple chunks contain the same "
    "information, state it once.\n\n"
    "At the end of your answer append:\n"
    "<used_chunks>[0, 2]</used_chunks>\n\n"
    "The values inside used_chunks must be integers only — "
    "not ranges, not strings."
)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def _token_length(text: str) -> int:
    return len(_ENC.encode(text))


@st.cache_resource(max_entries=1)
def load_vector_index(index_path: Path) -> FAISS:
    """Load FAISS index from disk with integrity check."""
    hash_path = index_path / "index.hash"
    if hash_path.exists():
        expected = hash_path.read_text().strip()
        actual = hashlib.sha256()
        for fname in ("index.faiss", "index.pkl"):
            with open(index_path / fname, "rb") as f:
                while chunk := f.read(65536):
                    actual.update(chunk)
        if actual.hexdigest() != expected:
            raise RuntimeError(f"FAISS index integrity mismatch at {index_path}")
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
    model_file = index_path / "embedding_model.txt"

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
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def _check_available_memory_mb():
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
    try:
        import psutil
        return int(psutil.virtual_memory().available / (1024 * 1024))
    except ImportError:
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


@st.cache_resource(max_entries=2)
def load_bm25_index(index_path: Path):
    corpus_path = index_path / "bm25_corpus.json"
    if not corpus_path.exists():
        logger.info("event=bm25_corpus_not_found path=%s", corpus_path)
        return None, []
    from rank_bm25 import BM25Okapi
    with open(corpus_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    texts = [e["text"] for e in entries]
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    logger.info("event=bm25_index_loaded entries=%d", len(entries))
    return bm25, entries


def discover_indices() -> list[Path]:
    """Scan INDEX_DIR for valid index subdirectories."""
    if not INDEX_DIR.exists():
        return []
    indices: list[Path] = []
    for path in INDEX_DIR.iterdir():
        if path.is_dir() and path.name.endswith("_index"):
            # Verify FAISS index files exist
            if (path / "index.faiss").exists() and (path / "index.pkl").exists():
                indices.append(path)
    return sorted(indices)


# ----------------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------------
def main() -> None:
    """Main entrypoint for the Streamlit retrieval application."""

    # --------------------------------------------------------------------
    # Secrets (read once in main, not at module scope)
    # --------------------------------------------------------------------
    # Parse API keys with backward compatibility for legacy single-key format
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

    # --------------------------------------------------------------------
    # UI Layout
    # --------------------------------------------------------------------
    st.set_page_config(
        page_title="DMRC Document Search",
        page_icon="📄",
        layout="wide",
    )

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

    div.stMarkdown blockquote {
        background-color: #FFF3CD;
        border-left: 4px solid #FFC107;
        padding: 0.75rem 1rem;
        margin: 0.75rem 0;
        border-radius: 4px;
        color: #856404;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    div.stMarkdown blockquote strong {
        color: #664D03;
        margin-right: 0.25rem;
    }
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

        index_names: list[str] = [p.name for p in available_indices]
        selected_index_name: str = st.selectbox(
            "Collection",
            index_names,
            help="Choose which document set to search",
        )
        selected_index_path: Path = INDEX_DIR / selected_index_name

        # Clear chat history when index changes
        if st.session_state.current_index != selected_index_name:
            st.session_state.messages = []
            st.session_state.current_index = selected_index_name

        # Load index
        with st.spinner("Loading..."):
            try:
                vector_store = load_vector_index(selected_index_path)
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

    # --------------------------------------------------------------------
    # Chat history display
    # --------------------------------------------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = re.sub(
                r'<inferred>(.*?)</inferred>',
                r'> **🧠 AI Gen:** \1',
                message["content"],
                flags=re.IGNORECASE | re.DOTALL,
            )
            st.markdown(content)

    if last_llm := st.session_state.get("last_llm"):
        st.caption(f"{last_llm}")

    # --------------------------------------------------------------------
    # Query handling
    # --------------------------------------------------------------------
    user_query: str = st.chat_input("Ask a question about DMRC documents...")

    if user_query:
        t_query_start: float = time.monotonic()

        # Cooldown protection
        if time.time() - st.session_state.last_query_time < COOLDOWN_SECONDS:
            st.warning("Please wait a moment before submitting another query.")
            st.stop()
        st.session_state.last_query_time = time.time()
        
        # Increment global query count for model rotation
        current_query_count = st.session_state.query_count
        st.session_state.query_count += 1

        logger.info("event=query_received query_len=%d query_count=%d", len(user_query), current_query_count)

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # --------------------------------------------------------------------
        # LLM Setup (for query rewriting, condensation, and generation)
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

        # Query rewriting: expand user query with LLM-generated variants
        queries_to_search = [user_query]
        rewrite_prompt = (
            f"Generate 2 alternative phrasings of this railway query "
            f"that might match different technical terminology.\n"
            f"Original: {user_query}\n"
            f"Return one variant per line, no prefixes."
        )
        rewrite_messages = [
            SystemMessage(content="You are a search query expansion assistant. Output only the variant queries, one per line."),
            HumanMessage(content=rewrite_prompt),
        ]
        try:
            rewrite_result = llm_selector.invoke_with_fallback(rewrite_messages)
            variants = [line.strip() for line in rewrite_result.strip().split("\n") if line.strip()]
            for v in variants:
                if v.lower() != user_query.lower() and v not in queries_to_search:
                    queries_to_search.append(v)
            logger.info("event=query_rewrite variants=%d queries=%s", len(queries_to_search) - 1, queries_to_search)
        except Exception as e:
            logger.warning("event=query_rewrite_failed error=%s", e)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown('<div class="status-msg">Searching documents...</div>', unsafe_allow_html=True)

            # ----------------------------------------------------------------
            # Hybrid Retrieval: multi-query FAISS + BM25 with RRF merge
            # ----------------------------------------------------------------
            bm25, corpus_entries = load_bm25_index(selected_index_path)
            seen_cids: set[str] = set()
            merged_docs: list[Document] = []
            for q in queries_to_search:
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
            if reranker := load_reranker():
                pairs = [[user_query, doc.page_content] for doc in diverse_docs]
                scores = reranker.predict(pairs)
                scored = sorted(zip(scores, diverse_docs), key=lambda x: x[0], reverse=True)
                reranked_docs = [doc for _, doc in scored[:RERANKER_TOP_N]]
            else:
                reranked_docs = diverse_docs[:RERANKER_TOP_N]

                logger.info(
                "event=retrieval_complete merged=%d deduped=%d diverse=%d reranked=%d",
                len(merged_docs),
                len(deduped_docs),
                len(diverse_docs),
                len(reranked_docs),
            )

            # ----------------------------------------------------------------
            # Score-aware token budget
            #   When reranker is active, weights are proportional to actual
            #   cross-encoder scores.  Otherwise fall back to position-based
            #   weights with finer granularity.
            # ----------------------------------------------------------------
            if reranker:
                available_scores = [score for score, _ in scored[:RERANKER_TOP_N]]
                min_s = min(available_scores) if available_scores else 0
                max_s = max(available_scores) if available_scores else 1
                span = max_s - min_s
                if span > 0:
                    chunk_weights = [(s - min_s) / span for s in available_scores]
                else:
                    chunk_weights = [1.0] * len(available_scores)
            else:
                n = len(reranked_docs)
                chunk_weights = [max(0.25, 1.0 - i * 0.12) for i in range(n)]
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
                    context_parts.append(_ENC.decode(_ENC.encode(doc.page_content)[:alloc]))
                total_tokens += alloc

            logger.info(
                "event=context_built chunks=%d tokens=%d",
                len(context_parts),
                total_tokens,
            )

            # ----------------------------------------------------------------
            # Build prompt
            # ----------------------------------------------------------------
            context_str: str = "\n\n---\n\n".join(
                f"[{i}] {c}" for i, c in enumerate(context_parts)
            )
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

            # ----------------------------------------------------------------
            # LLM Generation with intelligent fallback and model rotation
            # ----------------------------------------------------------------
            response_placeholder.markdown('<div class="status-msg">Generating answer...</div>', unsafe_allow_html=True)
            full_response: str = ""
            t_llm_start: float = time.monotonic()

            time.sleep(0.5 + random.random() * 0.5)

            try:
                stream_generator = llm_selector.stream_with_fallback(messages)
                
                # Stream the response chunk by chunk to Streamlit
                stream_text = ""
                for chunk in stream_generator:
                    if isinstance(chunk.content, str):
                        stream_text += chunk.content
                    
                    # Intercept `<used_chunks>` from display buffer so it doesn't render live
                    display_text = re.sub(r'<used_chunks>.*', '', stream_text, flags=re.DOTALL)
                    
                    # Format <inferred> blocks live if they exist
                    display_text = re.sub(
                        r'<inferred>(.*?)</inferred>',
                        r'> **🧠 AI Gen:** \1',
                        display_text,
                        flags=re.IGNORECASE | re.DOTALL
                    )
                    
                    response_placeholder.markdown(display_text + "▌")
                
                full_response = stream_text
                st.session_state.last_llm = llm_selector.last_provider
                # Note: Success logging is now handled inside ModelSelector.stream_with_fallback
            except LLMExhaustedError as e:
                logger.error(
                    "event=llm_all_providers_exhausted error=%s",
                    str(e)[:200],
                )
                st.error("All LLM providers are temporarily unavailable due to quota limits. Please try again later.")
                st.stop()

            # ----------------------------------------------------------------
            # Citation synthesis (programmatic, not trusting LLM)
            # ----------------------------------------------------------------
            display_response, chunk_indices = parse_citations(full_response)

            display_response = re.sub(
                r'<inferred>(.*?)</inferred>',
                r'> **🧠 AI Gen:** \1',
                display_response,
                flags=re.IGNORECASE | re.DOTALL
            )

            # Select top N most relevant citations based on word overlap with the response
            # 1. Filter valid indices
            valid_indices = [idx for idx in chunk_indices if 0 <= idx < len(reranked_docs)]

            # 2. Score and Rank
            if valid_indices:
                # Calculate scores and store as (score, original_index)
                scored_indices = []
                for idx in valid_indices:
                    score = _word_overlap_ratio(display_response, reranked_docs[idx].page_content)
                    scored_indices.append((score, idx))
                
                # Sort by score (descending), then by original index (ascending) for stability
                scored_indices.sort(key=lambda x: (-x[0], x[1]))
                
                # 3. Select top N
                final_indices = [idx for score, idx in scored_indices[:MAX_CITATION_EXPANDERS]]
            else:
                final_indices = []

            # Display the answer
            response_placeholder.markdown(display_response)

            # Render LLM-tagged citations with expandable chunk text
            seen_citations: set[str] = set()
            ref_num = 1  # Reference counter starting at 1
            for idx in final_indices:
                if 0 <= idx < len(reranked_docs):
                    doc = reranked_docs[idx]
                    source: str = doc.metadata.get("source", "unknown")
                    page: str = str(doc.metadata.get("page_number", "?"))
                    heading: str = doc.metadata.get("heading", "")
                    citation_key: str = f"{source}:{page}"
                    if citation_key not in seen_citations:
                        seen_citations.add(citation_key)
                        expander_label = f"**Refer {ref_num}**"
                        if heading:
                            expander_label += f" — {heading}"
                        with st.expander(
                            expander_label,
                            expanded=False,
                        ):
                            st.caption(f"{source} — Page {page}")
                            st.text(doc.page_content)
                        ref_num += 1

            logger.info(
                "event=citations_rendered count=%d indices=%s",
                len(seen_citations),
                chunk_indices,
            )

            logger.info(
                "event=query_complete tokens=%d citations=%d",
                total_tokens,
                len(seen_citations),
            )

        # Add assistant response to history
        st.session_state.messages.append(
            {"role": "assistant", "content": display_response}
        )


if __name__ == "__main__":
    main()