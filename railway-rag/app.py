#!/usr/bin/env python3
"""Streamlit Cloud Retrieval Application for Railway RAG Pipeline.

Hybrid retrieval with FAISS + BM25, FlashRank reranking, and LLM generation.
Runs within ~1 GB RAM constraint on Streamlit Cloud.
"""

import gc
import hashlib
import json
import logging
import time
from pathlib import Path

import streamlit as st
import tiktoken
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from google.api_core.exceptions import GoogleAPIError
from groq import APIStatusError, APITimeoutError, RateLimitError
from utils import (
    apply_source_diversity,
    deduplicate_chunks,
    parse_citations,
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
GROQ_TIMEOUT: int = 45
GEMINI_PRIMARY_MODEL: str = "gemini-2.5-flash"
RERANKER_MODEL: str = "ms-marco-MiniLM-L-12-v2"
RERANKER_TOP_N: int = 15
MAX_CITATION_EXPANDERS: int = 5
INDEX_DIR: Path = Path(__file__).resolve().parent / "vector_indices"
_ENC = tiktoken.get_encoding("cl100k_base")

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def _token_length(text: str) -> int:
    return len(_ENC.encode(text))


def _calculate_text_overlap(text1: str, text2: str) -> float:
    """Calculates a simple word-overlap score between two strings."""
    import re
    words1 = set(re.findall(r"\b[a-z0-9]+\b", text1.lower()))
    words2 = set(re.findall(r"\b[a-z0-9]+\b", text2.lower()))
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / max(len(words1), len(words2))


@st.cache_resource
def load_vector_index(index_path: Path) -> FAISS:
    """Load FAISS index from disk with integrity check."""
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
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store


@st.cache_resource
def load_bm25_corpus(index_dir: Path) -> tuple[BM25Retriever | None, bool]:
    """Load BM25 corpus from JSON file. Returns (retriever, available_flag)."""
    corpus_path: Path = index_dir / "bm25_corpus.json"
    if not corpus_path.exists():
        logger.warning("event=bm25_missing index=%s", index_dir.name)
        return None, False

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data: list[dict] = json.load(f)

    bm25_docs: list[Document] = [
        Document(page_content=item["text"], metadata=item["metadata"])
        for item in corpus_data
    ]
    del corpus_data
    gc.collect()

    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = RETRIEVAL_K
    logger.info("event=bm25_loaded index=%s docs=%d", index_dir.name, len(bm25_docs))
    return bm25_retriever, True


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
    groq_api_key: str = st.secrets["GROQ_API_KEY"]
    gemini_api_key: str = st.secrets["GEMINI_API_KEY"]

    # --------------------------------------------------------------------
    # Session state initialization
    # --------------------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_query_time" not in st.session_state:
        st.session_state.last_query_time = 0.0
    if "current_index" not in st.session_state:
        st.session_state.current_index = None

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

        # Load index and BM25
        with st.spinner("Loading..."):
            vector_store = load_vector_index(selected_index_path)
            faiss_docs_count: int = vector_store.index.ntotal
            logger.info("event=index_loaded index=%s faiss_docs=%d", selected_index_name, faiss_docs_count)

        bm25_retriever, bm25_available = load_bm25_corpus(selected_index_path)

        if not bm25_available:
            logger.warning("event=bm25_missing index=%s", selected_index_name)

    # --------------------------------------------------------------------
    # Chat history display
    # --------------------------------------------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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

        logger.info("event=query_received query_len=%d", len(user_query))

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # --------------------------------------------------------------------
        # LLM Setup (for query condensation and final generation)
        # --------------------------------------------------------------------
        # Get model lists from secrets with fallbacks to original values
        gemini_models = st.secrets.get("GEMINI_MODELS", [GEMINI_PRIMARY_MODEL])
        groq_models = st.secrets.get("GROQ_MODELS", [GROQ_FALLBACK_MODEL])
        
        # Initialize model selector
        llm_selector = ModelSelector(
            gemini_models=gemini_models,
            groq_models=groq_models,
            gemini_api_key=gemini_api_key,
            groq_api_key=groq_api_key
        )

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown('<div class="status-msg">Searching documents...</div>', unsafe_allow_html=True)

            # ----------------------------------------------------------------
            # Setup retrievers
            # ----------------------------------------------------------------
            faiss_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

            if bm25_available:
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[faiss_retriever, bm25_retriever],
                    weights=[0.3, 0.7],
                )
                active_retriever = ensemble_retriever
            else:
                active_retriever = faiss_retriever

            # ----------------------------------------------------------------
            # Reranking setup
            # ----------------------------------------------------------------
            reranker = FlashrankRerank(
                model=RERANKER_MODEL,
                top_n=RERANKER_TOP_N,
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=active_retriever,
            )

            # ----------------------------------------------------------------
            # Retrieval
            # ----------------------------------------------------------------
            retrieved_docs: list[Document] = compression_retriever.invoke(user_query)
            raw_count: int = len(retrieved_docs)

            # ----------------------------------------------------------------
            # Deduplication
            # ----------------------------------------------------------------
            deduped_docs = deduplicate_chunks(retrieved_docs)
            deduped_count: int = len(deduped_docs)

            # Source diversity: max 2 chunks per (source, page) pair
            diverse_docs = apply_source_diversity(deduped_docs, max_per_page=2)
            diverse_count: int = len(diverse_docs)
            logger.info(
                "event=dedup_complete before=%d deduped=%d diverse=%d",
                raw_count,
                deduped_count,
                diverse_count,
            )

            reranked_docs: list[Document] = diverse_docs[:RERANKER_TOP_N]
            reranked_count: int = len(reranked_docs)
            logger.info(
                "event=retrieval_complete raw=%d deduped=%d diverse=%d reranked=%d",
                raw_count,
                deduped_count,
                diverse_count,
                reranked_count,
            )

            # ----------------------------------------------------------------
            # Token budget protection
            # ----------------------------------------------------------------
            context_parts: list[str] = []
            total_tokens: int = 0

            for i, doc in enumerate(reranked_docs):
                chunk_tokens: int = _token_length(doc.page_content)
                if total_tokens + chunk_tokens > TOKEN_BUDGET:
                    remaining: int = TOKEN_BUDGET - total_tokens
                    truncated_text: str = _ENC.decode(_ENC.encode(doc.page_content)[:remaining])
                    context_parts.append(truncated_text)
                    total_tokens += remaining
                    logger.info(
                        "event=token_budget_truncated chunk=%d remaining=%d total=%d",
                        i,
                        remaining,
                        total_tokens,
                    )
                    break
                context_parts.append(doc.page_content)
                total_tokens += chunk_tokens

            logger.info(
                "event=context_built chunks=%d tokens=%d",
                len(context_parts),
                total_tokens,
            )

            # ----------------------------------------------------------------
            # Build prompt
            # ----------------------------------------------------------------
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
                "At the end of your answer append:\n"
                "<used_chunks>[0, 2]</used_chunks>\n\n"
                "The values inside used_chunks must be integers only — "
                "not ranges, not strings."
            )

            context_str: str = "\n\n---\n\n".join(context_parts)
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

            try:
                result = llm_selector.invoke_with_fallback(messages)
                full_response = result.content
                st.session_state.last_llm = llm_selector.last_provider
                # Note: Success logging is now handled inside ModelSelector.invoke_with_fallback
            except LLMExhaustedError as e:
                logger.error(
                    "event=llm_all_providers_exhausted error=%s",
                    str(e)[:200],
                )
                st.error("All LLM providers are temporarily unavailable due to quota limits. Please try again later.")
                st.stop()
            except Exception as e:
                logger.error(
                    "event=llm_unexpected_error error=%s",
                    str(e)[:200],
                )
                st.error("An unexpected error occurred. Please try again later.")
                st.stop()

            # ----------------------------------------------------------------
            # Citation synthesis (programmatic, not trusting LLM)
            # ----------------------------------------------------------------
            display_response, chunk_indices = parse_citations(full_response)

            # Select top N most relevant citations based on word overlap with the response
            # 1. Filter valid indices
            valid_indices = [idx for idx in chunk_indices if 0 <= idx < len(reranked_docs)]

            # 2. Score and Rank
            if valid_indices:
                # Calculate scores and store as (score, original_index)
                scored_indices = []
                for idx in valid_indices:
                    score = _calculate_text_overlap(display_response, reranked_docs[idx].page_content)
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
                    citation_key: str = f"{source}:{page}"
                    if citation_key not in seen_citations:
                        seen_citations.add(citation_key)
                        with st.expander(
                            f"**Refer {ref_num}**",
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
