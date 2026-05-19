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
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from google.api_core.exceptions import GoogleAPIError
from groq import APIStatusError, APITimeoutError, RateLimitError
from utils import (
    apply_source_diversity,
    deduplicate_chunks,
    extract_technical_identifiers,
    parse_citations,
    RETRIEVAL_K,
    TOKEN_BUDGET,
)

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
COOLDOWN_SECONDS: float = 2.0
GROQ_PRIMARY_MODEL: str = "llama-3.1-8b-instant"
GROQ_TIMEOUT: int = 25
GEMINI_FALLBACK_MODEL: str = "gemini-2.5-flash"
RERANKER_MODEL: str = "ms-marco-MiniLM-L-12-v2"
RERANKER_TOP_N: int = 5
INDEX_DIR: Path = Path("vector_indices")
_ENC = tiktoken.get_encoding("cl100k_base")

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def _token_length(text: str) -> int:
    return len(_ENC.encode(text))


@st.cache_resource
def load_vector_index(index_path: Path, gemini_api_key: str) -> FAISS:
    """Load FAISS index from disk with integrity check."""
    hash_path = index_path / "index.hash"
    if hash_path.exists():
        expected = hash_path.read_text().strip()
        actual = hashlib.sha256()
        actual.update((index_path / "index.faiss").read_bytes())
        actual.update((index_path / "index.pkl").read_bytes())
        if actual.hexdigest() != expected:
            raise RuntimeError(f"FAISS index integrity mismatch at {index_path}")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=gemini_api_key
    )
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


def condense_query(user_query: str, primary_llm: ChatGroq, fallback_llm: ChatGoogleGenerativeAI) -> str:
    """Shorten query while preserving intent, then append technical codes.
    
    Uses the same failover pattern as §3.13 — this is the ONLY other location
    where we need LLM calls before the main generation. Broad exception catching
    is NOT permitted here, so we use targeted exceptions only.
    """
    # Extract technical codes BEFORE condensation
    technical_codes: list[str] = extract_technical_identifiers(user_query)

    # Short condensation prompt
    condensation_prompt: str = (
        "Rewrite the following technical railway query concisely (max 20 words), "
        "preserving all technical terms and identifiers:\n\n"
        f"{user_query}"
    )
    
    # Try primary LLM with targeted exceptions only
    condensed: str = user_query  # fallback to original if both fail
    try:
        result = primary_llm.invoke([HumanMessage(content=condensation_prompt)])
        condensed = result.content.strip()
        logger.info("event=query_condensation_success model=%s", GROQ_PRIMARY_MODEL)
    except (RateLimitError, APIStatusError, APITimeoutError) as e:
        logger.warning(
            "event=query_condensation_primary_failed error=%s — attempting fallback",
            str(e),
        )
        try:
            result = fallback_llm.invoke([HumanMessage(content=condensation_prompt)])
            condensed = result.content.strip()
            logger.info("event=query_condensation_fallback_success model=%s", GEMINI_FALLBACK_MODEL)
        except (GoogleAPIError, TimeoutError, APIStatusError) as fb_e:
            logger.error(
                "event=query_condensation_fallback_failed err_type=%s err=%s — using original query",
                type(fb_e).__name__,
                str(fb_e),
            )

    # Append identifiers back to prevent loss
    if technical_codes:
        condensed = condensed + " " + " ".join(technical_codes)

    return condensed


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
        page_title="Railway RAG Retriever",
        page_icon="🚂",
        layout="wide",
    )
    st.title("🚂 Railway Technical Document Retrieval")
    st.markdown("---")

    # --------------------------------------------------------------------
    # Index selection sidebar
    # --------------------------------------------------------------------
    with st.sidebar:
        st.header("📚 Document Index")
        available_indices = discover_indices()

        if not available_indices:
            st.error(
                "No vector indices found. Please run local_builder.py first to "
                "ingest documents into the vector_indices/ directory."
            )
            st.stop()

        index_names: list[str] = [p.name for p in available_indices]
        selected_index_name: str = st.selectbox(
            "Select document collection",
            index_names,
            help="Choose which ingested document set to search",
        )
        selected_index_path: Path = INDEX_DIR / selected_index_name

        # Clear chat history when index changes
        if st.session_state.current_index != selected_index_name:
            st.session_state.messages = []
            st.session_state.current_index = selected_index_name

        # Load index and BM25
        with st.spinner("Loading vector index..."):
            vector_store = load_vector_index(selected_index_path, gemini_api_key)
            faiss_docs_count: int = vector_store.index.ntotal
            logger.info("event=index_loaded index=%s faiss_docs=%d", selected_index_name, faiss_docs_count)

        bm25_retriever, bm25_available = load_bm25_corpus(selected_index_path)

        if not bm25_available:
            st.warning(
                "BM25 corpus not found — using FAISS-only retrieval. "
                "Run local_builder.py to rebuild the full index."
            )

        st.divider()
        st.caption(f"FAISS contains {faiss_docs_count} document chunks")
        if bm25_available:
            st.caption("✓ BM25 index available for hybrid search")

    # --------------------------------------------------------------------
    # Chat history display
    # --------------------------------------------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --------------------------------------------------------------------
    # Query handling
    # --------------------------------------------------------------------
    user_query: str = st.chat_input("Ask a technical question about railway systems...")

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
        primary_llm = ChatGroq(
            model=GROQ_PRIMARY_MODEL,
            timeout=GROQ_TIMEOUT,
            temperature=0,
            api_key=groq_api_key,
        )
        fallback_llm = ChatGoogleGenerativeAI(
            model=GEMINI_FALLBACK_MODEL,
            temperature=0,
            google_api_key=gemini_api_key,
        )

        # Condense query using the two-LLM failover pattern
        condensed: str = condense_query(user_query, primary_llm, fallback_llm)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("🔍 **Retrieving relevant documents...**")

            # ----------------------------------------------------------------
            # Setup retrievers
            # ----------------------------------------------------------------
            faiss_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

            if bm25_available:
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[faiss_retriever, bm25_retriever],
                    weights=[0.5, 0.5],
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
            retrieved_docs: list[Document] = compression_retriever.invoke(condensed)
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
                    context_parts.append(f"[CHUNK {i}]\n{truncated_text}")
                    total_tokens += remaining
                    logger.info(
                        "event=token_budget_truncated chunk=%d remaining=%d total=%d",
                        i,
                        remaining,
                        total_tokens,
                    )
                    break
                context_parts.append(f"[CHUNK {i}]\n{doc.page_content}")
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
                "You are a risk-averse technical rail analyst.\n"
                "Answer ONLY using the provided context blocks.\n"
                "If the information does not exist in context, respond:\n"
                "'Information not found within local documents.'\n\n"
                "Never fabricate metrics, speeds, telemetry, or calculations.\n\n"
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
                "Provide a precise answer based only on the context above."
            )

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            # ----------------------------------------------------------------
            # LLM Generation with failover
            # ----------------------------------------------------------------
            response_placeholder.markdown("🤖 **Generating answer...**")
            full_response: str = ""
            t_llm_start: float = time.monotonic()

            try:
                result = primary_llm.invoke(messages)
                full_response = result.content
                logger.info(
                    "event=llm_primary_success model=%s elapsed_s=%.2f",
                    GROQ_PRIMARY_MODEL,
                    time.monotonic() - t_llm_start,
                )
            except (RateLimitError, APIStatusError, APITimeoutError) as e:
                logger.warning(
                    "event=llm_primary_failed model=%s error=%s — attempting fallback",
                    GROQ_PRIMARY_MODEL,
                    str(e),
                )
                st.warning("Primary LLM unavailable — switching to fallback model.")
                try:
                    result = fallback_llm.invoke(messages)
                    full_response = result.content
                    logger.info(
                        "event=llm_fallback_success model=%s elapsed_s=%.2f",
                        GEMINI_FALLBACK_MODEL,
                        time.monotonic() - t_llm_start,
                    )
                except (GoogleAPIError, TimeoutError, APIStatusError) as fb_exc:
                    logger.error(
                        "event=llm_fallback_failed model=%s err_type=%s err=%s",
                        GEMINI_FALLBACK_MODEL,
                        type(fb_exc).__name__,
                        str(fb_exc),
                    )
                    st.error("Both LLM providers failed. Please try again later.")
                    st.stop()

            # ----------------------------------------------------------------
            # Citation synthesis (programmatic, not trusting LLM)
            # ----------------------------------------------------------------
            display_response, chunk_indices = parse_citations(full_response)

            # Display the answer
            response_placeholder.markdown(display_response)

            # Render LLM-tagged citations
            seen_citations: set[str] = set()
            for idx in chunk_indices:
                if 0 <= idx < len(reranked_docs):
                    doc = reranked_docs[idx]
                    source: str = doc.metadata.get("source", "unknown")
                    page: str = str(doc.metadata.get("page_number", "?"))
                    citation_key: str = f"{source}:{page}"
                    if citation_key not in seen_citations:
                        seen_citations.add(citation_key)
                        st.markdown(
                            f"📄 **{source}** — Page {page}",
                            help=f"Chunk {idx} contributed to this answer",
                        )

            logger.info(
                "event=citations_rendered count=%d indices=%s",
                len(seen_citations),
                chunk_indices,
            )

            # ----------------------------------------------------------------
            # Fallback citations: always show top retrieved sources
            # ----------------------------------------------------------------
            st.markdown("#### Retrieved Sources")
            fallback_seen: set[str] = set()
            for idx, doc in enumerate(reranked_docs):
                source = doc.metadata.get("source", "unknown")
                page = str(doc.metadata.get("page_number", "?"))
                citation_key = f"{source}:{page}"
                if citation_key not in fallback_seen:
                    fallback_seen.add(citation_key)
                    cited_marker = " ✓" if idx in chunk_indices else ""
                    st.markdown(
                        f"📄 **{source}** — Page {page}{cited_marker}",
                        help=f"Retrieved chunk {idx}",
                    )

            # ----------------------------------------------------------------
            # Display metrics
            # ----------------------------------------------------------------
            elapsed_total: float = time.monotonic() - t_query_start
            st.caption(
                f"📊 Retrieved {raw_count} → deduped {deduped_count} → "
                f"diverse {diverse_count} → reranked {reranked_count} | "
                f"Context: {len(context_parts)} chunks | "
                f"{total_tokens} tokens | ⏱️ {elapsed_total:.2f}s"
            )

            logger.info(
                "event=query_complete elapsed_s=%.2f tokens=%d citations=%d",
                elapsed_total,
                total_tokens,
                len(seen_citations),
            )

        # Add assistant response to history
        st.session_state.messages.append(
            {"role": "assistant", "content": display_response}
        )


if __name__ == "__main__":
    main()
