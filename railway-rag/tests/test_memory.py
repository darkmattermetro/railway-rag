"""
Memory pressure test: load synthetic FAISS + BM25 index, run 5 queries,
track peak memory via tracemalloc. Warn (not fail) if peak > 800 MB.
"""
import gc
import logging
import tracemalloc
import uuid

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

DIM = 768
NUM_DOCS = 500
QUERIES = [
    "maximum permitted speed",
    "signal spacing requirements",
    "braking distance calculation",
    "track gauge standards",
    "overhead wire clearance",
]


def _build_synthetic_index():
    """Create an in-memory FAISS wrapper + BM25Okapi with random data."""
    rng = np.random.default_rng(42)
    embeddings = rng.random((NUM_DOCS, DIM), dtype=np.float32)
    index = faiss.IndexFlatIP(DIM)
    index.add(embeddings)

    documents = [
        Document(
            page_content=f"Synthetic railway document {i} with some technical content about safety standards and operational procedures.",
            metadata={
                "chunk_id": f"chunk_{i}",
                "source": "synth.pdf",
                "page_number": i % 20 + 1,
            },
        )
        for i in range(NUM_DOCS)
    ]
    docstore_ids = [str(uuid.uuid4()) for _ in documents]
    docstore = InMemoryDocstore(dict(zip(docstore_ids, documents)))
    index_to_docstore_id = dict(enumerate(docstore_ids))

    def _embed_query(text: str):
        return rng.random(DIM).tolist()

    vector_store = FAISS(
        embedding_function=_embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    tokenised = [d.page_content.lower().split() for d in documents]
    bm25 = BM25Okapi(tokenised)

    return vector_store, bm25


def test_retrieval_memory_under_800mb():
    """Run 5 queries against a synthetic index; warn if tracemalloc peak > 800 MB."""
    tracemalloc.start()
    gc.collect()

    vector_store, bm25 = _build_synthetic_index()

    for query in QUERIES:
        results = vector_store.similarity_search_with_score(query, k=20)
        filtered = [(d, s) for d, s in results if s >= 0.35]
        if bm25 and (not filtered or filtered[0][1] < 0.60):
            bm25.get_scores(query.lower().split())
        for d, _ in filtered[:10]:
            _ = d.page_content

    gc.collect()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    logger.info("event=memory_test peak_mb=%.1f threshold_mb=800", peak_mb)

    if peak_mb > 800:
        logger.warning(
            "event=memory_test_high_peak peak_mb=%.1f threshold_mb=800 "
            "— consider reducing index size or adding gc.collect() calls",
            peak_mb,
        )

    del vector_store, bm25
    gc.collect()
