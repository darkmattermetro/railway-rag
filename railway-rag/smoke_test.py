#!python
"""
Railway RAG Smoke Test — Standalone CLI validation.

Loads a FAISS index + BM25 corpus, runs a query, prints top results.
No API key required; uses BAAI/bge-base-en-v1.5 embeddings (downloaded once).
"""
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _verify_integrity(index_dir: Path) -> None:
    hash_path = index_dir / "index.hash"
    if not hash_path.exists():
        return
    expected = hash_path.read_text().strip()
    actual = hashlib.sha256()
    actual.update((index_dir / "index.faiss").read_bytes())
    actual.update((index_dir / "index.pkl").read_bytes())
    if actual.hexdigest() != expected:
        raise RuntimeError(f"FAISS integrity mismatch at {index_dir}")


def run_smoke_test(index_dir: Path, query: str, k: int = 3) -> int:
    logger.info("Smoke test — index=%s query='%s'", index_dir, query)
    _verify_integrity(index_dir)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vector_store = FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )
    logger.info("FAISS loaded — %d vectors", vector_store.index.ntotal)

    # BM25
    corpus_path = index_dir / "bm25_corpus.json"
    bm25 = None
    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            bm25 = BM25Okapi(json.load(f))
        logger.info("BM25 loaded — %d docs", bm25.corpus_size)

    results = vector_store.similarity_search(query, k=k)
    if not results:
        print("No results found.")
        return 0

    print(f"\nTop {len(results)} results for: '{query}'")
    print("=" * 60)
    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("source", "?")
        pg = doc.metadata.get("page_number", "?")
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"\n  [{i}] {src} — p.{pg}")
        print(f"       {preview}...")

    logger.info("Smoke test PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Railway RAG smoke test")
    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="e.g. vector_indices/RAILWAY_TECHNICAL_index",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is the maximum permitted speed?",
        help="Search query",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of results to return",
    )

    args = parser.parse_args()
    index_dir = Path(args.index_dir)

    if not index_dir.is_dir():
        print(f"Error: directory not found — {index_dir}")
        sys.exit(1)

    sys.exit(run_smoke_test(index_dir, args.query, args.k))


if __name__ == "__main__":
    main()
