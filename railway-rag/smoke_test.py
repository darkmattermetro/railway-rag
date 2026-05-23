#!venv311/Scripts/python.exe
"""
Railway RAG Smoke Test - Standalone CLI validation script
"""
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_index(index_dir: Path) -> FAISS:
    """
    Load FAISS index from disk (saved directly in <category>_index/).
    """
    logger.info("Loading FAISS index from %s", index_dir)

    hash_path = index_dir / "index.hash"
    if hash_path.exists():
        expected = hash_path.read_text().strip()
        actual = hashlib.sha256()
        actual.update((index_dir / "index.faiss").read_bytes())
        actual.update((index_dir / "index.pkl").read_bytes())
        if actual.hexdigest() != expected:
            raise RuntimeError(f"FAISS index integrity mismatch at {index_dir}")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # The builder saves index.faiss/index.pkl directly in index_dir (not a subfolder)
    try:
        faiss_index = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Successfully loaded FAISS index with %d vectors", faiss_index.index.ntotal)
        return faiss_index
    except RuntimeError as e:
        logger.error("Failed to load FAISS index: %s", e)
        raise RuntimeError(f"Could not load FAISS index: {e}") from e


def load_bm25_corpus(index_dir: Path) -> list[Document]:
    """
    Load BM25 corpus from JSON file (bm25_corpus.json in index_dir).
    Corpus is stored as list of dicts: [{"text": "...", "metadata": {...}}, ...]
    """
    corpus_path = index_dir / "bm25_corpus.json"
    corpus_data: list[dict] = json.loads(corpus_path.read_text(encoding="utf-8"))
    documents = [
        Document(page_content=item["text"], metadata=item["metadata"])
        for item in corpus_data
    ]
    logger.info("Loaded %d documents from BM25 corpus", len(documents))
    return documents


def run_smoke_test(
    index_dir: Path,
    query: str,
) -> int:
    """Run the smoke test."""
    try:
        logger.info("Starting smoke test with query: '%s'", query)
        faiss_index = load_index(index_dir)

        results = faiss_index.similarity_search(query, k=3)

        if not results:
            logger.warning("No results returned for query: '%s'", query)
            print("No results found for the query.")
            return 0

        print(f"\nTop {len(results)} results for query: '{query}'")
        print("=" * 60)

        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            page_number = doc.metadata.get("page_number", "Unknown")
            content_preview = doc.page_content[:200].replace('\n', ' ')

            print(f"\nResult {i}:")
            print(f"  Source: {source}")
            print(f"  Page: {page_number}")
            print(f"  Content: {content_preview}...")

        logger.info("Smoke test completed successfully")
        return 0

    except RuntimeError as e:
        logger.error("Smoke test failed: %s", e, exc_info=True)
        print(f"Error: {e}")
        return 1


def main() -> None:
    """Main entry point for the smoke test."""
    parser = argparse.ArgumentParser(description="Railway RAG smoke test")
    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="Directory containing FAISS index and BM25 corpus (e.g., vector_indices/RAILWAY_TECHNICAL_index)",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query to test",
    )
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        logger.error("Index directory does not exist: %s", index_dir)
        print(f"Error: Index directory '{index_dir}' does not exist")
        sys.exit(1)

    exit_code = run_smoke_test(index_dir, args.query)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
