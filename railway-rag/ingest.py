"""Ingestion pipeline for Railway RAG."""

import gc
import hashlib
import io
import json
import logging
import re
import tempfile
import uuid
from pathlib import Path

import numpy as np
import tiktoken
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from utils import (
    Chunk,
    count_tokens,
    GARBAGE_RATIO_THRESHOLD,
)

logger = logging.getLogger(__name__)

EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE: int = 400
CHUNK_OVERLAP: int = 50


def _garbage_ratio(text: str) -> float:
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return 1.0 - (len(clean) / max(len(text), 1))


def _split_page_text(text: str) -> list[tuple[str, int, int]]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks: list[tuple[str, int, int]] = []
    pos = 0
    while pos < len(tokens):
        end = min(pos + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[pos:end]
        chunk_text = enc.decode(chunk_tokens)
        char_start = text.find(chunk_text)
        if char_start == -1:
            char_start = 0
        char_end = char_start + len(chunk_text)
        chunks.append((chunk_text, char_start, char_end))
        if end >= len(tokens):
            break
        pos += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _extract_page_texts(pdf_path: Path) -> dict[int, str]:
    """Extract text per page from a PDF using Docling. Returns {page_no: text}."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    page_map: dict[int, list[str]] = {}
    for item, _level in doc.iterate_items():
        try:
            if hasattr(item, "text") and item.text:
                page_no = item.prov[0].page_no if item.prov else 1
                page_map.setdefault(page_no, []).append(item.text.strip())
        except (IndexError, TypeError, AttributeError):
            pass

    return {p: "\n".join(lines) for p, lines in page_map.items()}


def _ocr_page(pdf_path: Path, page_no: int) -> str:
    """Re-run OCR on a single page."""
    import pypdf
    from docling.datamodel.accelerator_options import AcceleratorDevice
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        OcrAutoOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    reader = pypdf.PdfReader(pdf_path)
    writer = pypdf.PdfWriter()
    writer.add_page(reader.pages[page_no - 1])
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode="wb") as f:
        f.write(buf.getvalue())
        ocr_path = Path(f.name)

    try:
        opts = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=OcrAutoOptions(force_full_page_ocr=True),
            accelerator_options={"device": AcceleratorDevice.CPU},
        )
        ocr_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        ocr_result = ocr_converter.convert(ocr_path)
        texts: list[str] = []
        for item, _level in ocr_result.document.iterate_items():
            if hasattr(item, "text") and item.text:
                texts.append(item.text.strip())
        return "\n".join(texts)
    finally:
        ocr_path.unlink(missing_ok=True)


def ingest_pdfs(pdf_paths: list[str], category: str) -> dict:
    """Parse PDFs, chunk, embed with BAAI/bge-base-en-v1.5, build FAISS + BM25.

    Args:
        pdf_paths: List of paths to PDF files.
        category: Collection name for the index directory.

    Returns:
        {"chunk_count": int, "category": str}
    """
    all_chunks: list[Chunk] = []

    for pdf_path_str in pdf_paths:
        pdf_path = Path(pdf_path_str)
        filename = pdf_path.name
        logger.info("event=pdf_start file=%s", filename)

        page_texts = _extract_page_texts(pdf_path)

        for page_no in sorted(page_texts.keys()):
            page_text = page_texts[page_no]

            if _garbage_ratio(page_text) > GARBAGE_RATIO_THRESHOLD:
                logger.warning("event=ocr_trigger file=%s page=%d", filename, page_no)
                page_text = _ocr_page(pdf_path, page_no)

            splits = _split_page_text(page_text)
            source_stem = Path(filename).stem
            for idx, (chunk_text, char_start, char_end) in enumerate(splits):
                chunk_id = f"{source_stem}_p{page_no}_{idx}"
                all_chunks.append(Chunk(
                    text=chunk_text,
                    source_file=filename,
                    page_number=page_no,
                    chunk_id=chunk_id,
                    char_start=char_start,
                    char_end=char_end,
                    token_count=count_tokens(chunk_text),
                ))

        logger.info("event=pdf_complete file=%s", filename)

    if not all_chunks:
        logger.error("event=no_chunks")
        raise ValueError("No chunks extracted from PDFs")

    # ----------------------------------------------------------------
    # Embed with BAAI/bge-base-en-v1.5
    # ----------------------------------------------------------------
    logger.info("event=embedding_start chunk_count=%d", len(all_chunks))
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c.text for c in all_chunks]
    embeddings_list: list[list[float]] = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_emb = model.encode(batch, show_progress_bar=False)
        embeddings_list.extend(batch_emb.tolist())

    # ----------------------------------------------------------------
    # Build FAISS IndexFlatIP + save in LangChain-compatible format
    # ----------------------------------------------------------------
    import faiss

    index_dir = Path("vector_indices") / f"{category}_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    embeddings_np = np.array(embeddings_list).astype("float32")
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)

    documents = [
        Document(
            page_content=c.text,
            metadata={
                "source": c.source_file,
                "page_number": c.page_number,
                "chunk_id": c.chunk_id,
            },
        )
        for c in all_chunks
    ]
    docstore_ids = [str(uuid.uuid4()) for _ in documents]
    docstore = InMemoryDocstore(dict(zip(docstore_ids, documents)))
    index_to_docstore_id = dict(enumerate(docstore_ids))

    def _embed_query(text: str) -> list[float]:
        return model.encode(text).tolist()

    vector_store = FAISS(
        embedding_function=_embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )
    vector_store.save_local(str(index_dir))

    # Integrity hash
    faiss_hash = hashlib.sha256()
    faiss_hash.update((index_dir / "index.faiss").read_bytes())
    faiss_hash.update((index_dir / "index.pkl").read_bytes())
    (index_dir / "index.hash").write_text(faiss_hash.hexdigest())
    logger.info("event=index_saved path=%s", index_dir)

    # ----------------------------------------------------------------
    # Build BM25 tokenised corpus
    # ----------------------------------------------------------------
    tokenised_corpus: list[list[str]] = [c.text.lower().split() for c in all_chunks]
    corpus_path = index_dir / "bm25_corpus.json"
    corpus_path.write_text(
        json.dumps(tokenised_corpus, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("event=bm25_corpus_saved path=%s", corpus_path)

    gc.collect()
    return {"chunk_count": len(all_chunks), "category": category}
