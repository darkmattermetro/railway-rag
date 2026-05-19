"""Shared ingestion pipeline for local_builder.py and build_index.py."""

import gc
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import torch
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def build_and_save_index(
    all_chunks: list[Document],
    category: str,
    google_api_key: str,
) -> Path:
    from google.api_core.exceptions import ResourceExhausted
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from local_builder import EMBEDDING_MODEL, EMBEDDING_RATE_LIMIT_WAIT_S

    # Rule 1.2: import error propagates naturally to caller's RuntimeError handler
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, google_api_key=google_api_key
    )

    vector_store: FAISS | None = None
    for attempt in range(2):
        try:
            vector_store = FAISS.from_documents(all_chunks, embeddings)
            break
        except ResourceExhausted:
            if attempt == 0:
                logger.warning(
                    "event=embedding_rate_limit attempt=1 waiting=%ds",
                    EMBEDDING_RATE_LIMIT_WAIT_S,
                )
                time.sleep(EMBEDDING_RATE_LIMIT_WAIT_S)
            else:
                raise RuntimeError("Embedding rate limit hit again. Aborting index build.")

    if not vector_store:
        raise RuntimeError("Vector store was not built.")

    index_dir = Path("vector_indices") / f"{category}_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))

    faiss_hash = hashlib.sha256()
    faiss_hash.update((index_dir / "index.faiss").read_bytes())
    faiss_hash.update((index_dir / "index.pkl").read_bytes())
    (index_dir / "index.hash").write_text(faiss_hash.hexdigest())
    logger.info("event=index_saved path=%s", index_dir)

    corpus: list[dict] = [
        {"text": doc.page_content, "metadata": doc.metadata}
        for doc in all_chunks
    ]
    corpus_path = index_dir / "bm25_corpus.json"
    corpus_path.write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "event=bm25_corpus_saved path=%s docs=%d",
        corpus_path,
        len(corpus),
    )
    return index_dir


def process_pdf(
    pdf_path: Path,
    filename: str,
    category: str,
) -> tuple[list[Document], float, bool]:
    from docling.datamodel.accelerator_options import AcceleratorDevice
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from local_builder import build_pipeline_options, _DEVICE, extract_chunks

    tmp_path = pdf_path
    t_start = time.monotonic()
    file_chunks: list[Document] = []
    success = False

    try:
        gpu_options = build_pipeline_options(device=_DEVICE)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=gpu_options)
            }
        )
        result = converter.convert(tmp_path)
        doc = result.document
        file_chunks = extract_chunks(doc, filename, category, tmp_path)
        success = True

    except RuntimeError as e:
        oom_keywords = ["out of memory", "alloc", "cuda error"]
        if any(m in str(e).lower() for m in oom_keywords):
            logger.warning("event=oom_fallback source=%s error=%s", filename, str(e))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if tmp_path and tmp_path.exists():
                try:
                    cpu_options = build_pipeline_options(
                        device=AcceleratorDevice.CPU,
                        ocr_batch=1,
                        layout_batch=1,
                        table_batch=1,
                    )
                    converter = DocumentConverter(
                        format_options={
                            InputFormat.PDF: PdfFormatOption(pipeline_options=cpu_options)
                        }
                    )
                    result = converter.convert(tmp_path)
                    doc = result.document
                    file_chunks = extract_chunks(doc, filename, category, tmp_path)
                    success = True
                except RuntimeError as cpu_e:
                    logger.error(
                        "event=cpu_fallback_failed source=%s error=%s",
                        filename,
                        str(cpu_e),
                    )
        else:
            logger.error(
                "event=pdf_failed source=%s error=%s", filename, str(e)
            )

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.monotonic() - t_start
    return file_chunks, elapsed, success
