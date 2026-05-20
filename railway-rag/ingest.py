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
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def build_and_save_index(
    all_chunks: list[Document],
    category: str,
) -> Path:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from local_builder import EMBEDDING_MODEL

    # Rule 1.2: import error propagates naturally to caller's RuntimeError handler
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Create vector store from documents
    vector_store = FAISS.from_documents(all_chunks, embeddings)

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
    from datetime import datetime, timezone

    import pypdf
    from docling.datamodel.accelerator_options import AcceleratorDevice
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from local_builder import (
        build_pipeline_options,
        _DEVICE,
        extract_chunks,
        perform_ocr_fallback,
    )

    tmp_path = pdf_path
    t_start = time.monotonic()
    file_chunks: list[Document] = []
    success = False

    # Get total page count to detect dropped pages later
    total_pages = 0
    try:
        reader = pypdf.PdfReader(tmp_path)
        total_pages = len(reader.pages)
    except Exception:
        pass

    try:
        pipeline_opts = build_pipeline_options(device=_DEVICE)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
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

    else:
        # Page-level recovery: reprocess any pages silently dropped by docling
        if total_pages > 0:
            pages_in_chunks: set[int] = set()
            for chunk in file_chunks:
                pn = chunk.metadata.get("page_number")
                if pn is not None:
                    pages_in_chunks.add(int(pn))

            missing = sorted(set(range(1, total_pages + 1)) - pages_in_chunks)
            if missing:
                logger.warning(
                    "event=missing_pages source=%s total=%d present=%d missing_count=%d",
                    filename,
                    total_pages,
                    len(pages_in_chunks),
                    len(missing),
                )
                for page_no in missing:
                    try:
                        recovered = perform_ocr_fallback(tmp_path, page_no, filename)
                        if recovered and recovered.strip():
                            metadata = {
                                "source": Path(filename).name,
                                "page_number": page_no,
                                "category": category,
                                "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
                                "is_table": False,
                                "is_table_continuation": False,
                                "is_recovered": True,
                            }
                            file_chunks.append(
                                Document(page_content=recovered.strip(), metadata=metadata)
                            )
                            logger.info(
                                "event=page_recovered source=%s page=%d text_len=%d",
                                filename,
                                page_no,
                                len(recovered),
                            )
                    except Exception as page_e:
                        logger.warning(
                            "event=page_recovery_failed source=%s page=%d error=%s",
                            filename,
                            page_no,
                            str(page_e),
                        )

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.monotonic() - t_start
    return file_chunks, elapsed, success
