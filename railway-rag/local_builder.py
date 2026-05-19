# RAILWAY RAG PIPELINE — MODULE 01: CORE RULES (v8)
# RAILWAY RAG PIPELINE — MODULE 02: LOCAL BUILDER SPEC (v8)
# RAILWAY RAG PIPELINE — MODULE 06: OBSERVABILITY SPEC (v8)

"""
local_builder.py

Streamlit application for local ingestion of technical railway PDFs.
This script handles:
- PDF upload and validation.
- Docling-based parsing with GPU acceleration and CPU fallback.
- Structural chunking of text and tables.
- OCR fallback for problematic pages.
- FAISS vector index and BM25 corpus creation.
"""

import gc
import io
import logging
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pypdf
import tiktoken
import torch
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    OcrAutoOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from google.api_core.exceptions import GoogleAPIError
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter

from ingest import process_pdf

# WARNING: Verify these APIs against installed docling version.
# Uncertain: docling_core types may be subject to change in minor versions.
# Alternative if unavailable: Check for types in docling.datamodel.
from docling_core.types.doc import (
    DoclingDocument,
    SectionHeaderItem,
    TableItem,
    TextItem,
)

# ==============================================================================
# 1. LOGGING & CONSTANTS
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Module-level constants
MAX_FILE_SIZE_BYTES: int = 200 * 1024 * 1024
CATEGORY_MAX_LEN: int = 64
OCR_MIN_TEXT_LEN: int = 20
OCR_GARBAGE_RATIO_THRESHOLD: float = 0.4
CHUNK_SIZE_TOKENS: int = 1000
CHUNK_OVERLAP_TOKENS: int = 150
CHUNK_SIZE_CHARS: int = 4000
CHUNK_OVERLAP_CHARS: int = 600
EMBEDDING_MODEL: str = "models/gemini-embedding-001"
EMBEDDING_RATE_LIMIT_WAIT_S: int = 60

# Determine device profile once at startup
_DEVICE: AcceleratorDevice = (
    AcceleratorDevice.CUDA
    if torch.cuda.is_available()
    else AcceleratorDevice.CPU
)


# ==============================================================================
# 2. CHUNKING & SPLITTING INITIALIZATION
# ==============================================================================

# Per strict exception rules, do not catch errors here.
# A failure to load the tokenizer is a fatal setup error.
_enc = tiktoken.get_encoding("cl100k_base")


def _token_length(text: str) -> int:
    """Calculate token length of a string."""
    return len(_enc.encode(text))


try:
    _splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=_token_length,
    )
except TypeError:
    # This specific catch is permitted by Rule 1.2
    logger.warning(
        "event=splitter_fallback "
        "reason=length_function_unsupported fallback=character_based"
    )
    _splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
    )


# ==============================================================================
# 3. CORE PROCESSING FUNCTIONS
# ==============================================================================


def build_pipeline_options(
    device: AcceleratorDevice,
    ocr_batch: int = 4,
    layout_batch: int = 2,
    table_batch: int = 1,
    num_threads: int = 4,
) -> PdfPipelineOptions:
    """Builds Docling pipeline options safe for GTX 1650 and CPU."""
    if device == AcceleratorDevice.CPU:
        # Constrain CPU threads to avoid RAM pressure on smaller instances
        num_threads = 2

    return PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            device=device,
            num_threads=num_threads,
        ),
        do_ocr=True,
        ocr_options=OcrAutoOptions(),
        ocr_batch_size=ocr_batch,
        layout_batch_size=layout_batch,
        table_batch_size=table_batch,
    )


def _should_trigger_ocr_fallback(page_text: str) -> bool:
    """Check if OCR fallback should be triggered based on text quality only.

    Returns True if the text is:
    - Short and contains no alphanumeric characters (likely garbage/scanned image), OR
    - Has a high ratio of non-alphanumeric characters (garbage ratio > threshold)

    This is the standalone trigger check, without the per-page cache guard.
    The caller (extract_chunks) applies page_no not in ocr_fallback_cache.
    """
    # Condition 1: short text without any alphanumeric content
    if len(page_text) < OCR_MIN_TEXT_LEN and not re.search(r"[A-Za-z0-9]", page_text):
        return True

    # Condition 2: high garbage ratio
    clean = re.sub(r"[^a-zA-Z0-9\s#\-\|\*\_\`]", "", page_text)
    garbage_ratio = 1.0 - (len(clean) / max(len(page_text), 1))
    return garbage_ratio > OCR_GARBAGE_RATIO_THRESHOLD


def perform_ocr_fallback(
    source_pdf_path: Path, page_no: int, filename: str
) -> str | None:
    """Extracts a single page and forces OCR on it using CPU."""
    ocr_tmp_path: Path | None = None
    reprocessed_text: str | None = None
    try:
        reader = pypdf.PdfReader(source_pdf_path)
        writer = pypdf.PdfWriter()
        writer.add_page(reader.pages[page_no - 1])  # page_no is 1-based

        page_bytes_io = io.BytesIO()
        writer.write(page_bytes_io)
        page_bytes_io.seek(0)

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pdf", delete=False
        ) as ocr_tmp_file:
            ocr_tmp_path = Path(ocr_tmp_file.name)
            ocr_tmp_file.write(page_bytes_io.getvalue())

        # Re-run Docling with forced OCR on the single page
        cpu_ocr_options = build_pipeline_options(
            device=AcceleratorDevice.CPU,
            ocr_batch=1,
            layout_batch=1,
            table_batch=1,
        )
        cpu_ocr_options.do_ocr = True
        cpu_ocr_options.ocr_options.force_full_page_ocr = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=cpu_ocr_options
                )
            }
        )
        result = converter.convert(ocr_tmp_path)
        ocr_doc: DoclingDocument = result.document
        reprocessed_text = ocr_doc.export_to_text()
        logger.info(
            "event=ocr_fallback_success source=%s page=%d", filename, page_no
        )
        return reprocessed_text
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "alloc" in str(e).lower():
            logger.error(
                "event=ocr_fallback_oom source=%s page=%d", filename, page_no
            )
        else:
            logger.error(
                "event=ocr_fallback_failed source=%s page=%d error=%s",
                filename,
                page_no,
                str(e),
            )
            # Re-raise if not an OOM error during fallback
            raise
    finally:
        if ocr_tmp_path is not None and ocr_tmp_path.exists():
            ocr_tmp_path.unlink()
    return reprocessed_text


def extract_chunks(
    doc: DoclingDocument,
    filename: str,
    category: str,
    source_pdf_path: Path,
) -> list[Document]:
    """Extracts structured, metadata-rich chunks from a DoclingDocument."""
    chunks: list[Document] = []
    prose_buffer: list[str] = []
    buffer_page_no: int = 1
    current_heading: str = "Introduction"
    last_known_page: int = 1
    last_element_was_table: bool = False
    last_table_page: int = -1
    last_table_heading: str = ""
    ocr_fallback_cache: dict[int, str | None] = {}
    ocr_replaced_pages: set[int] = set()

    def flush_prose_buffer() -> None:
        nonlocal prose_buffer, buffer_page_no
        if not prose_buffer:
            return
        full_prose = f"## {current_heading}\n\n" + "\n".join(prose_buffer)
        prose_chunks = _splitter.split_text(full_prose)
        for prose_chunk in prose_chunks:
            metadata = {
                "source": Path(filename).name,
                "page_number": buffer_page_no,
                "category": category,
                "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
                "is_table": False,
                "is_table_continuation": False,
            }
            chunks.append(Document(page_content=prose_chunk, metadata=metadata))
        prose_buffer = []
        buffer_page_no = last_known_page

    for element, level in doc.iterate_items():
        try:
            if hasattr(element, "prov") and element.prov:
                page_no = element.prov[0].page_no
                last_known_page = page_no
            else:
                page_no = last_known_page
        except (IndexError, TypeError):
            page_no = last_known_page

        if isinstance(element, SectionHeaderItem):
            flush_prose_buffer()
            heading_text = element.text.strip() if element.text else ""
            if heading_text:
                metadata = {
                    "source": Path(filename).name,
                    "page_number": last_known_page,
                    "category": category,
                    "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
                    "is_table": False,
                    "is_table_continuation": False,
                    "is_section_header": True,
                }
                chunks.append(Document(page_content=f"## {heading_text}", metadata=metadata))
            current_heading = heading_text
            last_element_was_table = False

        elif isinstance(element, TableItem):
            flush_prose_buffer()
            table_markdown = element.export_to_markdown().strip()
            if not table_markdown:
                logger.warning(
                    "event=empty_table_skipped source=%s page=%d heading=%s",
                    filename,
                    page_no,
                    current_heading,
                )
                last_element_was_table = False
                continue

            is_continuation = (
                last_element_was_table
                and current_heading == last_table_heading
                and (
                    page_no == last_table_page or page_no == last_table_page + 1
                )
            )

            metadata = {
                "source": Path(filename).name,
                "page_number": page_no,
                "category": category,
                "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
                "is_table": True,
                "is_table_continuation": is_continuation,
            }
            contextual_table = f"### {current_heading}\n\n{table_markdown}"
            chunks.append(
                Document(page_content=contextual_table, metadata=metadata)
            )
            last_element_was_table = True
            last_table_page = page_no
            last_table_heading = current_heading

        elif isinstance(element, TextItem):
            page_text = element.text.strip() if element.text else ""
            if not page_text:
                continue

            if page_no in ocr_replaced_pages:
                continue

            trigger_ocr_fallback = (
                _should_trigger_ocr_fallback(page_text)
                and page_no not in ocr_fallback_cache
            )

            if trigger_ocr_fallback:
                logger.warning(
                    "event=ocr_trigger source=%s page=%d text_len=%d",
                    filename,
                    page_no,
                    len(page_text),
                )
                reprocessed_text = perform_ocr_fallback(
                    source_pdf_path, page_no, filename
                )
                ocr_fallback_cache[page_no] = reprocessed_text
                if reprocessed_text:
                    page_text = reprocessed_text.strip()
                    ocr_replaced_pages.add(page_no)

            if page_no != buffer_page_no:
                flush_prose_buffer()

            prose_buffer.append(page_text)
            last_element_was_table = False

    flush_prose_buffer()
    return chunks


def main() -> None:
    """Main Streamlit application entrypoint."""
    import streamlit as st
    st.set_page_config(
        page_title="Railway RAG Builder",
        page_icon="🚆",
        layout="wide",
    )
    st.title("🚆 Railway RAG: Local Index Builder")
    st.markdown(
        "Upload technical railway PDFs to create a searchable vector index."
    )

    # --- UI & Input Validation ---
    st.sidebar.header("Configuration")
    google_api_key = st.sidebar.text_input(
        "Google API Key", type="password", help="Required for embeddings."
    )
    category_name = st.sidebar.text_input(
        "Category Name",
        help="A name for this document set, e.g., 'Class_390_Maintenance'.",
    )
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if not st.sidebar.button("Build Index"):
        st.info("Configure settings in the sidebar and click 'Build Index'.")
        st.stop()

    if not google_api_key:
        st.error("Google API Key is required.")
        st.stop()
    if not category_name or not category_name.strip():
        st.error("Category Name is required.")
        st.stop()

    safe_category = re.sub(r"[^A-Za-z0-9_]", "_", category_name.strip())
    if safe_category != category_name.strip():
        st.info(
            f"Category name sanitized to: `{safe_category}`. "
            "This will be the directory name."
        )

    if not safe_category or len(safe_category) > CATEGORY_MAX_LEN:
        st.error(
            f"Sanitized category name is invalid or too long "
            f"(max {CATEGORY_MAX_LEN} chars)."
        )
        st.stop()

    if not uploaded_files:
        st.error("At least one PDF file must be uploaded.")
        st.stop()

    for f in uploaded_files:
        if f.size > MAX_FILE_SIZE_BYTES:
            msg = f"{f.name} exceeds {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB."
            logger.error(
                "event=file_too_large source=%s size=%d", f.name, f.size
            )
            st.error(msg)
            st.stop()

    # --- Processing Loop ---
    all_chunks: list[Document] = []
    per_file_stats: list[tuple[str, int, float]] = []
    t_session_start = time.monotonic()
    total_bytes = sum(f.size for f in uploaded_files if f.size)
    logger.info(
        "event=session_start file_count=%d total_bytes=%d category=%s",
        len(uploaded_files),
        total_bytes,
        safe_category,
    )

    progress_bar = st.progress(0, "Starting ingestion...")
    status_text = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        tmp_path: Path | None = None
        filename = uploaded_file.name
        status_text.info(f"Processing ({i+1}/{len(uploaded_files)}): {filename}")
        progress_bar.progress(
            (i / len(uploaded_files)), text=f"Processing: {filename}"
        )

        logger.info(
            "event=pdf_start source=%s size_bytes=%d index=%d total=%d",
            filename,
            uploaded_file.size,
            i + 1,
            len(uploaded_files),
        )

        doc_processed_successfully = False
        file_chunks: list[Document] = []

        fd: int | None = None
        try:
            fd, tmp_path_str = tempfile.mkstemp(suffix=".pdf")
            tmp_path = Path(tmp_path_str)
            os.close(fd)
            fd = None
            tmp_path.write_bytes(uploaded_file.getvalue())

            file_chunks, elapsed, success = process_pdf(
                tmp_path, filename, safe_category
            )
            if success:
                doc_processed_successfully = True
        except (RuntimeError, GoogleAPIError) as e:
            logger.error("event=pdf_failed source=%s err_type=%s err=%s", filename, type(e).__name__, str(e))
            st.error(f"Failed to process {filename}: {e}")
        finally:
            if doc_processed_successfully:
                all_chunks.extend(file_chunks)
                per_file_stats.append((filename, len(file_chunks), elapsed))
                logger.info(
                    "event=pdf_complete source=%s chunks=%d elapsed_s=%.2f",
                    filename,
                    len(file_chunks),
                    elapsed,
                )

            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress_bar.progress(1.0, text="Processing complete. Building index...")
    status_text.info(
        f"Processed {len(uploaded_files)} files, "
        f"extracted {len(all_chunks)} total chunks. Now building index..."
    )

    # --- Guard & Profiling ---
    if not all_chunks:
        logger.error("event=no_chunks source=all files")
        st.error(
            "No text chunks were extracted. "
            "Please check if the uploaded PDFs contain readable text."
        )
        st.stop()

    st.subheader("Ingestion Profile")
    if per_file_stats:
        profile_data = [
            {
                "File": filename,
                "Chunks": count,
                "Elapsed (s)": f"{elapsed:.2f}",
                "Chunks/s": f"{count / max(elapsed, 0.01):.1f}",
            }
            for filename, count, elapsed in per_file_stats
        ]
        st.dataframe(profile_data, use_container_width=True)

    total_elapsed = time.monotonic() - t_session_start
    c1, c2 = st.columns(2)
    c1.metric("Total Chunks", len(all_chunks))
    c2.metric("Total Time (s)", f"{total_elapsed:.2f}")

    # --- Index Compilation ---
    st.subheader("Index Compilation")
    with st.spinner("Embedding documents and building FAISS index..."):
        from ingest import build_and_save_index
        try:
            index_dir = build_and_save_index(all_chunks, safe_category, google_api_key)
            logger.info(
                "event=session_complete total_chunks=%d total_elapsed_s=%.2f files=%d",
                len(all_chunks),
                total_elapsed,
                len(uploaded_files),
            )
            st.success(f"✅ Index built and saved to `{index_dir}`")
        except RuntimeError as exc:
            logger.error("event=index_build_failed error=%s", str(exc))
            st.error(str(exc))
            st.stop()


if __name__ == "__main__":
    main()
