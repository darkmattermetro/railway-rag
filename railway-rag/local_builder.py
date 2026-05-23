# RAILWAY RAG PIPELINE — MODULE 01: CORE RULES (v8)
# RAILWAY RAG PIPELINE — MODULE 02: LOCAL BUILDER SPEC (v8)
# RAILWAY RAG PIPELINE — MODULE 06: OBSERVABILITY SPEC (v8)

"""
local_builder.py

Streamlit application for local ingestion of technical railway PDFs.
This script handles:
- PDF upload and validation.
- Resume capability for crash recovery.
- State-aware indexing.
"""

import gc
import logging
import re
import tempfile
from pathlib import Path

import torch
import streamlit as st

from ingest import ingest_pdfs

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

MAX_FILE_SIZE_BYTES: int = 200 * 1024 * 1024
CATEGORY_MAX_LEN: int = 64

def main() -> None:
    st.set_page_config(
        page_title="Railway RAG Builder",
        page_icon="🚆",
        layout="wide",
    )
    st.title("🚆 Railway RAG: Local Index Builder")
    st.markdown("Upload technical railway PDFs to create a searchable vector index.")

    st.sidebar.header("Configuration")
    category_name = st.sidebar.text_input(
        "Category Name",
        help="A name for this document set, e.g., 'Class_390_Maintenance'.",
    )

    if not category_name or not category_name.strip():
        st.info("Configure settings in the sidebar and upload files to start.")
        st.stop()

    if len(category_name.strip()) > CATEGORY_MAX_LEN:
        st.error(f"Category name is too long (max {CATEGORY_MAX_LEN} chars).")
        st.stop()

    safe_category = re.sub(r"[^A-Za-z0-9_]", "_", category_name.strip())
    if safe_category != category_name.strip():
        st.info(f"Category name sanitized to: `{safe_category}`.")

    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # Resume Logic
    state = read_ingest_state(safe_category)
    has_incomplete_session = state and len(state.get("completed_files", [])) > 0

    if has_incomplete_session:
        completed = len(state["completed_files"])
        st.sidebar.warning(f"⚠ Incomplete session detected ({completed} files completed).")
        
        if st.sidebar.button("Start Fresh"):
            clear_cache(safe_category, delete_chunks=True)
            st.rerun()
            
        build_btn_label = "Resume Indexing"
    else:
        build_btn_label = "Build Index"

    if not st.sidebar.button(build_btn_label):
        st.info(f"Upload files and click {build_btn_label} to start.")
        st.stop()

    if not uploaded_files:
        st.error("At least one PDF file must be uploaded before building.")
        st.stop()

    for f in uploaded_files:
        if f.size > MAX_FILE_SIZE_BYTES:
            st.error(f"{f.name} exceeds {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB.")
            st.stop()

    # --- Save uploads to temp directory ---
    tmp_dir = Path(tempfile.mkdtemp())
    pdf_paths: list[str] = []
    try:
        for uploaded_file in uploaded_files:
            tmp_path = tmp_dir / uploaded_file.name
            tmp_path.write_bytes(uploaded_file.getvalue())
            pdf_paths.append(str(tmp_path.resolve()))

        logger.info(
            "event=session_start file_count=%d category=%s",
            len(pdf_paths),
            safe_category,
        )

        # --- Single ingest call ---
        file_names = [Path(p).name for p in pdf_paths]
        with st.status(f"Processing {len(pdf_paths)} files...") as status:
            for name in file_names:
                status.write(f"  {name}")
            result = ingest_pdfs(pdf_paths, safe_category)

        st.success(f"✅ Index built — {result['chunk_count']} chunks in `{safe_category}_index`")
        logger.info(
            "event=session_complete chunk_count=%d category=%s",
            result["chunk_count"],
            safe_category,
        )

    except ValueError as e:
        logger.error("event=session_failed error=%s", str(e))
        st.error(str(e))
        st.stop()
    except RuntimeError as e:
        logger.error("event=session_failed error=%s", str(e))
        st.error(str(e))
        st.stop()
    finally:
        # Clean up temp files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
