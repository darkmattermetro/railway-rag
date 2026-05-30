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

from ingest import ingest_pdfs, read_ingest_state, clear_cache

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

    tmp_dir = Path(tempfile.mkdtemp())
    pdf_paths: list[str] = []
    try:
        for uploaded_file in uploaded_files:
            tmp_path = tmp_dir / uploaded_file.name
            tmp_path.write_bytes(uploaded_file.getvalue())
            pdf_paths.append(str(tmp_path.resolve()))

        import time
        class ProgressState:
            def __init__(self):
                self.start_time = time.time()
                self.global_pages_done = 0
                self.global_total_pages = 0
                self.cached_files = set()
                self.file_page_counts = {}

        pstate = ProgressState()

        progress_container = st.container()
        progress_bar = progress_container.progress(0.0)
        time_text = progress_container.empty()
        file_status = progress_container.empty()
        page_detail = progress_container.empty()
        
        def update_progress():
            if pstate.global_total_pages > 0:
                prog = min(pstate.global_pages_done / pstate.global_total_pages, 1.0)
                progress_bar.progress(prog)
                
                elapsed = time.time() - pstate.start_time
                if pstate.global_pages_done > 0:
                    pages_remaining = pstate.global_total_pages - pstate.global_pages_done
                    time_per_page = elapsed / pstate.global_pages_done
                    eta = pages_remaining * time_per_page
                    
                    if pages_remaining > 0:
                        em, es = divmod(int(elapsed), 60)
                        mins, secs = divmod(int(eta), 60)
                        time_text.text(f"⏱ Elapsed: {em}m {es}s | Estimated remaining: {mins}m {secs}s")
                    else:
                        time_text.text("")
                else:
                    em, es = divmod(int(elapsed), 60)
                    time_text.text(f"⏱ Elapsed: {em}m {es}s | Estimating time remaining...")
        
        def ui_callback(event, filename, val1, val2, dev=None, warn=None):
            if event == "INIT":
                pstate.global_total_pages = val2
                pstate.file_page_counts = dev or {}
                update_progress()
            elif event == "FILE_CACHED":
                pstate.cached_files.add(filename)
                pages_in_file = pstate.file_page_counts.get(filename, 0)
                pstate.global_pages_done += pages_in_file
                file_status.info(f"📄 File {val1}/{val2}: `{filename}` (cached)")
                page_detail.info(f"⏭ Loaded {pages_in_file} pages from cache")
                update_progress()
            elif event == "FILE_START":
                file_status.info(f"📄 File {val1}/{val2}: `{filename}`")
            elif event == "FILE_SPLITTING":
                page_detail.info(f"✂️ Splitting PDF into pages...")
            elif event == "PAGE_PROG":
                pstate.global_pages_done += 1
                dev_str = f"[{dev}]" if dev else ""
                page_detail.text(f"📖 Page {val1}/{val2}: Converting {dev_str}")
                update_progress()
            elif event == "PAGE_OCR_RETRY":
                page_detail.text(f"📖 Page {val1}/{val2}: OCR retry [CPU]")
            elif event == "FILE_CHUNKING":
                page_detail.info(f"📦 Extracting chunks...")
            elif event == "BUILDING_INDEX":
                progress_bar.progress(0.0)
                elapsed = time.time() - pstate.start_time
                em, es = divmod(int(elapsed), 60)
                time_text.text(f"⏱ Elapsed: {em}m {es}s | Building index...")
                page_detail.info("🔨 Building FAISS & BM25 index...")
            elif event == "INDEX_PROGRESS":
                prog = val1 / val2 if val2 > 0 else 0.0
                progress_bar.progress(prog)
                page_detail.info(f"🔨 Indexing: {val1}/{val2} chunks...")

        result = ingest_pdfs(pdf_paths, safe_category, ui_callback=ui_callback)

        elapsed = time.time() - pstate.start_time
        em, es = divmod(int(elapsed), 60)
        st.success(f"✅ Index built — {result['chunk_count']} chunks in `{safe_category}_index` (⏱ {em}m {es}s)")
        
    except Exception as e:
        logger.exception("Ingestion failed")
        st.error(f"Error during ingestion: {str(e)}")
        st.stop()
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
