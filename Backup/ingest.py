"""Ingestion pipeline for Railway RAG."""
import gc
import hashlib
import json
import logging
import os
import re
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Iterator, Callable
from datetime import datetime, timezone

import pypdf
import pypdfium2 as pdfium
import tiktoken
import torch

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import OcrAutoOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, SectionHeaderItem, TableItem, TextItem
from docling.exceptions import ConversionError

logger = logging.getLogger(__name__)

# Paths
_BUILD_DIR = Path(__file__).resolve().parent / "vector_indices"
_CACHE_DIR = _BUILD_DIR / ".chunks_cache"

# Constants
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE_TOKENS: int = 384
CHUNK_OVERLAP_TOKENS: int = 64
CHUNK_SIZE_CHARS: int = 1500
CHUNK_OVERLAP_CHARS: int = 128

OCR_MIN_TEXT_LEN: int = 20
OCR_GARBAGE_RATIO_THRESHOLD: float = 0.4
HAS_CUDA: bool = torch.cuda.is_available()

# Page dimension thresholds for GPU sizing check (~A1 paper = 1684 x 2384 pt)
# Pages exceeding 2000pt in any dimension skip GPU to prevent OOM
MAX_PAGE_WIDTH_PT: int = 2000
MAX_PAGE_HEIGHT_PT: int = 2000

# Cache format version — bump to invalidate old contaminated caches
CACHE_VERSION: int = 2

# FAISS batch size for incremental index building (overridable via env)
_FAISS_BATCH_SIZE_ENV = os.environ.get("FAISS_BATCH_SIZE")
FAISS_BATCH_SIZE: int = int(_FAISS_BATCH_SIZE_ENV) if _FAISS_BATCH_SIZE_ENV else 500

# VRAM threshold (GB) below which batch size is halved
LOW_VRAM_THRESHOLD_GB: float = 2.0

# --- State & Caching ---

def _get_file_hash(pdf_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(pdf_path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def _save_chunks_cache(file_hash: str, jsonl_path: Path):
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"{file_hash}.json"
    chunks = list(_iter_jsonl(jsonl_path))
    data = {
        "version": CACHE_VERSION,
        "chunks": chunks,
    }
    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    page_dir = _CACHE_DIR / f"{file_hash}.pages"
    if page_dir.is_dir():
        shutil.rmtree(page_dir, ignore_errors=True)

def _load_chunks_cache(file_hash: str) -> Optional[List[Document]]:
    cache_path = _CACHE_DIR / f"{file_hash}.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("version") == CACHE_VERSION:
            return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data["chunks"]]
        return None
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to load cache from {cache_path}: {e}")
        return None

def _save_page_chunks(file_hash: str, page_no: int, chunks: List[Document]):
    page_dir = _CACHE_DIR / f"{file_hash}.pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    path = page_dir / f"page_{page_no:04d}.json"
    data = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

def _load_page_chunks(file_hash: str, page_no: int) -> Optional[List[Document]]:
    path = _CACHE_DIR / f"{file_hash}.pages" / f"page_{page_no:04d}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to load page cache {path.name}: {e}")
        return None

def read_ingest_state(category: str) -> dict:
    state_file = _BUILD_DIR / f".ingest_state_{category}.json"
    if state_file.exists():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to load state from {state_file}: {e}")
    return {"completed_files": [], "total_files": 0}

def _write_ingest_state(category: str, state: dict):
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    state_file = _BUILD_DIR / f".ingest_state_{category}.json"
    tmp = state_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(state_file)

def clear_cache(category: str, delete_chunks: bool = False):
    state_file = _BUILD_DIR / f".ingest_state_{category}.json"
    if state_file.exists():
        state_file.unlink()
    if delete_chunks and _CACHE_DIR.exists():
        shutil.rmtree(_CACHE_DIR, ignore_errors=True)

# Temp directory for streaming pipeline
_TMP_DIR = _BUILD_DIR / ".tmp_streaming"

# --- Streaming JSONL helpers ---

def _write_chunks_jsonl(path: Path, chunks: List[Document]):
    """Append chunks to a JSONL file (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "page_content": c.page_content,
                "metadata": c.metadata,
            }, ensure_ascii=False) + "\n")

def _iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield chunk dicts from a JSONL file, one at a time."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# --- Dedup ---

def _normalize_chunk(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())

def _stream_deduplicate(input_paths: List[Path], output_path: Path) -> int:
    """Stream-read chunks from multiple JSONL files, dedup, write deduped JSONL.
    Returns the number of duplicates removed. Only one chunk in RAM at a time."""
    seen: set[str] = set()
    total = 0
    deduped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for p in input_paths:
            for chunk_dict in _iter_jsonl(p):
                total += 1
                key = hashlib.sha256(_normalize_chunk(chunk_dict["page_content"]).encode()).hexdigest()
                if key not in seen:
                    seen.add(key)
                    out.write(json.dumps(chunk_dict, ensure_ascii=False) + "\n")
                    deduped += 1
    return total - deduped

# --- Docling Processing ---

class _MergedDocument:
    def __init__(self):
        self.items = []

    def add_page_doc(self, p_doc: DoclingDocument, page_no: int):
        for item, level in p_doc.iterate_items():
            if hasattr(item, "prov") and item.prov:
                for p in item.prov:
                    p.page_no = page_no
            self.items.append((item, level))

    def iterate_items(self):
        return iter(self.items)

def build_pipeline_options(device: AcceleratorDevice) -> PdfPipelineOptions:
    num_threads = 2 if device == AcceleratorDevice.CPU else 1
    return PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            device=device,
            num_threads=num_threads,
        ),
        do_ocr=True,
        ocr_options=OcrAutoOptions(force_full_page_ocr=False),
        ocr_batch_size=1,
        layout_batch_size=1,
        table_batch_size=1,
    )

def _split_pdf_pages(pdf_path: Path, tmp_dir: Path) -> Iterator[tuple[int, Path, int]]:
    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    for i, page in enumerate(reader.pages):
        writer = pypdf.PdfWriter()
        writer.add_page(page)
        tmp_path = tmp_dir / f"page_{i+1}.pdf"
        with open(tmp_path, "wb") as out:
            writer.write(out)
        yield i + 1, tmp_path, total_pages

class ConverterCache:
    def __init__(self):
        self.gpu_conv = None
        self.cpu_conv = None
        self.cpu_ocr_conv = None
        self.gpu_ocr_conv = None
        
    def get_gpu(self):
        if self.gpu_conv is None:
            opts = build_pipeline_options(AcceleratorDevice.CUDA)
            self.gpu_conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        return self.gpu_conv
        
    def get_cpu(self):
        if self.cpu_conv is None:
            opts = build_pipeline_options(AcceleratorDevice.CPU)
            self.cpu_conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        return self.cpu_conv
        
    def get_cpu_ocr(self):
        if self.cpu_ocr_conv is None:
            opts = build_pipeline_options(AcceleratorDevice.CPU)
            opts.ocr_options.force_full_page_ocr = True
            self.cpu_ocr_conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        return self.cpu_ocr_conv

    def get_gpu_ocr(self):
        if self.gpu_ocr_conv is None:
            opts = build_pipeline_options(AcceleratorDevice.CUDA)
            opts.ocr_options.force_full_page_ocr = True
            self.gpu_ocr_conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        return self.gpu_ocr_conv

_CONVERTERS = ConverterCache()


def _page_below_resolution_limit(page_path: Path, max_dim: int = MAX_PAGE_WIDTH_PT) -> bool:
    """Check if a PDF page is small enough for GPU processing to avoid OOM."""
    try:
        doc = pdfium.PdfDocument(str(page_path))
        if len(doc) == 0:
            return False
        w, h = doc.get_page_size(0)
        return max(w, h) <= max_dim
    except Exception:
        return True  # safe default — let GPU try


def _convert_page(page_path: Path, page_no: int, callback: Optional[Callable] = None) -> DoclingDocument:
    if HAS_CUDA and _page_below_resolution_limit(page_path):
        try:
            converter = _CONVERTERS.get_gpu()
            if callback: callback(page_no, "GPU", None)
            return converter.convert(page_path).document
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "alloc" in str(e).lower():
                gc.collect()
                torch.cuda.empty_cache()
                if callback: callback(page_no, "CPU", "OOM")
                converter = _CONVERTERS.get_cpu()
                return converter.convert(page_path).document
            else:
                raise
        except ConversionError as e:
            if "std::bad_alloc" in str(e) or "out of memory" in str(e).lower():
                gc.collect()
                if HAS_CUDA:
                    torch.cuda.empty_cache()
                if callback: callback(page_no, "CPU", "OOM")
                converter = _CONVERTERS.get_cpu()
                return converter.convert(page_path).document
            raise
    else:
        converter = _CONVERTERS.get_cpu()
        if callback: callback(page_no, "CPU", None)
        return converter.convert(page_path).document

def _convert_page_force_ocr(page_path: Path, page_no: int, callback: Optional[Callable] = None) -> DoclingDocument:
    if HAS_CUDA and _page_below_resolution_limit(page_path):
        try:
            converter = _CONVERTERS.get_gpu_ocr()
            if callback: callback(page_no, "GPU", None)
            return converter.convert(page_path).document
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "alloc" in str(e).lower():
                logger.warning("event=gpu_ocr_oom page=%d", page_no)
                gc.collect()
                torch.cuda.empty_cache()
            else:
                raise
        except ConversionError as e:
            if "std::bad_alloc" in str(e) or "out of memory" in str(e).lower():
                logger.warning("event=gpu_ocr_oom page=%d", page_no)
                gc.collect()
                if HAS_CUDA:
                    torch.cuda.empty_cache()
                if callback: callback(page_no, "CPU", "OOM")
                converter = _CONVERTERS.get_cpu_ocr()
                return converter.convert(page_path).document
            raise
    converter = _CONVERTERS.get_cpu_ocr()
    if callback: callback(page_no, "CPU", None)
    return converter.convert(page_path).document

def _should_trigger_ocr_fallback(page_text: str) -> bool:
    if len(page_text) < OCR_MIN_TEXT_LEN and not re.search(r"[A-Za-z0-9]", page_text):
        return True
    clean = re.sub(r"[^a-zA-Z0-9\s#\-\|\*\_\`]", "", page_text)
    garbage_ratio = 1.0 - (len(clean) / max(len(page_text), 1))
    return garbage_ratio > OCR_GARBAGE_RATIO_THRESHOLD

try:
    _enc = tiktoken.get_encoding("cl100k_base")
    def _token_length(text: str) -> int:
        return len(_enc.encode(text))
    GLOBAL_SPLITTER = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=_token_length,
    )
except TypeError:
    GLOBAL_SPLITTER = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
    )

def extract_chunks(doc: _MergedDocument, filename: str, category: str) -> List[Document]:

    chunks: list[Document] = []
    prose_buffer: list[str] = []
    buffer_page_no: int = 1
    current_heading: str = "Introduction"
    last_known_page: int = 1
    last_element_was_table: bool = False
    last_table_page: int = -1
    last_table_heading: str = ""

    def flush_prose_buffer():
        nonlocal prose_buffer, buffer_page_no
        if not prose_buffer:
            return
        full_prose = f"## {current_heading}\n\n" + "\n".join(prose_buffer)
        prose_chunks = GLOBAL_SPLITTER.split_text(full_prose)
        for prose_chunk in prose_chunks:
            metadata = {
                "source": filename,
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
                    "source": filename,
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
                last_element_was_table = False
                continue

            is_continuation = (
                last_element_was_table
                and current_heading == last_table_heading
                and (page_no == last_table_page or page_no == last_table_page + 1)
            )

            metadata = {
                "source": filename,
                "page_number": page_no,
                "category": category,
                "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
                "is_table": True,
                "is_table_continuation": is_continuation,
            }
            contextual_table = f"### {current_heading}\n\n{table_markdown}"
            chunks.append(Document(page_content=contextual_table, metadata=metadata))
            last_element_was_table = True
            last_table_page = page_no
            last_table_heading = current_heading

        elif isinstance(element, TextItem):
            page_text = element.text.strip() if element.text else ""
            if not page_text:
                continue

            if page_no != buffer_page_no:
                flush_prose_buffer()

            prose_buffer.append(page_text)
            last_element_was_table = False

    flush_prose_buffer()

    for idx, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"{Path(filename).stem}_p{c.metadata.get('page_number', 1)}_{idx}"

    return chunks

def _count_jsonl_lines(path: Path) -> int:
    """Count lines in a JSONL file efficiently."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def build_and_save_index(deduped_jsonl: Path, category: str, ui_callback: Optional[Callable] = None) -> Path:
    """Build FAISS index by streaming chunks from a deduped JSONL file.
    Never loads more than one batch into RAM."""
    batch_size = FAISS_BATCH_SIZE

    if HAS_CUDA:
        free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        free_vram_gb = free_vram / (1024**3)
        if free_vram_gb < LOW_VRAM_THRESHOLD_GB:
            batch_size = max(FAISS_BATCH_SIZE // 2, 50)
            logger.warning("Low VRAM (%.1f GB), reduced batch to %d", free_vram_gb, batch_size)
        torch.cuda.empty_cache()

    total = _count_jsonl_lines(deduped_jsonl)
    if total == 0:
        raise ValueError("No chunks to index — deduped corpus is empty")
    if total > 50000:
        logger.warning("Large corpus detected (%d chunks), this may take significant time/VRAM", total)

    vector_store = None
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"} if HAS_CUDA else {},
        encode_kwargs={"normalize_embeddings": True},
    )

    batch: list[Document] = []
    processed = 0
    for chunk_dict in _iter_jsonl(deduped_jsonl):
        batch.append(Document(page_content=chunk_dict["page_content"], metadata=chunk_dict["metadata"]))
        if len(batch) >= batch_size:
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)
            processed += len(batch)
            if ui_callback:
                ui_callback("INDEX_PROGRESS", None, processed, total)
            batch = []

    if batch:
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)

    index_dir = _BUILD_DIR / f"{category}_index"
    os.makedirs(str(index_dir), exist_ok=True)
    vector_store.save_local(str(index_dir))

    (index_dir / "embedding_model.txt").write_text(EMBEDDING_MODEL, encoding="utf-8")

    faiss_hash = hashlib.sha256()
    faiss_hash.update((index_dir / "index.faiss").read_bytes())
    faiss_hash.update((index_dir / "index.pkl").read_bytes())
    (index_dir / "index.hash").write_text(faiss_hash.hexdigest())

    if HAS_CUDA:
        torch.cuda.empty_cache()

    return index_dir

def _build_bm25_corpus(input_jsonl: Path, output_path: Path):
    """Stream-read deduped JSONL, write BM25 corpus JSON array.
    Never loads the full corpus into RAM."""
    with open(output_path, "w", encoding="utf-8") as out:
        out.write("[")
        first = True
        for chunk_dict in _iter_jsonl(input_jsonl):
            obj = json.dumps({
                "text": chunk_dict["page_content"],
                "metadata": chunk_dict["metadata"],
            }, ensure_ascii=False)
            if first:
                out.write(obj)
                first = False
            else:
                out.write("," + obj)
        out.write("]")


def ingest_pdfs(pdf_paths: list[str], category: str, ui_callback: Optional[Callable] = None) -> dict:
    session_dir = _TMP_DIR / f"session_{category}"
    chunks_dir = session_dir / "chunks"
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir, ignore_errors=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    state = read_ingest_state(category)
    completed_files = set(state.get("completed_files", []))

    total_pages_overall = 0
    file_page_counts = {}
    for p_path_str in pdf_paths:
        p_path = Path(p_path_str)
        try:
            r = pypdf.PdfReader(p_path)
            cnt = len(r.pages)
            total_pages_overall += cnt
            file_page_counts[p_path.name] = cnt
        except (pypdf.errors.PdfReadError, OSError) as e:
            logger.warning(f"Skipping unreadable file {p_path.name}: {e}")
            continue

    if ui_callback:
        ui_callback("INIT", None, 0, total_pages_overall, dev=file_page_counts)

    jsonl_paths: list[Path] = []

    for idx, pdf_path_str in enumerate(pdf_paths):
        pdf_path = Path(pdf_path_str)
        filename = pdf_path.name

        try:
            r_check = pypdf.PdfReader(pdf_path)
            _ = len(r_check.pages)
        except pypdf.errors.PdfReadError:
            logger.error(f"Corrupt PDF, skipping: {filename}")
            if ui_callback:
                ui_callback("FILE_SKIP", filename, idx + 1, len(pdf_paths))
            continue

        file_hash = _get_file_hash(pdf_path)
        jsonl_path = chunks_dir / f"{file_hash}.jsonl"

        cached = _load_chunks_cache(file_hash)
        if cached is not None:
            for c in cached:
                if c.metadata.get("category") != category:
                    c.metadata = {**c.metadata, "category": category}
            _write_chunks_jsonl(jsonl_path, cached)
            jsonl_paths.append(jsonl_path)
            completed_files.add(filename)
            _write_ingest_state(category, {"completed_files": list(completed_files), "total_files": len(pdf_paths)})
            if ui_callback:
                ui_callback("FILE_CACHED", filename, idx + 1, len(pdf_paths))
            continue
        elif filename in completed_files:
            completed_files.remove(filename)

        if ui_callback:
            ui_callback("FILE_START", filename, idx + 1, len(pdf_paths))

        if ui_callback:
            ui_callback("FILE_SPLITTING", filename, idx + 1, len(pdf_paths))

        tmp_dir = Path(tempfile.mkdtemp(prefix="docling_pdf_"))
        restored_pages = 0

        try:
            for page_no, page_tmp_path, total_pages in _split_pdf_pages(pdf_path, tmp_dir):

                def page_cb(p_no, dev, warn):
                    if ui_callback:
                        ui_callback("PAGE_PROG", filename, p_no, total_pages, dev, warn)

                cached_chunks = _load_page_chunks(file_hash, page_no)
                if cached_chunks is not None:
                    _write_chunks_jsonl(jsonl_path, cached_chunks)
                    restored_pages += 1
                    if ui_callback:
                        ui_callback("PAGE_PROG", filename, page_no, total_pages, "checkpoint", "Restored")
                    continue

                try:
                    doc_page = _convert_page(page_tmp_path, page_no, callback=page_cb)
                    page_text = doc_page.export_to_text()

                    if _should_trigger_ocr_fallback(page_text):
                        if ui_callback:
                            ui_callback("PAGE_OCR_RETRY", filename, page_no, total_pages, "CPU", "OCR Triggered")
                        doc_page = _convert_page_force_ocr(page_tmp_path, page_no, callback=page_cb)

                    page_merged = _MergedDocument()
                    page_merged.add_page_doc(doc_page, page_no)
                    page_chunks = extract_chunks(page_merged, filename, category)
                    _save_page_chunks(file_hash, page_no, page_chunks)
                    _write_chunks_jsonl(jsonl_path, page_chunks)
                except (RuntimeError, ConversionError) as e:
                    if "std::bad_alloc" in str(e) or "out of memory" in str(e).lower():
                        logger.error("OOM on page %d of %s — skipping", page_no, filename)
                        gc.collect()
                        if HAS_CUDA:
                            torch.cuda.empty_cache()
                        continue
                    raise
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if restored_pages:
            logger.info("Restored %d pages from checkpoint for %s", restored_pages, filename)

        jsonl_paths.append(jsonl_path)

        if ui_callback:
            ui_callback("FILE_CHUNKING", filename, idx + 1, len(pdf_paths))

        _save_chunks_cache(file_hash, jsonl_path)

        completed_files.add(filename)
        _write_ingest_state(category, {"completed_files": list(completed_files), "total_files": len(pdf_paths)})

    if not jsonl_paths:
        raise ValueError("No chunks extracted from PDFs")

    if ui_callback:
        ui_callback("DEDUP", None, 0, 0)

    deduped_path = session_dir / "_deduped.jsonl"
    deduped_count = _stream_deduplicate(jsonl_paths, deduped_path)
    if deduped_count:
        logger.info("Deduplicated %d duplicate chunks across all files", deduped_count)

    if ui_callback:
        ui_callback("BUILDING_INDEX", None, 0, 0)

    index_dir = build_and_save_index(deduped_path, category, ui_callback=ui_callback)

    corpus_path = index_dir / "bm25_corpus.json"
    _build_bm25_corpus(deduped_path, corpus_path)

    chunk_count = _count_jsonl_lines(deduped_path)
    shutil.rmtree(session_dir, ignore_errors=True)
    clear_cache(category, delete_chunks=True)
    gc.collect()

    return {"chunk_count": chunk_count, "category": category}

