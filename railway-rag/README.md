# Railway RAG Pipeline

A production-ready Retrieval-Augmented Generation system for dense technical railway PDFs, designed for low-resource environments.

## Overview

This system processes railway PDFs using Docling for parsing and OCR, creates hybrid FAISS + BM25 indices, and provides retrieval-augmented generation via Streamlit. It operates under strict constraints:
- **Local builder**: Runs on NVIDIA GTX 1650 (4 GB VRAM)
- **Cloud app**: Runs on Streamlit Cloud (~1 GB RAM)
- **Handles**: Dense technical railway PDFs with complex tables and diagrams

## Architecture

```
┌─────────────────────────────────────┐
│        LOCAL MACHINE (GPU)          │
│  local_builder.py                   │
│  ┌──────────┐  ┌──────────────────┐ │
│  │ Docling  │→ │ Chunker + Meta   │ │
│  │ OCR/GPU  │  │ (tiktoken split) │ │
│  └──────────┘  └────────┬─────────┘ │
│                         ↓           │
│              FAISS index + BM25     │
│              bm25_corpus.json       │
└─────────────────┬───────────────────┘
                  │ git push / copy
                  ↓
┌─────────────────────────────────────┐
│       STREAMLIT CLOUD (~1 GB RAM)   │
│  app.py                             │
│  ┌──────────┐  ┌──────────────────┐ │
│  │ FAISS +  │→ │ FlashRank Rerank │ │
│  │ BM25     │  │ Token Budget     │ │
│  └──────────┘  └────────┬─────────┘ │
│                         ↓           │
│         Groq LLM → Gemini Fallback  │
│         Programmatic Citation Map   │
└─────────────────────────────────────┘
```

## Project Directory Tree

```
railway-rag/
├── local_builder.py               ← Streamlit local ingestion UI
├── build_index.py                  ← CLI ingestion (no Streamlit)
├── app.py                         ← Streamlit Cloud retrieval UI
├── ingest.py                      ← Shared ingestion logic
├── utils.py                       ← Shared utility functions
├── local_requirements.txt         ← Minimal deps for local ingestion
├── requirements.txt               ← Full pinned deps (Cloud + local)
├── .gitignore
├── vector_indices/
│   └── [Category]_index/
│       ├── index.faiss
│       ├── index.pkl
│       └── bm25_corpus.json
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_ocr_failure.py
│   └── test_memory.py
├── smoke_test.py
└── README.md
```

## Prerequisites

- Python 3.11 (required; 3.12+ not tested)
- NVIDIA GPU with CUDA support (GTX 1650 or better; 4 GB VRAM minimum)
- CUDA 12.4 drivers installed (NVIDIA driver 596.49+)
- Groq API key (free tier works for retrieval)
- Google AI Studio API key (for embeddings and Gemini fallback)

## GPU Setup

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output: `True`

For CUDA 12.4 (recommended):
```
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

## Local Environment Setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install PyTorch FIRST (see GPU Setup for CUDA version)
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies (local ingestion only)
pip install -r local_requirements.txt

# For full Cloud deployment, also install:
pip install -r requirements.txt
```

## Local Ingestion

### Option A — Streamlit UI
1. Activate your venv.
2. Run: `streamlit run local_builder.py`
3. In the sidebar:
   - Enter a Category Name (e.g. "Signalling_Standards").
   - Upload one or more railway PDFs (max 200 MB each).
4. Click Process / Run.
5. Wait for "Index saved" success message.
6. Verify output: `dir vector_indices\[Category]_index\`
   Expected: `index.faiss  index.pkl  bm25_corpus.json`

### Option B — CLI (no Streamlit required)
```bash
set GOOGLE_API_KEY=your_key_here
python build_index.py --category Signalling_Standards path/to/document1.pdf path/to/document2.pdf
```

## Secrets Configuration

Create `.streamlit/secrets.toml` (NEVER commit this file):

```toml
GROQ_API_KEY = "gsk_your_groq_key_here"
GEMINI_API_KEY = "your_gemini_key_here"
```

On Streamlit Cloud, add these via the app's Secrets panel in the dashboard.

## Streamlit Cloud Deployment

1. Push the repository to GitHub (indices tracked via git-lfs).
2. Go to share.streamlit.io → New app.
3. Select the repository and set Main file path: `app.py`
4. Add secrets via the Secrets panel (see Secrets Configuration above).
5. Deploy. The app reads from `vector_indices/` in the repository.

## Smoke Test

After ingestion, validate the index locally:

```bash
python smoke_test.py \
    --index_dir ./vector_indices/Signalling_Standards_index \
    --query "maximum permitted speed" \
    --google_api_key YOUR_KEY
```

Expected output:
  Top 3 results with source filename, page number, and content preview.
  Exit code 0 on success, 1 on failure.

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

See `tests/` directory for individual test descriptions (Module 05).

## Troubleshooting

### OOM / CUDA errors:
  The app automatically falls back to CPU processing on OOM.
  If CPU fallback also fails, reduce the number of PDFs in a single session.
  Check logs for: `event=oom_fallback`

### OCR silent failure:
  Pages with garbage_ratio > 0.40 trigger automatic OCR re-processing.
  Check logs for: `event=ocr_trigger`

### BM25 corpus missing:
  If `bm25_corpus.json` is absent, the app degrades to FAISS-only retrieval.
  Re-run `local_builder.py` to regenerate the corpus.
  Check logs for: `event=bm25_missing`

### Embedding rate limits:
  The builder waits 60 seconds and retries once automatically.
  Check logs for: `event=embedding_rate_limit`

### Empty chunk count:
  If no chunks are extracted, verify PDFs are not password-protected and
  contain selectable text or valid OCR content.
  Check logs for: `event=no_chunks`

### Hallucinated import errors:
  Docling APIs may differ between minor versions.
  Check comments marked: `# WARNING: Verify this API against installed docling version.`
  Then verify: `pip show docling`

### Import errors:
  If you see `ModuleNotFoundError`, run: `pip install -r requirements.txt`
  Docling and langchain packages are updated frequently — pin versions if breaking changes occur.

## Security Notes

- NEVER commit `.streamlit/secrets.toml`.
- The FAISS index uses `allow_dangerous_deserialization=True` — only load
  indices you have built yourself from trusted sources.
- API keys entered in `local_builder.py` are never written to disk.