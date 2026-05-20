#!/usr/bin/env python3
"""
build_index.py
Command line script to build a vector index from railway PDFs.

Usage:
  # Process individual files
  python build_index.py --category Safety_Circulars file1.pdf file2.pdf

  # Process all PDFs in a directory (non-recursive)
  python build_index.py --category Safety_Circulars --directory /safety_circulars

  # Process all PDFs in directory and subdirectories
  python build_index.py --category Safety_Circulars --directory /safety_circulars --recursive

  # Mix of explicit files and directory
  python build_index.py --category Safety_Circulars file1.pdf --directory /safety_circulars
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

from langchain_core.documents import Document

from ingest import process_pdf
from local_builder import (
    CATEGORY_MAX_LEN,
    MAX_FILE_SIZE_BYTES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build a vector index from railway PDFs.")
    parser.add_argument(
        "--category",
        required=True,
        help="Category name for the document set (e.g., 'Class_390_Maintenance').",
    )
    parser.add_argument(
        "--directory",
        type=str,
        help="Directory containing PDF files to process (non-recursive).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process PDFs in directory and all subdirectories (requires --directory).",
    )
    parser.add_argument(
        "pdf_files",
        nargs="*",
        help="Individual PDF file paths to process.",
    )
    args = parser.parse_args()

    # Validate category
    if not args.category or not args.category.strip():
        logger.error("Category name is required.")
        sys.exit(1)
    safe_category = re.sub(r"[^A-Za-z0-9_]", "_", args.category.strip())
    if safe_category != args.category.strip():
        logger.info(
            f"Category name sanitized to: `{safe_category}`. "
            "This will be the directory name."
        )
    if not safe_category or len(safe_category) > CATEGORY_MAX_LEN:
        logger.error(
            f"Sanitized category name is invalid or too long "
            f"(max {CATEGORY_MAX_LEN} chars)."
        )
        sys.exit(1)

    # Collect PDF files from various sources
    pdf_paths = []
    
    # Add explicitly provided PDF files
    for f in args.pdf_files:
        pdf_path = Path(f)
        if not pdf_path.is_file():
            logger.error(f"File not found: {pdf_path}")
            sys.exit(1)
        if pdf_path.suffix.lower() != ".pdf":
            logger.error(f"File is not a PDF: {pdf_path}")
            sys.exit(1)
        # Check file size
        try:
            size = pdf_path.stat().st_size
            if size > MAX_FILE_SIZE_BYTES:
                logger.error(
                    f"{pdf_path.name} exceeds {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB."
                )
                sys.exit(1)
        except OSError as e:
            logger.error(f"Unable to read file {pdf_path}: {e}")
            sys.exit(1)
        pdf_paths.append(pdf_path)

    # Add PDFs from directory if specified
    if args.directory:
        directory_path = Path(args.directory)
        if not directory_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            sys.exit(1)
        
        # Determine search pattern based on recursive flag
        if args.recursive:
            pattern = "**/*.pdf"
            pdf_files_found = directory_path.rglob("*.pdf")
        else:
            pattern = "*.pdf"
            pdf_files_found = directory_path.glob("*.pdf")
        
        for pdf_path in pdf_files_found:
            # Skip if not a file (could be directory matching pattern)
            if not pdf_path.is_file():
                continue
                
            # Avoid duplicates if file was also in pdf_files
            if pdf_path in pdf_paths:
                continue
                
            # Validate file size
            try:
                size = pdf_path.stat().st_size
                if size > MAX_FILE_SIZE_BYTES:
                    logger.error(
                        f"{pdf_path.name} exceeds {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB."
                    )
                    sys.exit(1)
            except OSError as e:
                logger.error(f"Unable to read file {pdf_path}: {e}")
                sys.exit(1)
            pdf_paths.append(pdf_path)
    
    # If no PDF files specified at all, show error
    if not pdf_paths:
        logger.error("No PDF files specified. Provide files as arguments or use --directory.")
        sys.exit(1)
    
    logger.info(f"Found {len(pdf_paths)} PDF files to process")

    # No longer need Google API key for local embeddings
    # Keeping this section for compatibility but not requiring the key
    google_api_key = os.environ.get('GOOGLE_API_KEY', None)

    # Processing loop
    all_chunks: list[Document] = []
    per_file_stats: list[tuple[str, int, float]] = []
    t_session_start = time.monotonic()
    total_bytes = sum(p.stat().st_size for p in pdf_paths if p.stat().st_size)
    logger.info(
        "event=session_start file_count=%d total_bytes=%d category=%s",
        len(pdf_paths),
        total_bytes,
        safe_category,
    )

    for i, pdf_path in enumerate(pdf_paths):
        filename = pdf_path.name
        logger.info(f"Processing ({i+1}/{len(pdf_paths)}): {filename}")

        file_chunks, elapsed, success = process_pdf(
            pdf_path, filename, safe_category
        )

        if success:
            all_chunks.extend(file_chunks)
            per_file_stats.append((filename, len(file_chunks), elapsed))
            logger.info(
                "event=pdf_complete source=%s chunks=%d elapsed_s=%.2f",
                filename,
                len(file_chunks),
                elapsed,
            )

    print(
        f"Processed {len(pdf_paths)} files, "
        f"extracted {len(all_chunks)} total chunks. Now building index..."
    )

    # --- Guard & Profiling ---
    if not all_chunks:
        logger.error("event=no_chunks source=all files")
        print(
            "No text chunks were extracted. "
            "Please check if the uploaded PDFs contain readable text."
        )
        sys.exit(1)

    print("\nIngestion Profile")
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
        # Print as a simple table
        print(f"{'File':<30} {'Chunks':<8} {'Elapsed (s)':<12} {'Chunks/s'}")
        print("-" * 60)
        for row in profile_data:
            print(f"{row['File']:<30} {row['Chunks']:<8} {row['Elapsed (s)']:<12} {row['Chunks/s']}")

    total_elapsed = time.monotonic() - t_session_start
    print(f"\nTotal Chunks: {len(all_chunks)}")
    print(f"Total Time (s): {total_elapsed:.2f}")

    # --- Index Compilation ---
    print("\nEmbedding documents and building FAISS index...")
    from ingest import build_and_save_index
    try:
        index_dir = build_and_save_index(all_chunks, safe_category)
        logger.info(
            "event=session_complete total_chunks=%d total_elapsed_s=%.2f files=%d",
            len(all_chunks),
            total_elapsed,
            len(pdf_paths),
        )
        print(f"[SUCCESS] Index built and saved to `{index_dir}`")
    except RuntimeError as exc:
        logger.error("event=index_build_failed error=%s", str(exc))
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

