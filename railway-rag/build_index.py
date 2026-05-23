#!/usr/bin/env python3
"""
build_index.py
Command line script to build a vector index from railway PDFs.

Usage:
  python build_index.py --category Safety_Circulars file1.pdf file2.pdf
  python build_index.py --category Safety_Circulars --directory /safety_circulars
  python build_index.py --category Safety_Circulars --directory /safety_circulars --recursive
"""

import argparse
import logging
import re
import sys
from pathlib import Path

from ingest import ingest_pdfs
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
        help="Directory containing PDF files to process.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process PDFs recursively (requires --directory).",
    )
    parser.add_argument(
        "pdf_files",
        nargs="*",
        help="Individual PDF file paths.",
    )
    args = parser.parse_args()

    if not args.category or not args.category.strip():
        logger.error("Category name is required.")
        sys.exit(1)
    safe_category = re.sub(r"[^A-Za-z0-9_]", "_", args.category.strip())
    if not safe_category or len(safe_category) > CATEGORY_MAX_LEN:
        logger.error(f"Sanitized category name invalid or too long (max {CATEGORY_MAX_LEN}).")
        sys.exit(1)

    pdf_paths = []

    for f in args.pdf_files:
        pdf_path = Path(f)
        if not pdf_path.is_file():
            logger.error(f"File not found: {pdf_path}")
            sys.exit(1)
        if pdf_path.suffix.lower() != ".pdf":
            logger.error(f"Not a PDF: {pdf_path}")
            sys.exit(1)
        try:
            if pdf_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                logger.error(f"{pdf_path.name} exceeds {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB.")
                sys.exit(1)
        except OSError as e:
            logger.error(f"Cannot read {pdf_path}: {e}")
            sys.exit(1)
        pdf_paths.append(str(pdf_path.resolve()))

    if args.directory:
        dir_path = Path(args.directory)
        if not dir_path.is_dir():
            logger.error(f"Directory not found: {dir_path}")
            sys.exit(1)
        pdf_files = dir_path.rglob("*.pdf") if args.recursive else dir_path.glob("*.pdf")
        for pdf_path in pdf_files:
            if not pdf_path.is_file():
                continue
            resolved = str(pdf_path.resolve())
            if resolved not in pdf_paths:
                try:
                    if pdf_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                        logger.error(f"{pdf_path.name} exceeds {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB.")
                        sys.exit(1)
                except OSError as e:
                    logger.error(f"Cannot read {pdf_path}: {e}")
                    sys.exit(1)
                pdf_paths.append(resolved)

    if not pdf_paths:
        logger.error("No PDF files specified.")
        sys.exit(1)

    logger.info(f"Found {len(pdf_paths)} PDF files to process")

    # --- Single ingest call ---
    print("Processing:")
    for p in pdf_paths:
        print(f"  {p.name}")
    try:
        pdf_path_strs = [str(p.resolve()) for p in pdf_paths]
        result = ingest_pdfs(pdf_path_strs, safe_category)
        print(f"[SUCCESS] Index built — {result['chunk_count']} chunks in `{safe_category}_index`")
        logger.info(
            "event=session_complete chunk_count=%d category=%s",
            result["chunk_count"],
            safe_category,
        )
    except (ValueError, RuntimeError) as exc:
        logger.error("event=session_failed error=%s", str(exc))
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

