"""PDF loading with PyMuPDF: page-level text extraction plus metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pymupdf

from app.core.exceptions import PDFExtractionError
from app.core.logging import get_logger
from app.ingestion.metadata import (
    find_circular_number,
    find_publication_date,
    find_title,
    title_from_filename,
)
from app.ingestion.text_cleaning import clean_page_text, find_repeated_lines
from app.schemas.models import DocumentRecord, ExtractedDocument, PageText

logger = get_logger(__name__)

MAX_PDF_BYTES = 100 * 1024 * 1024


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(raw_dir: Path) -> dict[str, dict]:
    """Optional ``manifest.json`` (written by the download script) keyed by filename."""
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {d["filename"]: d for d in data.get("documents", []) if "filename" in d}
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Ignoring unreadable manifest", extra={"path": str(manifest_path), "error": str(exc)})
        return {}


def extract_pdf(path: Path, manifest_entry: dict | None = None, sha256: str | None = None) -> ExtractedDocument:
    """Extract cleaned per-page text and metadata from one PDF.

    Raises:
        PDFExtractionError: if the file is missing, too large, corrupted,
            encrypted, or contains no extractable text (e.g. scanned images).
    """
    if not path.is_file():
        raise PDFExtractionError(f"File not found: {path.name}")
    size = path.stat().st_size
    if size == 0:
        raise PDFExtractionError(f"Empty file: {path.name}")
    if size > MAX_PDF_BYTES:
        raise PDFExtractionError(f"File too large ({size} bytes): {path.name}")

    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # PyMuPDF raises several error types for bad files
        raise PDFExtractionError(f"Cannot open PDF {path.name}: {exc}") from exc

    try:
        if doc.needs_pass:
            raise PDFExtractionError(f"PDF is password protected: {path.name}")
        if doc.page_count == 0:
            raise PDFExtractionError(f"PDF has no pages: {path.name}")
        raw_pages = []
        for page in doc:
            try:
                raw_pages.append(page.get_text("text", sort=True))
            except Exception as exc:  # a single broken page should not kill the document
                logger.warning("Page extraction failed", extra={"file": path.name, "page": page.number + 1, "error": str(exc)})
                raw_pages.append("")
        pdf_meta_title = (doc.metadata or {}).get("title") or ""
    finally:
        doc.close()

    repeated = find_repeated_lines(raw_pages)
    pages: list[PageText] = []
    empty_pages: list[int] = []
    for i, raw in enumerate(raw_pages, start=1):
        cleaned = clean_page_text(raw, repeated)
        if cleaned:
            pages.append(PageText(page_number=i, text=cleaned))
        else:
            empty_pages.append(i)

    if not pages:
        raise PDFExtractionError(
            f"No extractable text in {path.name} (scanned/image-only PDFs need OCR, which is not enabled)"
        )

    head_text = "\n".join(p.text for p in pages[:2])
    manifest_entry = manifest_entry or {}
    title = (
        manifest_entry.get("title")
        or find_title(pages[0].text)
        or (pdf_meta_title.strip() if len(pdf_meta_title.strip()) > 8 else None)
        or title_from_filename(path.name)
    )
    digest = sha256 or file_sha256(path)
    record = DocumentRecord(
        doc_id=digest[:16],
        title=title,
        source_filename=path.name,
        source_url=manifest_entry.get("source_url"),
        circular_number=find_circular_number(head_text),
        publication_date=find_publication_date(head_text),
        sha256=digest,
        page_count=len(raw_pages),
        char_count=sum(len(p.text) for p in pages),
        empty_pages=empty_pages,
    )
    logger.info(
        "Extracted PDF",
        extra={"file": path.name, "pages": record.page_count, "chars": record.char_count, "empty_pages": len(empty_pages)},
    )
    return ExtractedDocument(record=record, pages=pages)
