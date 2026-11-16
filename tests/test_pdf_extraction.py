from pathlib import Path

import pymupdf
import pytest

from app.core.exceptions import PDFExtractionError
from app.ingestion.pdf_loader import extract_pdf
from app.ingestion.text_cleaning import clean_page_text, find_repeated_lines
from app.utils.fixtures import FIXTURE_DOCS, write_blank_pdf, write_corrupted_pdf


def test_extracts_every_page_with_correct_page_numbers(fixture_pdf_dir):
    doc = extract_pdf(fixture_pdf_dir / "fixture_grievance_redress.pdf")
    assert doc.record.page_count == 3
    assert [p.page_number for p in doc.pages] == [1, 2, 3]
    # Text that was written on page 2 must be extracted on page 2 (and only there).
    page_of = {p.page_number: p.text for p in doc.pages}
    assert "resolve every complaint within 30 days" in page_of[2]
    assert "30 days" not in page_of[1]
    assert "compensation of 100 rupees" in page_of[3]


def test_extracts_metadata_printed_in_document(fixture_pdf_dir):
    rec = extract_pdf(fixture_pdf_dir / "fixture_digital_lending.pdf").record
    spec = FIXTURE_DOCS["fixture_digital_lending.pdf"]
    assert rec.circular_number == f"{spec['ref']}; {spec['dept']}"
    assert rec.publication_date == "2099-03-02"
    assert rec.title.startswith("[SYNTHETIC TEST FIXTURE]")
    assert rec.source_filename == "fixture_digital_lending.pdf"
    assert len(rec.doc_id) == 16 and rec.sha256.startswith(rec.doc_id)


def test_manifest_entry_supplies_title_and_source_url(fixture_pdf_dir):
    entry = {"title": "Official Title", "source_url": "https://rbidocs.rbi.org.in/x.pdf"}
    rec = extract_pdf(fixture_pdf_dir / "fixture_cyber_security.pdf", manifest_entry=entry).record
    assert rec.title == "Official Title"
    assert rec.source_url == "https://rbidocs.rbi.org.in/x.pdf"


def test_corrupted_pdf_raises(tmp_path):
    with pytest.raises(PDFExtractionError, match="Cannot open"):
        extract_pdf(write_corrupted_pdf(tmp_path / "bad.pdf"))


def test_pdf_without_text_raises(tmp_path):
    with pytest.raises(PDFExtractionError, match="No extractable text"):
        extract_pdf(write_blank_pdf(tmp_path / "blank.pdf"))


def test_empty_and_missing_files_raise(tmp_path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(PDFExtractionError, match="Empty file"):
        extract_pdf(empty)
    with pytest.raises(PDFExtractionError, match="not found"):
        extract_pdf(tmp_path / "missing.pdf")


def test_blank_pages_are_reported_not_indexed(tmp_path):
    path = tmp_path / "mixed.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 80), "1. A bank shall maintain records for five years.")
    doc.new_page()  # blank
    doc.new_page().insert_text((50, 80), "2. Records shall be produced on demand.")
    doc.save(path)
    doc.close()
    extracted = extract_pdf(path)
    assert [p.page_number for p in extracted.pages] == [1, 3]
    assert extracted.record.empty_pages == [2]


def test_cleaning_joins_wrapped_lines_and_dehyphenates():
    raw = "1. A bank shall put in place a\nBoard-approved policy on regu-\nlation of outsourcing.\n2. Second clause."
    cleaned = clean_page_text(raw)
    paragraphs = cleaned.split("\n\n")
    assert paragraphs[0] == "1. A bank shall put in place a Board-approved policy on regulation of outsourcing."
    assert paragraphs[1] == "2. Second clause."


def test_cleaning_removes_page_numbers_and_running_headers():
    pages = [f"Reserve Bank of India\nClause text {i} here.\n{i}" for i in range(1, 6)]
    repeated = find_repeated_lines(pages)
    assert "Reserve Bank of India" in repeated
    cleaned = clean_page_text(pages[2], repeated)
    assert cleaned == "Clause text 3 here."
