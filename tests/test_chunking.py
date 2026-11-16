import pytest

from app.ingestion.chunker import chunk_document, is_heading, split_sentences
from app.ingestion.pdf_loader import extract_pdf
from app.schemas.models import DocumentRecord, PageText


def _record(**kw) -> DocumentRecord:
    base = dict(doc_id="abc123", title="Test Directions", source_filename="t.pdf", sha256="abc123" * 10,
                page_count=2, char_count=0, circular_number="RBI/2099-00/1", publication_date="2099-01-01",
                source_url="https://example.org/t.pdf")
    base.update(kw)
    return DocumentRecord(**base)


LONG_PARA = " ".join(f"Sentence number {i} states that the bank shall comply with requirement {i}." for i in range(40))


def test_chunks_never_cross_pages_and_keep_page_numbers():
    pages = [PageText(page_number=1, text=LONG_PARA), PageText(page_number=2, text="2. Short clause on page two.")]
    chunks = chunk_document(_record(), pages, chunk_size=400, chunk_overlap=100, min_chunk_chars=10)
    assert {c.page_number for c in chunks} == {1, 2}
    assert all("page two" not in c.text for c in chunks if c.page_number == 1)
    assert [c for c in chunks if c.page_number == 2][0].text == "2. Short clause on page two."


def test_chunk_size_respected_and_content_preserved():
    pages = [PageText(page_number=1, text=LONG_PARA)]
    chunks = chunk_document(_record(), pages, chunk_size=400, chunk_overlap=100, min_chunk_chars=10)
    assert len(chunks) > 3
    assert all(len(c.text) <= 400 for c in chunks)
    joined = " ".join(c.text for c in chunks)
    for i in range(40):  # no sentence lost
        assert f"requirement {i}." in joined


def test_overlap_is_whole_sentences():
    pages = [PageText(page_number=1, text=LONG_PARA)]
    chunks = chunk_document(_record(), pages, chunk_size=400, chunk_overlap=150, min_chunk_chars=10)
    for prev, nxt in zip(chunks, chunks[1:]):
        first_sentence = split_sentences(nxt.text)[0]
        assert first_sentence in prev.text, "next chunk should start with a sentence repeated from the previous one"
        assert first_sentence.startswith("Sentence number")


def test_metadata_is_copied_to_every_chunk():
    rec = _record()
    chunks = chunk_document(rec, [PageText(page_number=1, text=LONG_PARA)], 400, 100, 10)
    for i, c in enumerate(chunks):
        assert (c.doc_id, c.title, c.circular_number, c.publication_date, c.source_url, c.source_filename) == (
            rec.doc_id, rec.title, rec.circular_number, rec.publication_date, rec.source_url, rec.source_filename)
        assert c.chunk_index == i
        assert c.chunk_id == f"abc123:p1:c{i}"


def test_paragraphs_are_kept_together_when_they_fit():
    text = "\n\n".join([
        "1. First clause about grievance redress policy approved by the Board.",
        "(a) the policy shall be displayed on the website;",
        "(b) the policy shall be reviewed annually.",
    ])
    chunks = chunk_document(_record(), [PageText(page_number=1, text=text)], 1000, 100, 10)
    assert len(chunks) == 1
    assert "(a) the policy" in chunks[0].text and "(b) the policy" in chunks[0].text


def test_section_heading_is_carried_across_pages():
    pages = [
        PageText(page_number=1, text="Chapter II - Know Your Customer\n\n3. Banks shall verify identity."),
        PageText(page_number=2, text="4. Banks shall update records periodically."),
    ]
    chunks = chunk_document(_record(), pages, 1000, 100, 10)
    assert chunks[-1].page_number == 2
    assert chunks[-1].section == "Chapter II - Know Your Customer"


def test_sentences_ending_in_numbers_and_clause_numbers():
    s = split_sentences("1. Resolve within 30 days. Report by 2025. (iv) Keep records.")
    assert s == ["1. Resolve within 30 days.", "Report by 2025.", "(iv) Keep records."]


def test_sentence_splitter_respects_abbreviations():
    s = split_sentences("Refer to circular No. 45 dated today. The limit is Rs. 50,000 per account. Done.")
    assert s == ["Refer to circular No. 45 dated today.", "The limit is Rs. 50,000 per account.", "Done."]


def test_heading_detection():
    assert is_heading("Chapter III - Reporting")
    assert is_heading("SECTION 5 GOVERNANCE")
    assert not is_heading("3. A bank shall resolve every complaint within 30 days.")


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_document(_record(), [PageText(page_number=1, text="x")], chunk_size=100, chunk_overlap=100)


def test_fixture_pdf_chunks_map_to_true_pages(fixture_pdf_dir):
    doc = extract_pdf(fixture_pdf_dir / "fixture_grievance_redress.pdf")
    chunks = chunk_document(doc.record, doc.pages, 400, 80, 40)
    timeline = [c for c in chunks if "within 30 days of receipt" in c.text]
    assert timeline and all(c.page_number == 2 for c in timeline)
    assert all(c.section and c.section.startswith("Chapter II") for c in timeline)
