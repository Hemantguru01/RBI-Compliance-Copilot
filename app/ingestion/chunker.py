"""Structure-aware, page-bounded chunking.

Strategy (see README "Chunking strategy"):

1. **Page-bounded** – a chunk never crosses a page boundary, so every chunk
   maps to exactly one verifiable page number.
2. **Paragraph / clause packing** – cleaned pages arrive as paragraphs whose
   boundaries follow the regulation's own numbering ("3.1", "(a)", "(iv)").
   Whole paragraphs are packed greedily up to ``chunk_size`` characters, so
   related clauses stay together.
3. **Sentence fallback** – only a paragraph longer than ``chunk_size`` is
   split, at sentence boundaries (abbreviation-aware: "No.", "Rs.", "i.e.").
   A single over-long sentence is split on word boundaries as a last resort.
4. **Sentence-level overlap** – each new chunk starts with the trailing
   sentences of the previous chunk (up to ``chunk_overlap`` characters),
   never a half sentence.
5. **Section context** – the most recent heading ("Chapter III – ...",
   "Section 5 ...") is carried across pages and stored on the chunk; it is
   prepended to the text that gets embedded.
"""

from __future__ import annotations

import re

from app.schemas.models import Chunk, DocumentRecord, PageText

_ABBREVIATIONS = (
    "No", "Nos", "Rs", "Re", "viz", "i.e", "e.g", "etc", "Sr", "Mr", "Ms", "Dr", "Ltd",
    "Co", "Govt", "Dept", "Para", "para", "Sec", "Cl", "Art", "Vol", "approx", "w.e.f", "Ch",
)
_ABBREV_RE = re.compile(r"(?:\b(?:" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")|(?<![A-Za-z])[A-Z])\.$")
# A "sentence" that is only a clause number such as "1." or "(iv)." belongs to the next sentence.
_ENUMERATION_ONLY_RE = re.compile(r"\(?(?:\d{1,3}(?:\.\d{1,3})*|[ivxlc]{1,6}|[a-z])\)?\.?")
_SENTENCE_END_RE = re.compile(r"(?<=[.;:?!])\s+(?=[\"'(\[]?[A-Z0-9(])")

HEADING_RE = re.compile(
    r"^(?:(?:CHAPTER|Chapter|PART|Part|SECTION|Section|ANNEX|Annex|Appendix|APPENDIX|Schedule|SCHEDULE)"
    r"\s*[-–—:]?\s*[A-Z0-9IVXLC]{0,8}\b.{0,140}"
    r"|[A-Z][A-Z0-9 ,&'()/\-–]{6,120})$"
)


def is_heading(paragraph: str) -> bool:
    p = paragraph.strip()
    if len(p) > 150 or p.endswith((".", ";", ",")):
        return False
    return bool(HEADING_RE.match(p))


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation, re-joining splits that follow abbreviations."""
    pieces = _SENTENCE_END_RE.split(text.strip())
    sentences: list[str] = []
    for piece in pieces:
        if sentences and (_ABBREV_RE.search(sentences[-1]) or _ENUMERATION_ONLY_RE.fullmatch(sentences[-1].strip())):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return [s for s in sentences if s.strip()]


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    words, parts, current = sentence.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


def _units(paragraph: str, chunk_size: int) -> list[str]:
    """Break a paragraph into units no longer than ``chunk_size``."""
    if len(paragraph) <= chunk_size:
        return [paragraph]
    units: list[str] = []
    for sentence in split_sentences(paragraph):
        if len(sentence) <= chunk_size:
            units.append(sentence)
        else:
            units.extend(_split_long_sentence(sentence, chunk_size))
    return units


def _overlap_tail(text: str, overlap: int) -> str:
    """Trailing whole sentences of ``text`` whose total length is <= ``overlap``."""
    if overlap <= 0:
        return ""
    tail: list[str] = []
    total = 0
    for sentence in reversed(split_sentences(text)):
        if total + len(sentence) + 1 > overlap:
            break
        tail.insert(0, sentence)
        total += len(sentence) + 1
    return " ".join(tail)


def chunk_page(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, list[str]]]:
    """Chunk one page. Returns ``(chunk_text, headings_seen_before_chunk_start)`` pairs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[tuple[str, list[str]]] = []
    current_parts: list[str] = []
    current_len = 0
    headings_pending: list[str] = []
    headings_for_current: list[str] = []

    def flush() -> str:
        nonlocal current_parts, current_len, headings_for_current
        body = "\n".join(current_parts).strip()
        if body:
            chunks.append((body, headings_for_current))
        current_parts, current_len, headings_for_current = [], 0, []
        return body

    for paragraph in paragraphs:
        if is_heading(paragraph):
            headings_pending.append(paragraph)
        for unit in _units(paragraph, chunk_size):
            if current_parts and current_len + len(unit) + 1 > chunk_size:
                previous = flush()
                tail = _overlap_tail(previous, chunk_overlap)
                if tail and len(tail) + len(unit) + 1 <= chunk_size:
                    current_parts, current_len = [tail], len(tail)
            if not current_parts or headings_pending:
                headings_for_current = headings_for_current + headings_pending
                headings_pending = []
            current_parts.append(unit)
            current_len += len(unit) + 1
    flush()
    return chunks


def chunk_document(
    record: DocumentRecord,
    pages: list[PageText],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    min_chunk_chars: int = 80,
) -> list[Chunk]:
    """Chunk a whole document, preserving page numbers and metadata on every chunk."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    out: list[Chunk] = []
    section: str | None = None
    index = 0
    for page in pages:
        page_chunks = chunk_page(page.text, chunk_size, chunk_overlap)
        # Merge a too-small trailing chunk into its predecessor on the same page.
        if len(page_chunks) >= 2 and len(page_chunks[-1][0]) < min_chunk_chars:
            last_text, last_heads = page_chunks.pop()
            prev_text, prev_heads = page_chunks[-1]
            page_chunks[-1] = (f"{prev_text}\n{last_text}", prev_heads + last_heads)
        for text, headings in page_chunks:
            if headings:
                section = headings[-1][:160]
            if len(text) < min_chunk_chars and len(page_chunks) == 1 and len(text.split()) < 5:
                continue  # stray fragment such as a lone heading or signature line
            out.append(
                Chunk(
                    chunk_id=f"{record.doc_id}:p{page.page_number}:c{index}",
                    doc_id=record.doc_id,
                    title=record.title,
                    source_filename=record.source_filename,
                    source_url=record.source_url,
                    page_number=page.page_number,
                    circular_number=record.circular_number,
                    publication_date=record.publication_date,
                    section=section,
                    chunk_index=index,
                    text=text,
                )
            )
            index += 1
    return out
