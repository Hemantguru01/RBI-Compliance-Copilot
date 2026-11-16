"""Text normalisation for PDF-extracted pages.

PDF text arrives with hard line breaks inside paragraphs, hyphenated words
split across lines, running headers/footers and bare page numbers. We clean
those while keeping paragraph and clause boundaries, which the chunker uses.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Lines that start a new logical block: numbered clauses, lettered/roman
# sub-clauses, bullets, chapter/section headings, annexes.
BLOCK_START_RE = re.compile(
    r"^\s*(?:"
    r"\d{1,3}(?:\.\d{1,3}){0,4}[.)]?\s"  # 1.  2.1  3.2.4  4)
    r"|\(\s*[a-zA-Z]{1,2}\s*\)"  # (a)  (ab)
    r"|\(\s*[ivxlcdm]{1,6}\s*\)"  # (iv)
    r"|[a-z]\)\s"  # a)
    r"|[•▪●◦\-–]\s"  # bullets
    r"|(?:CHAPTER|Chapter|SECTION|Section|PART|Part|ANNEX|Annex|Appendix|APPENDIX|Schedule|SCHEDULE)\b"
    r")"
)
PAGE_NUMBER_RE = re.compile(r"^\s*(?:[-–]?\s*\d{1,4}\s*[-–]?|Page\s+\d+(?:\s+of\s+\d+)?)\s*$", re.I)
_SPACES_RE = re.compile(r"[ \t  -​]+")


def normalise_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    replacements = {"‘": "'", "’": "'", "“": '"', "”": '"', "": "•", "\x00": ""}
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def find_repeated_lines(pages: list[str], min_pages: int = 4, ratio: float = 0.6) -> set[str]:
    """Lines that occur on most pages are running headers/footers."""
    if len(pages) < min_pages:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        lines = {ln.strip() for ln in page.splitlines() if ln.strip()}
        counts.update(lines)
    threshold = max(min_pages, int(len(pages) * ratio))
    return {line for line, c in counts.items() if c >= threshold and len(line) < 120}


def clean_page_text(text: str, repeated_lines: set[str] | None = None) -> str:
    """Clean one page and return paragraphs separated by blank lines."""
    repeated_lines = repeated_lines or set()
    text = normalise_unicode(text)
    raw_lines = [_SPACES_RE.sub(" ", ln).strip() for ln in text.splitlines()]

    paragraphs: list[str] = []
    current = ""
    for line in raw_lines:
        if not line:
            if current:
                paragraphs.append(current)
                current = ""
            continue
        if line in repeated_lines or PAGE_NUMBER_RE.match(line):
            continue
        if not current:
            current = line
        elif BLOCK_START_RE.match(line):
            paragraphs.append(current)
            current = line
        elif current.endswith("-") and not current.endswith(" -") and line[:1].islower():
            current = current[:-1] + line  # de-hyphenate "regu-\nlation"
        else:
            current = f"{current} {line}"
    if current:
        paragraphs.append(current)
    return "\n\n".join(p.strip() for p in paragraphs if p.strip())
