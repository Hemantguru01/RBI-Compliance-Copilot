"""Heuristic extraction of RBI-specific metadata (circular number, date, title).

These patterns only *find* values printed in the document; nothing is ever
synthesised. If a value cannot be found it stays ``None``.
"""

from __future__ import annotations

import re
from datetime import datetime

# e.g. "RBI/2025-26/123"
RBI_REF_RE = re.compile(r"\bRBI/\d{4}-\d{2}/\d{1,4}\b")
# e.g. "DOR.CRE.REC.No.45/21.04.048/2025-26", "DoR.MCS.REC.No.39/01.01.001/2025-26",
#      "CO.DPSS.POLC.No.S-123/02.14.003/2025-26"
DEPT_REF_RE = re.compile(
    r"\b[A-Z][A-Za-z]{1,7}(?:\.[A-Za-z0-9&()\-]{1,12}){0,6}\.No\.\s?[A-Z0-9\-]{1,10}"
    r"(?:/[0-9.]+)?/\d{4}-\d{2}\b"
)

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
DATE_MDY_RE = re.compile(rf"\b({_MONTHS})\s+(\d{{1,2}}),\s*(\d{{4}})\b")
DATE_DMY_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTHS}),?\s+(\d{{4}})\b")

TITLE_HINT_RE = re.compile(
    r"(Reserve Bank of India\s*[\(\[].*?[\)\]]\s*(?:Supervisory\s+)?Directions?,?\s*\d{4}"
    r"|Master Directions?\s*[-–—:]\s*[^\n]{5,160})",
    re.IGNORECASE | re.DOTALL,
)


def find_circular_number(text: str) -> str | None:
    """Return the RBI reference and/or department reference printed in ``text``."""
    parts: list[str] = []
    rbi = RBI_REF_RE.search(text)
    if rbi:
        parts.append(rbi.group(0))
    dept = DEPT_REF_RE.search(text)
    if dept:
        parts.append(" ".join(dept.group(0).split()))
    return "; ".join(parts) if parts else None


def _iso(day: str, month: str, year: str) -> str | None:
    try:
        return datetime.strptime(f"{int(day)} {month} {year}", "%d %B %Y").date().isoformat()
    except ValueError:
        return None


def find_publication_date(text: str) -> str | None:
    """Return the first full date printed in ``text`` as ISO ``YYYY-MM-DD``."""
    candidates: list[tuple[int, str]] = []
    m = DATE_MDY_RE.search(text)
    if m:
        iso = _iso(m.group(2), m.group(1), m.group(3))
        if iso:
            candidates.append((m.start(), iso))
    m = DATE_DMY_RE.search(text)
    if m:
        iso = _iso(m.group(1), m.group(2), m.group(3))
        if iso:
            candidates.append((m.start(), iso))
    if not candidates:
        return None
    return min(candidates)[1]


def find_title(first_page_text: str) -> str | None:
    """Find a Directions/Master Direction title on the first page, if printed."""
    m = TITLE_HINT_RE.search(first_page_text)
    if not m:
        return None
    title = " ".join(m.group(0).split())
    return title[:200] if len(title) >= 10 else None


def title_from_filename(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    stem = re.sub(r"^\d+_", "", stem)
    return " ".join(re.sub(r"[_\-]+", " ", stem).split()) or filename
