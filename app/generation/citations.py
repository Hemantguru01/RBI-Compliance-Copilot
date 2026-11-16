"""Citation parsing, validation and grounding metrics.

The LLM cites with ``[S#]`` markers that refer to the retrieved chunks in the
order they were given. Here we:

* keep only markers that map to a real retrieved chunk (invalid ones are
  stripped from the answer and reported),
* build ``Citation`` objects from the chunk metadata, never from LLM output, so
  titles, circular numbers and page numbers are always the real ones,
* pick the excerpt of each chunk that best supports the claims citing it,
* compute simple grounding metrics (citation coverage, lexical support).
"""

from __future__ import annotations

import re

from app.ingestion.chunker import split_sentences
from app.retrieval.keyword_search import tokenize
from app.schemas.models import Citation, Grounding, RetrievedChunk

MARKER_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
MAX_EXCERPT_CHARS = 700


def extract_markers(text: str) -> list[str]:
    """All ``S#`` markers in order of first appearance (``[S1, S3]`` supported)."""
    seen: list[str] = []
    for group in MARKER_RE.findall(text):
        for marker in (m.strip() for m in group.split(",")):
            if marker not in seen:
                seen.append(marker)
    return seen


def strip_invalid_markers(text: str, valid: set[str]) -> str:
    def repl(match: re.Match) -> str:
        kept = [m.strip() for m in match.group(1).split(",") if m.strip() in valid]
        return "".join(f"[{m}]" for m in kept)

    cleaned = MARKER_RE.sub(repl, text)
    return re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)


_TRAILING_MARKERS_RE = re.compile(r"([.;:?!])\s*((?:\[S\d+(?:\s*,\s*S\d+)*\]\s*)+)")


def _attach_trailing_markers(line: str) -> str:
    """'... days. [S1]' -> '... days [S1].' so the marker stays with its sentence."""
    return _TRAILING_MARKERS_RE.sub(lambda m: f" {m.group(2).strip()}{m.group(1)} ", line).strip()


def _answer_sentences(answer: str) -> list[str]:
    sentences: list[str] = []
    for line in answer.splitlines():
        line = _attach_trailing_markers(line.strip().lstrip("-*•").strip())
        if not line or line.endswith(":") or line.startswith("#"):
            continue  # headings
        sentences.extend(s for s in split_sentences(line) if len(tokenize(MARKER_RE.sub("", s))) >= 4)
    return sentences


def claims_by_marker(answer: str) -> dict[str, list[str]]:
    claims: dict[str, list[str]] = {}
    for sentence in _answer_sentences(answer):
        for marker in extract_markers(sentence):
            claims.setdefault(marker, []).append(MARKER_RE.sub("", sentence))
    return claims


def best_excerpt(chunk_text: str, claims: list[str], max_chars: int = MAX_EXCERPT_CHARS) -> str:
    """The window of chunk sentences with the highest token overlap with ``claims``."""
    text = " ".join(chunk_text.split())
    if len(text) <= max_chars:
        return text
    sentences = split_sentences(text)
    if not claims:
        out = ""
        for s in sentences:
            if len(out) + len(s) + 1 > max_chars:
                break
            out = f"{out} {s}".strip()
        return out or text[:max_chars]
    claim_tokens = set(tokenize(" ".join(claims)))
    scores = [len(claim_tokens & set(tokenize(s))) for s in sentences]
    best = max(range(len(sentences)), key=lambda i: scores[i])
    lo = hi = best
    excerpt = sentences[best]
    while True:
        grown = False
        if lo > 0 and len(excerpt) + len(sentences[lo - 1]) + 1 <= max_chars:
            lo -= 1
            excerpt = f"{sentences[lo]} {excerpt}"
            grown = True
        if hi < len(sentences) - 1 and len(excerpt) + len(sentences[hi + 1]) + 1 <= max_chars:
            hi += 1
            excerpt = f"{excerpt} {sentences[hi]}"
            grown = True
        if not grown:
            break
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    prefix = "… " if lo > 0 else ""
    suffix = " …" if hi < len(sentences) - 1 and not excerpt.endswith("…") else ""
    return f"{prefix}{excerpt}{suffix}"


def to_citation(marker: str, item: RetrievedChunk, claims: list[str] | None = None) -> Citation:
    c = item.chunk
    return Citation(
        source_id=marker,
        chunk_id=c.chunk_id,
        doc_id=c.doc_id,
        title=c.title,
        circular_number=c.circular_number,
        publication_date=c.publication_date,
        page_number=c.page_number,
        section=c.section,
        excerpt=best_excerpt(c.text, claims or []),
        source_filename=c.source_filename,
        source_url=c.source_url,
        relevance_score=item.rerank_score if item.rerank_score is not None else item.semantic_score,
    )


def resolve_citations(answer: str, retrieved: list[RetrievedChunk]) -> tuple[str, list[Citation], list[str]]:
    """Return ``(cleaned_answer, citations_in_order, invalid_markers)``."""
    valid = {f"S{i}" for i in range(1, len(retrieved) + 1)}
    markers = extract_markers(answer)
    invalid = [m for m in markers if m not in valid]
    cleaned = strip_invalid_markers(answer, valid) if invalid else answer
    claims = claims_by_marker(cleaned)
    citations = [
        to_citation(m, retrieved[int(m[1:]) - 1], claims.get(m)) for m in markers if m in valid
    ]
    return cleaned, citations, invalid


def grounding_metrics(answer: str, citations: list[Citation], retrieved: list[RetrievedChunk], invalid: list[str]) -> Grounding:
    sentences = _answer_sentences(answer)
    with_cite = [s for s in sentences if MARKER_RE.search(s)]
    cited_text = " ".join(retrieved[int(c.source_id[1:]) - 1].chunk.text for c in citations)
    source_tokens = set(tokenize(cited_text))
    answer_tokens = [t for t in tokenize(MARKER_RE.sub("", answer)) if len(t) > 2]
    support = sum(1 for t in answer_tokens if t in source_tokens) / len(answer_tokens) if answer_tokens else 0.0
    return Grounding(
        cited_sources=len(citations),
        invalid_citation_markers=invalid,
        sentences_total=len(sentences),
        sentences_with_citation=len(with_cite),
        citation_coverage=round(len(with_cite) / len(sentences), 3) if sentences else 0.0,
        lexical_support=round(support, 3),
    )
