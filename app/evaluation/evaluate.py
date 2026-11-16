"""Offline evaluation of retrieval relevance and answer grounding.

Gold file format (JSON lines), one question per line:

    {"question": "...", "expected_source": "Know_Your_Customer", "expected_pages": [12, 13]}
    {"question": "...", "expected_source": null}      # out of scope -> must refuse

``expected_source`` is a case-insensitive substring of the expected PDF's
filename; ``expected_pages`` is optional. Metrics:

* **hit@k**   – share of in-scope questions whose expected document appears in the top-k
* **MRR**     – mean reciprocal rank of the first chunk from the expected document
* **page hit@k** – same as hit@k, but the chunk must also be on an expected page
* **refusal accuracy** – share of out-of-scope questions answered with "insufficient evidence"
* **false refusals** – in-scope questions that were refused
* **citation coverage / lexical support** – averaged over answered questions
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.schemas.models import QueryRequest
from app.services.rag_service import RAGService


@dataclass
class EvalCase:
    question: str
    expected_source: str | None
    expected_pages: list[int] | None = None


@dataclass
class EvalReport:
    in_scope: int = 0
    out_of_scope: int = 0
    hit_at_k: float = 0.0
    page_hit_at_k: float | None = None
    mrr: float = 0.0
    refusal_accuracy: float | None = None
    false_refusals: int = 0
    mean_citation_coverage: float | None = None
    mean_lexical_support: float | None = None
    mean_latency_ms: float = 0.0
    failures: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_cases(path: Path) -> list[EvalCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("//"):
            row = json.loads(line)
            cases.append(EvalCase(row["question"], row.get("expected_source"), row.get("expected_pages")))
    return cases


def evaluate(rag: RAGService, cases: list[EvalCase], top_k: int = 5) -> EvalReport:
    report = EvalReport()
    hits = page_hits = page_cases = 0
    rr_total = 0.0
    refusals_correct = 0
    coverages: list[float] = []
    supports: list[float] = []
    latencies: list[float] = []

    for case in cases:
        resp = rag.answer(QueryRequest(question=case.question, top_k=top_k))
        latencies.append(resp.latency.total_ms)
        if resp.grounding:
            coverages.append(resp.grounding.citation_coverage)
            supports.append(resp.grounding.lexical_support)

        if case.expected_source is None:
            report.out_of_scope += 1
            if resp.status == "insufficient_evidence":
                refusals_correct += 1
            else:
                report.failures.append({"question": case.question, "problem": "answered an out-of-scope question"})
            continue

        report.in_scope += 1
        needle = case.expected_source.lower()
        ranks = [i for i, c in enumerate(resp.retrieved, start=1) if needle in c.source_filename.lower()]
        if ranks:
            hits += 1
            rr_total += 1.0 / ranks[0]
        else:
            report.failures.append(
                {"question": case.question, "problem": "expected document not retrieved",
                 "retrieved": [c.source_filename for c in resp.retrieved]}
            )
        if case.expected_pages:
            page_cases += 1
            if any(needle in c.source_filename.lower() and c.page_number in case.expected_pages for c in resp.retrieved):
                page_hits += 1
        if resp.status == "insufficient_evidence":
            report.false_refusals += 1

    if report.in_scope:
        report.hit_at_k = round(hits / report.in_scope, 3)
        report.mrr = round(rr_total / report.in_scope, 3)
    if page_cases:
        report.page_hit_at_k = round(page_hits / page_cases, 3)
    if report.out_of_scope:
        report.refusal_accuracy = round(refusals_correct / report.out_of_scope, 3)
    if coverages:
        report.mean_citation_coverage = round(sum(coverages) / len(coverages), 3)
        report.mean_lexical_support = round(sum(supports) / len(supports), 3)
    report.mean_latency_ms = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
    return report
