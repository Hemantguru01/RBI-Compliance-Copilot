"""End-to-end question answering: retrieve -> generate -> cite -> score -> log."""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.logging import get_logger
from app.generation.citations import grounding_metrics, resolve_citations, to_citation
from app.generation.llm import LLMProvider
from app.generation.prompts import INSUFFICIENT_EVIDENCE_TAG
from app.retrieval.retriever import Retriever
from app.schemas.models import (
    ChatTurn,
    Confidence,
    Grounding,
    Latency,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)

logger = get_logger(__name__)

NO_EVIDENCE_ANSWER = (
    "I could not find sufficient evidence in the indexed RBI documents to answer this question. "
    "No passage met the relevance threshold, so I am not providing an answer rather than guessing. "
    "Try rephrasing the question, or check that the relevant direction or circular has been ingested."
)


_FOLLOW_UP_RE = re.compile(
    r"^(?:and|also|what about|how about|same|then)\b|\b(?:it|its|this|that|these|those|they|them|such|above|same|"
    r"the said|mentioned|previous)\b",
    re.IGNORECASE,
)


def is_follow_up(question: str) -> bool:
    return len(question.split()) <= 12 and bool(_FOLLOW_UP_RE.search(question))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_confidence(retrieved: list[RetrievedChunk], grounding: Grounding | None, insufficient: bool) -> Confidence:
    """Heuristic evidence indicator (not a calibrated probability).

    Combines: strength of the best retrieval match, how many independent
    passages support the answer, and how much of the answer is cited.
    """
    if insufficient or not retrieved:
        return Confidence(level="none", score=0.0, reasons=["No sufficiently relevant evidence was found."])
    top = retrieved[0]
    if top.rerank_score is not None:
        retrieval_strength = _sigmoid(top.rerank_score)
        reasons = [f"Best passage reranker score {top.rerank_score:.2f}"]
    else:
        # Map cosine 0.45..0.85 onto 0..1 for the bge model family.
        retrieval_strength = min(1.0, max(0.0, ((top.semantic_score or 0) - 0.45) / 0.4))
        reasons = [f"Best passage similarity {top.semantic_score:.2f}"]
    cited = grounding.cited_sources if grounding else 0
    support = min(1.0, cited / 2)
    coverage = grounding.citation_coverage if grounding else 0.0
    score = round(0.5 * retrieval_strength + 0.2 * support + 0.3 * coverage, 3)
    reasons.append(f"{cited} cited source passage(s)")
    reasons.append(f"{coverage:.0%} of answer sentences carry a citation")
    if grounding and grounding.invalid_citation_markers:
        score = round(score * 0.7, 3)
        reasons.append(f"Removed invalid citation markers: {', '.join(grounding.invalid_citation_markers)}")
    level = "high" if score >= 0.7 else "medium" if score >= 0.45 else "low"
    return Confidence(level=level, score=score, reasons=reasons)


class RAGService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        max_history_turns: int = 3,
        query_log_path: Path | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.max_history_turns = max_history_turns
        self.query_log_path = query_log_path

    def _retrieve(self, request: QueryRequest, history: list[ChatTurn]) -> list[RetrievedChunk]:
        """Retrieve for the question as asked; expand genuine follow-ups with the previous question.

        "What about NBFCs?" or "Is there a penalty for exiting it?" cannot be
        searched on their own. Expansion is only tried when the standalone
        question finds nothing *and* it reads like a follow-up, so an unrelated
        new question is never answered from the previous topic's documents.
        """
        retrieved = self.retriever.retrieve(request.question, top_k=request.top_k, doc_ids=request.doc_ids)
        previous = [t.content for t in history if t.role == "user"]
        if not retrieved and previous and is_follow_up(request.question):
            expanded = f"{previous[-1]} {request.question}"
            retrieved = self.retriever.retrieve(expanded, top_k=request.top_k, doc_ids=request.doc_ids)
        return retrieved

    def answer(self, request: QueryRequest) -> QueryResponse:
        t0 = time.perf_counter()
        history = request.history[-2 * self.max_history_turns :] if self.max_history_turns else []
        retrieved = self._retrieve(request, history)
        t1 = time.perf_counter()

        grounding: Grounding | None = None
        citations = []
        if not retrieved:
            answer, status, model = NO_EVIDENCE_ANSWER, "insufficient_evidence", None
        else:
            result = self.llm.generate(request.question, retrieved, history)
            model = result.model
            text = result.text.strip()
            if text.startswith(INSUFFICIENT_EVIDENCE_TAG):
                detail = text[len(INSUFFICIENT_EVIDENCE_TAG) :].strip(" :\n")
                answer = "The retrieved RBI passages do not contain sufficient evidence to answer this question."
                if detail:
                    answer += f" {detail}"
                answer, _, _ = resolve_citations(answer, retrieved)
                status = "insufficient_evidence"
            else:
                answer, citations, invalid = resolve_citations(text, retrieved)
                grounding = grounding_metrics(answer, citations, retrieved, invalid)
                status = "answered"
        t2 = time.perf_counter()

        response = QueryResponse(
            query_id=uuid.uuid4().hex[:12],
            question=request.question,
            answer=answer,
            status=status,
            citations=citations,
            retrieved=[to_citation(f"S{i}", r) for i, r in enumerate(retrieved, start=1)],
            confidence=compute_confidence(retrieved, grounding, status == "insufficient_evidence"),
            grounding=grounding,
            latency=Latency(
                retrieval_ms=round((t1 - t0) * 1000, 1),
                generation_ms=round((t2 - t1) * 1000, 1),
                total_ms=round((t2 - t0) * 1000, 1),
            ),
            provider=self.llm.name,
            model=model,
        )
        self._log(response, retrieved)
        return response

    def _log(self, response: QueryResponse, retrieved: list[RetrievedChunk]) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "query_id": response.query_id,
            "question": response.question,
            "status": response.status,
            "provider": response.provider,
            "model": response.model,
            "retrieved": [
                {"chunk_id": r.chunk.chunk_id, "semantic": r.semantic_score, "rerank": r.rerank_score} for r in retrieved
            ],
            "cited": [c.chunk_id for c in response.citations],
            "confidence": response.confidence.score,
            "latency_ms": response.latency.model_dump(),
        }
        logger.info("Query answered", extra={k: entry[k] for k in ("query_id", "status", "confidence")} | {"total_ms": response.latency.total_ms})
        if self.query_log_path is not None:
            try:
                self.query_log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.query_log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.warning("Could not write query log", extra={"error": str(exc)})
