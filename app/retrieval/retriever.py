"""Hybrid retrieval: semantic (FAISS) + keyword (BM25) -> RRF fusion -> gate -> rerank.

Pipeline for one question:

1. Embed the question (same model as the documents) and fetch
   ``candidate_k`` nearest chunks from FAISS.
2. Fetch ``candidate_k`` chunks from BM25 (if hybrid search is enabled).
3. Fuse both ranked lists with Reciprocal Rank Fusion (score = sum 1/(60+rank)).
4. **Relevance gate**: compute the exact cosine similarity of every fused
   candidate and drop those below ``similarity_threshold``. This is what
   prevents irrelevant context from reaching the LLM; if nothing survives,
   the question is answered with "insufficient evidence".
5. Rerank survivors with a cross-encoder (if enabled), drop those below
   ``rerank_threshold`` and keep the best ``top_k``.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger
from app.embeddings.embedder import Embedder
from app.retrieval.keyword_search import KeywordIndex
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import VectorStore
from app.schemas.models import RetrievedChunk

logger = get_logger(__name__)

RRF_K = 60


@dataclass
class RetrievalConfig:
    top_k: int = 5
    candidate_k: int = 30
    similarity_threshold: float = 0.45
    enable_hybrid: bool = True
    enable_rerank: bool = True
    rerank_threshold: float = -6.0


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        keyword_index: KeywordIndex | None = None,
        reranker: CrossEncoderReranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.keyword_index = keyword_index
        self.reranker = reranker
        self.config = config or RetrievalConfig()

    def retrieve(self, query: str, top_k: int | None = None, doc_ids: list[str] | None = None) -> list[RetrievedChunk]:
        cfg = self.config
        top_k = top_k or cfg.top_k
        query_vec = self.embedder.embed_query(query)

        semantic = self.store.search(query_vec, cfg.candidate_k, doc_ids)
        rankings = [[i for i, _ in semantic]]
        keyword_scores: dict[int, float] = {}
        if cfg.enable_hybrid and self.keyword_index is not None:
            keyword = self.keyword_index.search(query, cfg.candidate_k, doc_ids)
            keyword_scores = dict(keyword)
            rankings.append([i for i, _ in keyword])

        fused = reciprocal_rank_fusion(rankings)
        if not fused:
            return []

        # Relevance gate on exact cosine similarity (covers keyword-only hits too).
        similarities = self.store.similarity(query_vec, list(fused))
        survivors = [i for i in fused if similarities[i] >= cfg.similarity_threshold]
        logger.debug(
            "Retrieval candidates",
            extra={"candidates": len(fused), "survivors": len(survivors), "best_sim": max(similarities.values())},
        )
        if not survivors:
            return []

        survivors.sort(key=lambda i: fused[i], reverse=True)
        results = [
            RetrievedChunk(
                chunk=self.store.chunks[i],
                semantic_score=round(similarities[i], 4),
                keyword_score=round(keyword_scores[i], 4) if i in keyword_scores else None,
                fused_score=round(fused[i], 6),
            )
            for i in survivors
        ]

        if cfg.enable_rerank and self.reranker is not None:
            scores = self.reranker.score(query, [r.chunk.text for r in results])
            for r, s in zip(results, scores):
                r.rerank_score = round(s, 4)
            results = [r for r in results if (r.rerank_score or 0.0) >= cfg.rerank_threshold]
            results.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)

        return results[:top_k]
