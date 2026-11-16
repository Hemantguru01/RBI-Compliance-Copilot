"""BM25 keyword search over chunk texts.

Semantic embeddings are weak at exact tokens such as clause numbers,
acronyms ("KYC", "CRR", "SMA-2") and figures ("90 days"); BM25 catches those.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from app.schemas.models import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")
STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in into is it its may
    of on or shall should such that the their there these this to under was were what when
    which who will with within would must any all also not no other than then they""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class KeywordIndex:
    def __init__(self) -> None:
        self._ids: list[int] = []
        self._doc_ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: dict[int, Chunk]) -> None:
        self._ids = sorted(chunks)
        self._doc_ids = [chunks[i].doc_id for i in self._ids]
        corpus = [tokenize(chunks[i].embedding_text()) for i in self._ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int, doc_ids: list[str] | None = None) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        if doc_ids:
            allowed = set(doc_ids)
            mask = np.array([d in allowed for d in self._doc_ids])
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [(self._ids[i], float(scores[i])) for i in order if np.isfinite(scores[i]) and scores[i] > 0]
