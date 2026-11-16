"""Cross-encoder reranking.

A bi-encoder (the embedding model) scores query and passage independently;
a cross-encoder reads them *together* and is markedly more precise, but too
slow to run over the whole corpus. We therefore rerank only the fused
candidate list.
"""

from __future__ import annotations

import threading

from app.core.logging import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    logger.info("Loading reranker", extra={"model": self.model_name})
                    self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict([(query, p) for p in passages], show_progress_bar=False)
        return [float(s) for s in scores]
