"""Sentence-Transformers embedding wrapper.

Documents and queries are embedded with the same model. Vectors are
L2-normalised so that inner product == cosine similarity in FAISS.
"""

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class Embedder(Protocol):
    model_name: str

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class SentenceTransformerEmbedder:
    """Lazily-loaded SentenceTransformer model (thread-safe)."""

    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model", extra={"model": self.model_name})
                    self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        dim = getter()
        if dim is None:
            raise RuntimeError(f"Model {self.model_name} does not report an embedding dimension")
        return int(dim)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return l2_normalise(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self.model.encode(
            [self.query_prefix + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return l2_normalise(vector)[0]
