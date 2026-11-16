"""FAISS vector store with separately persisted chunk metadata.

Files in ``vector_store/``:

* ``index.faiss``     – ``IndexIDMap2(IndexFlatIP)`` over L2-normalised vectors
* ``chunks.jsonl``    – one ``{"id": int, "chunk": {...}}`` per line
* ``documents.json``  – ``DocumentRecord`` per ingested document
* ``store_meta.json`` – embedding model, dimension and chunking config used

``IndexFlatIP`` performs exact search, which is fast enough for tens of
thousands of chunks and avoids recall loss from approximate indexes.
``IndexIDMap2`` lets us remove a document's vectors (``remove_ids``) and add
new ones, so changed documents are re-embedded without touching the rest.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import faiss
import numpy as np

from app.core.exceptions import IndexCorruptedError, IndexNotReadyError
from app.core.logging import get_logger
from app.schemas.models import Chunk, DocumentRecord

logger = get_logger(__name__)

STORE_VERSION = 1


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class VectorStore:
    def __init__(self, directory: Path, dimension: int, embedding_model: str, config_signature: dict | None = None):
        self.directory = directory
        self.dimension = dimension
        self.embedding_model = embedding_model
        self.config_signature = config_signature or {}
        self._lock = threading.RLock()
        self._reset()

    # ------------------------------------------------------------------ state
    def _reset(self) -> None:
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))
        self.chunks: dict[int, Chunk] = {}
        self.documents: dict[str, DocumentRecord] = {}
        self._next_id = 0

    @property
    def index_path(self) -> Path:
        return self.directory / "index.faiss"

    @property
    def chunks_path(self) -> Path:
        return self.directory / "chunks.jsonl"

    @property
    def documents_path(self) -> Path:
        return self.directory / "documents.json"

    @property
    def meta_path(self) -> Path:
        return self.directory / "store_meta.json"

    @property
    def is_empty(self) -> bool:
        return self.index.ntotal == 0

    def exists_on_disk(self) -> bool:
        return self.index_path.exists() and self.chunks_path.exists() and self.meta_path.exists()

    # --------------------------------------------------------------- mutation
    def clear(self) -> None:
        with self._lock:
            self._reset()

    def add_document(self, record: DocumentRecord, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if len(chunks) and vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} != index dimension {self.dimension}")
        with self._lock:
            if record.doc_id in self.documents:
                self.remove_document(record.doc_id)
            ids = np.arange(self._next_id, self._next_id + len(chunks), dtype=np.int64)
            if len(chunks):
                self.index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), ids)
            for i, chunk in zip(ids.tolist(), chunks):
                self.chunks[i] = chunk
            self._next_id += len(chunks)
            self.documents[record.doc_id] = record

    def remove_document(self, doc_id: str) -> int:
        with self._lock:
            ids = [i for i, c in self.chunks.items() if c.doc_id == doc_id]
            if ids:
                self.index.remove_ids(np.array(ids, dtype=np.int64))
                for i in ids:
                    del self.chunks[i]
            self.documents.pop(doc_id, None)
            return len(ids)

    # ----------------------------------------------------------------- search
    def ids_for_documents(self, doc_ids: list[str]) -> list[int]:
        wanted = set(doc_ids)
        return [i for i, c in self.chunks.items() if c.doc_id in wanted]

    def search(self, query_vector: np.ndarray, k: int, doc_ids: list[str] | None = None) -> list[tuple[int, float]]:
        """Return ``(internal_id, cosine_similarity)`` pairs, best first."""
        if self.is_empty:
            raise IndexNotReadyError("The vector index is empty. Ingest documents first.")
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        params = None
        if doc_ids:
            allowed = self.ids_for_documents(doc_ids)
            if not allowed:
                return []
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(np.array(allowed, dtype=np.int64)))
        k = min(k, self.index.ntotal)
        with self._lock:
            scores, ids = self.index.search(query, k, params=params)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def similarity(self, query_vector: np.ndarray, ids: list[int]) -> dict[int, float]:
        """Exact cosine similarity between the query and specific stored chunks."""
        if not ids:
            return {}
        with self._lock:
            vectors = np.vstack([self.index.reconstruct(int(i)) for i in ids])
        sims = vectors @ query_vector.astype(np.float32)
        return {i: float(s) for i, s in zip(ids, sims)}

    def chunk_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.chunks.values():
            counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
        return counts

    # ------------------------------------------------------------ persistence
    def save(self) -> None:
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            tmp_index = self.index_path.with_suffix(".faiss.tmp")
            faiss.write_index(self.index, str(tmp_index))
            os.replace(tmp_index, self.index_path)
            lines = [json.dumps({"id": i, "chunk": c.model_dump()}, ensure_ascii=False) for i, c in sorted(self.chunks.items())]
            _atomic_write_text(self.chunks_path, "\n".join(lines) + ("\n" if lines else ""))
            _atomic_write_text(
                self.documents_path,
                json.dumps([d.model_dump() for d in self.documents.values()], indent=2, ensure_ascii=False),
            )
            meta = {
                "version": STORE_VERSION,
                "embedding_model": self.embedding_model,
                "dimension": self.dimension,
                "next_id": self._next_id,
                "vectors": int(self.index.ntotal),
                "config": self.config_signature,
            }
            _atomic_write_text(self.meta_path, json.dumps(meta, indent=2))
        logger.info("Saved vector store", extra={"vectors": int(self.index.ntotal), "documents": len(self.documents)})

    def load(self) -> None:
        """Load from disk. Raises ``IndexCorruptedError`` on any inconsistency."""
        with self._lock:
            if not self.exists_on_disk():
                raise IndexNotReadyError("No saved index found")
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                index = faiss.read_index(str(self.index_path))
                chunks: dict[int, Chunk] = {}
                with self.chunks_path.open(encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            row = json.loads(line)
                            chunks[int(row["id"])] = Chunk(**row["chunk"])
                docs_raw = json.loads(self.documents_path.read_text(encoding="utf-8")) if self.documents_path.exists() else []
                documents = {d["doc_id"]: DocumentRecord(**d) for d in docs_raw}
            except Exception as exc:
                raise IndexCorruptedError(f"Failed to read vector store: {exc}") from exc

            problems = []
            if meta.get("version") != STORE_VERSION:
                problems.append(f"store version {meta.get('version')} != {STORE_VERSION}")
            if meta.get("embedding_model") != self.embedding_model:
                problems.append(f"index built with {meta.get('embedding_model')!r}, configured {self.embedding_model!r}")
            if index.d != self.dimension:
                problems.append(f"index dimension {index.d} != model dimension {self.dimension}")
            if index.ntotal != len(chunks):
                problems.append(f"{index.ntotal} vectors but {len(chunks)} chunk records")
            if meta.get("config", {}) != self.config_signature:
                problems.append("chunking configuration changed since the index was built")
            if {c.doc_id for c in chunks.values()} - set(documents):
                problems.append("chunks reference unknown documents")
            if problems:
                raise IndexCorruptedError("; ".join(problems))

            self.index = index
            self.chunks = chunks
            self.documents = documents
            self._next_id = int(meta.get("next_id", max(chunks, default=-1) + 1))
        logger.info("Loaded vector store", extra={"vectors": int(self.index.ntotal), "documents": len(self.documents)})
