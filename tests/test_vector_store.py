import json

import numpy as np
import pytest

from app.core.exceptions import IndexCorruptedError, IndexNotReadyError
from app.retrieval.vector_store import VectorStore
from app.schemas.models import Chunk, DocumentRecord

DIM = 8


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _doc(doc_id: str, n: int, seed: int):
    rec = DocumentRecord(doc_id=doc_id, title=f"Doc {doc_id}", source_filename=f"{doc_id}.pdf", sha256=doc_id * 4,
                         page_count=n, char_count=10)
    chunks = [Chunk(chunk_id=f"{doc_id}:p{i + 1}:c{i}", doc_id=doc_id, title=rec.title, source_filename=rec.source_filename,
                    page_number=i + 1, chunk_index=i, text=f"text {doc_id} {i}") for i in range(n)]
    vecs = np.vstack([_unit(seed + i) for i in range(n)])
    return rec, chunks, vecs


def _store(tmp_path, **kw) -> VectorStore:
    return VectorStore(tmp_path / "vs", DIM, kw.get("model", "test-model"), kw.get("cfg", {"chunk_size": 1}))


def test_exact_vector_is_top_hit_with_cosine_one(tmp_path):
    store = _store(tmp_path)
    rec, chunks, vecs = _doc("a", 5, 0)
    store.add_document(rec, chunks, vecs)
    hits = store.search(vecs[3], k=3)
    top_id, score = hits[0]
    assert store.chunks[top_id].chunk_id == "a:p4:c3"
    assert score == pytest.approx(1.0, abs=1e-5)
    assert [s for _, s in hits] == sorted([s for _, s in hits], reverse=True)


def test_save_and_load_roundtrip(tmp_path):
    store = _store(tmp_path)
    for d, seed in (("a", 0), ("b", 100)):
        store.add_document(*_doc(d, 4, seed))
    store.save()
    loaded = _store(tmp_path)
    loaded.load()
    assert loaded.index.ntotal == 8
    assert set(loaded.documents) == {"a", "b"}
    q = _doc("b", 4, 100)[2][2]
    assert loaded.chunks[loaded.search(q, 1)[0][0]].chunk_id == "b:p3:c2"


def test_remove_document_and_incremental_add(tmp_path):
    store = _store(tmp_path)
    store.add_document(*_doc("a", 3, 0))
    store.add_document(*_doc("b", 3, 50))
    assert store.remove_document("a") == 3
    assert store.index.ntotal == 3 and "a" not in store.documents
    assert all(c.doc_id == "b" for c in store.chunks.values())
    store.add_document(*_doc("c", 2, 70))
    assert store.index.ntotal == 5
    # Re-adding an existing doc replaces it instead of duplicating vectors.
    store.add_document(*_doc("c", 2, 70))
    assert store.index.ntotal == 5


def test_document_filter_restricts_results(tmp_path):
    store = _store(tmp_path)
    store.add_document(*_doc("a", 4, 0))
    store.add_document(*_doc("b", 4, 10))
    q = _doc("a", 4, 0)[2][0]
    hits = store.search(q, 10, doc_ids=["b"])
    assert hits and all(store.chunks[i].doc_id == "b" for i, _ in hits)
    assert store.search(q, 10, doc_ids=["missing"]) == []


def test_similarity_matches_search_scores(tmp_path):
    store = _store(tmp_path)
    store.add_document(*_doc("a", 4, 0))
    q = _unit(999)
    hits = store.search(q, 4)
    sims = store.similarity(q, [i for i, _ in hits])
    for i, s in hits:
        assert sims[i] == pytest.approx(s, abs=1e-5)


def test_empty_store_search_raises(tmp_path):
    with pytest.raises(IndexNotReadyError):
        _store(tmp_path).search(_unit(0), 3)


def test_missing_index_files_raise_not_ready(tmp_path):
    with pytest.raises(IndexNotReadyError):
        _store(tmp_path).load()


def test_corrupted_index_file_detected(tmp_path):
    store = _store(tmp_path)
    store.add_document(*_doc("a", 3, 0))
    store.save()
    store.index_path.write_bytes(b"garbage")
    with pytest.raises(IndexCorruptedError):
        _store(tmp_path).load()


def test_metadata_vector_count_mismatch_detected(tmp_path):
    store = _store(tmp_path)
    store.add_document(*_doc("a", 3, 0))
    store.save()
    lines = store.chunks_path.read_text().splitlines()
    store.chunks_path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(IndexCorruptedError, match="vectors but"):
        _store(tmp_path).load()


def test_embedding_model_or_config_change_detected(tmp_path):
    store = _store(tmp_path)
    store.add_document(*_doc("a", 3, 0))
    store.save()
    with pytest.raises(IndexCorruptedError, match="built with"):
        _store(tmp_path, model="other-model").load()
    with pytest.raises(IndexCorruptedError, match="chunking configuration"):
        _store(tmp_path, cfg={"chunk_size": 2}).load()


def test_dimension_mismatch_on_add_rejected(tmp_path):
    store = _store(tmp_path)
    rec, chunks, _ = _doc("a", 2, 0)
    with pytest.raises(ValueError):
        store.add_document(rec, chunks, np.ones((2, DIM + 1), dtype=np.float32))
    with pytest.raises(ValueError):
        store.add_document(rec, chunks, np.ones((1, DIM), dtype=np.float32))


def test_meta_file_records_model_and_dimension(tmp_path):
    store = _store(tmp_path)
    store.add_document(*_doc("a", 2, 0))
    store.save()
    meta = json.loads(store.meta_path.read_text())
    assert meta["embedding_model"] == "test-model" and meta["dimension"] == DIM and meta["vectors"] == 2
