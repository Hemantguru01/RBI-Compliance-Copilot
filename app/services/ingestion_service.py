"""Incremental ingestion: scan PDFs -> extract -> chunk -> embed -> index.

Change detection is by SHA-256 of the file bytes, so unchanged documents are
never re-extracted or re-embedded. Exact duplicates (same bytes) and text
duplicates (same normalised text, different bytes) are skipped.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from app.core.exceptions import PDFExtractionError
from app.core.logging import get_logger
from app.embeddings.embedder import Embedder
from app.ingestion.chunker import chunk_document
from app.ingestion.pdf_loader import extract_pdf, file_sha256, load_manifest
from app.retrieval.vector_store import VectorStore
from app.schemas.models import IngestResponse

logger = get_logger(__name__)


def text_fingerprint(pages_text: list[str]) -> str:
    normalised = " ".join(" ".join(pages_text).lower().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


class IngestionService:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        raw_dir: Path,
        processed_dir: Path,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_chars: int,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars

    def list_pdfs(self) -> list[Path]:
        if not self.raw_dir.exists():
            return []
        return sorted(p for p in self.raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")

    def ingest(self, force: bool = False) -> IngestResponse:
        start = time.perf_counter()
        manifest = load_manifest(self.raw_dir)
        if force:
            self.store.clear()

        known_by_file = {d.source_filename: d for d in self.store.documents.values()}
        known_hashes = {d.sha256: d for d in self.store.documents.values()}
        known_fingerprints: dict[str, str] = {}
        for doc in self.store.documents.values():
            fp_path = self.processed_dir / f"{doc.doc_id}.json"
            if fp_path.exists():
                try:
                    known_fingerprints[json.loads(fp_path.read_text(encoding="utf-8"))["text_fingerprint"]] = doc.source_filename
                except (json.JSONDecodeError, KeyError):
                    pass

        added, updated, unchanged, removed = [], [], [], []
        skipped: dict[str, str] = {}
        seen_files: set[str] = set()

        for path in self.list_pdfs():
            seen_files.add(path.name)
            try:
                digest = file_sha256(path)
            except OSError as exc:
                skipped[path.name] = f"unreadable: {exc}"
                continue

            previous = known_by_file.get(path.name)
            if previous and previous.sha256 == digest:
                unchanged.append(path.name)
                continue
            if digest in known_hashes and known_hashes[digest].source_filename != path.name:
                skipped[path.name] = f"duplicate of {known_hashes[digest].source_filename}"
                continue

            try:
                extracted = extract_pdf(path, manifest.get(path.name), sha256=digest)
            except PDFExtractionError as exc:
                skipped[path.name] = str(exc)
                logger.warning("Skipping PDF", extra={"file": path.name, "reason": str(exc)})
                continue

            fingerprint = text_fingerprint([p.text for p in extracted.pages])
            dup_of = known_fingerprints.get(fingerprint)
            if dup_of and dup_of != path.name:
                skipped[path.name] = f"same text as {dup_of}"
                continue

            chunks = chunk_document(
                extracted.record, extracted.pages, self.chunk_size, self.chunk_overlap, self.min_chunk_chars
            )
            if not chunks:
                skipped[path.name] = "no chunks produced"
                continue
            vectors = self.embedder.embed_documents([c.embedding_text() for c in chunks])

            if previous:
                self.store.remove_document(previous.doc_id)
                self._delete_processed(previous.doc_id)
                updated.append(path.name)
            else:
                added.append(path.name)
            self.store.add_document(extracted.record, chunks, vectors)
            known_hashes[digest] = extracted.record
            known_fingerprints[fingerprint] = path.name
            self._save_processed(extracted, fingerprint, len(chunks))
            logger.info("Indexed document", extra={"file": path.name, "chunks": len(chunks)})

        # Documents whose files were deleted from data/raw are removed from the index.
        for doc in list(self.store.documents.values()):
            if doc.source_filename not in seen_files:
                self.store.remove_document(doc.doc_id)
                self._delete_processed(doc.doc_id)
                removed.append(doc.source_filename)

        if added or updated or removed or force:
            self.store.save()

        return IngestResponse(
            added=added,
            updated=updated,
            removed=removed,
            unchanged=unchanged,
            skipped=skipped,
            total_documents=len(self.store.documents),
            total_chunks=len(self.store.chunks),
            duration_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    def _save_processed(self, extracted, fingerprint: str, chunk_count: int) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "record": extracted.record.model_dump(),
            "text_fingerprint": fingerprint,
            "chunk_count": chunk_count,
            "pages": [p.model_dump() for p in extracted.pages],
        }
        (self.processed_dir / f"{extracted.record.doc_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def _delete_processed(self, doc_id: str) -> None:
        (self.processed_dir / f"{doc_id}.json").unlink(missing_ok=True)
