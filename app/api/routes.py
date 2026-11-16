"""HTTP endpoints. Business logic lives in ``app.services``."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from app.core.logging import get_logger
from app.schemas.models import (
    DocumentInfo,
    DocumentsResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.container import AppContainer

logger = get_logger(__name__)
router = APIRouter()

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._\-]+")


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(c: AppContainer = Depends(get_container)) -> HealthResponse:
    """Service status, index statistics and LLM configuration."""
    messages = list(c.messages)
    if not c.index_ready:
        messages.append("Index is empty: queries will fail until documents are ingested.")
    return HealthResponse(
        status="ok" if c.index_ready else "degraded",
        index_ready=c.index_ready,
        documents=len(c.store.documents),
        chunks=len(c.store.chunks),
        embedding_model=c.embedder.model_name,
        llm_provider=c.llm.name,
        llm_model=c.llm.model,
        llm_configured=c.llm_configured,
        messages=messages,
    )


@router.get("/documents", response_model=DocumentsResponse, tags=["documents"])
def list_documents(c: AppContainer = Depends(get_container)) -> DocumentsResponse:
    """All indexed documents with their extracted metadata."""
    counts = c.store.chunk_counts()
    docs = sorted(c.store.documents.values(), key=lambda d: d.title.lower())
    return DocumentsResponse(
        count=len(docs),
        documents=[DocumentInfo(**d.model_dump(), chunk_count=counts.get(d.doc_id, 0)) for d in docs],
    )


@router.get("/documents/{doc_id}/pdf", tags=["documents"], response_class=FileResponse)
def get_document_pdf(doc_id: str, c: AppContainer = Depends(get_container)) -> FileResponse:
    """Serve the original PDF so a citation can be verified against the source page."""
    doc = c.store.documents.get(doc_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown document")
    path = (c.settings.raw_data_dir / doc.source_filename).resolve()
    if c.settings.raw_data_dir.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Source file is no longer available")
    return FileResponse(path, media_type="application/pdf", filename=doc.source_filename,
                        content_disposition_type="inline")


@router.post("/documents/upload", tags=["documents"], status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...), c: AppContainer = Depends(get_container)) -> dict:
    """Save a PDF into ``data/raw/``. Call ``POST /ingest`` afterwards to index it."""
    name = _SAFE_NAME_RE.sub("_", Path(file.filename or "").name).strip("._")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only .pdf files are accepted")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "File exceeds 50 MB")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is not a valid PDF")
    c.settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    dest = c.settings.raw_data_dir / name
    dest.write_bytes(content)
    logger.info("Uploaded PDF", extra={"file": name, "bytes": len(content)})
    return {"filename": name, "bytes": len(content), "next_step": "POST /ingest"}


@router.post("/query", response_model=QueryResponse, tags=["qa"])
def query(body: QueryRequest, c: AppContainer = Depends(get_container)) -> QueryResponse:
    """Answer a question from the indexed RBI documents, with citations."""
    if not c.index_ready:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Index is empty. Ingest documents first.")
    if body.doc_ids:
        unknown = [d for d in body.doc_ids if d not in c.store.documents]
        if unknown:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Unknown doc_ids: {', '.join(unknown)}")
    return c.rag.answer(body)


@router.post("/ingest", response_model=IngestResponse, tags=["indexing"])
def ingest(body: IngestRequest | None = None, c: AppContainer = Depends(get_container)) -> IngestResponse:
    """Incrementally index new/changed PDFs in ``data/raw/`` (unchanged files are skipped)."""
    try:
        return c.ingest(force=bool(body and body.force))
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.post("/reindex", response_model=IngestResponse, tags=["indexing"])
def reindex(c: AppContainer = Depends(get_container)) -> IngestResponse:
    """Discard the index and rebuild it from every PDF in ``data/raw/``."""
    try:
        return c.ingest(force=True)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
