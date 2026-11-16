"""Pydantic models shared by the pipeline (domain) and the API (request/response)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


class PageText(BaseModel):
    page_number: int = Field(..., ge=1, description="1-indexed page number in the PDF")
    text: str


class DocumentRecord(BaseModel):
    """Metadata for one ingested source document."""

    doc_id: str = Field(..., description="Stable ID derived from the file's SHA-256")
    title: str
    source_filename: str
    source_url: str | None = None
    circular_number: str | None = None
    publication_date: str | None = None
    sha256: str
    page_count: int
    char_count: int
    empty_pages: list[int] = Field(default_factory=list)


class ExtractedDocument(BaseModel):
    record: DocumentRecord
    pages: list[PageText]


class Chunk(BaseModel):
    """A retrievable unit of text. Every chunk lives on exactly one page."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    title: str
    source_filename: str
    source_url: str | None = None
    page_number: int
    circular_number: str | None = None
    publication_date: str | None = None
    section: str | None = Field(None, description="Nearest preceding section heading, if detected")
    chunk_index: int = Field(..., description="Position of the chunk within its document")
    text: str

    def embedding_text(self) -> str:
        """Text used for embedding: title + section give the chunk its context."""
        parts = [self.title]
        if self.section:
            parts.append(self.section)
        parts.append(self.text)
        return "\n".join(parts)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    semantic_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    fused_score: float = 0.0


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8000)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Natural-language question")
    top_k: int | None = Field(None, ge=1, le=20, description="Number of sources to use")
    doc_ids: list[str] | None = Field(None, max_length=50, description="Restrict search to these documents")
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 3:
            raise ValueError("question must contain at least 3 non-whitespace characters")
        return v


class Citation(BaseModel):
    source_id: str = Field(..., description="Marker used in the answer, e.g. 'S1'")
    chunk_id: str
    doc_id: str
    title: str
    circular_number: str | None
    publication_date: str | None
    page_number: int
    section: str | None
    excerpt: str
    source_filename: str
    source_url: str | None
    relevance_score: float | None = None


class Confidence(BaseModel):
    level: Literal["high", "medium", "low", "none"]
    score: float = Field(..., ge=0.0, le=1.0)
    reasons: list[str]


class Grounding(BaseModel):
    cited_sources: int
    invalid_citation_markers: list[str]
    sentences_total: int
    sentences_with_citation: int
    citation_coverage: float
    lexical_support: float = Field(..., description="Share of answer content words found in cited excerpts")


class Latency(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    status: Literal["answered", "insufficient_evidence"]
    citations: list[Citation]
    retrieved: list[Citation] = Field(..., description="All retrieved evidence, cited or not")
    confidence: Confidence
    grounding: Grounding | None
    latency: Latency
    provider: str
    model: str | None


class DocumentInfo(DocumentRecord):
    chunk_count: int


class DocumentsResponse(BaseModel):
    count: int
    documents: list[DocumentInfo]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    index_ready: bool
    documents: int
    chunks: int
    embedding_model: str
    llm_provider: str
    llm_model: str | None
    llm_configured: bool
    messages: list[str]


class IngestRequest(BaseModel):
    force: bool = Field(False, description="Re-process documents even if unchanged")


class IngestResponse(BaseModel):
    added: list[str]
    updated: list[str]
    removed: list[str]
    unchanged: list[str]
    skipped: dict[str, str] = Field(..., description="filename -> reason")
    total_documents: int
    total_chunks: int
    duration_ms: float
