"""Wires the components together. One instance is shared by the API process.

Components can be injected (tests pass lightweight fakes); otherwise they are
built from ``Settings``.
"""

from __future__ import annotations

import threading

from app.core.config import Settings
from app.core.exceptions import IndexCorruptedError, IndexNotReadyError
from app.core.logging import get_logger
from app.embeddings.embedder import Embedder, SentenceTransformerEmbedder
from app.generation.llm import AnthropicProvider, ExtractiveProvider, LLMProvider
from app.retrieval.keyword_search import KeywordIndex
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import RetrievalConfig, Retriever
from app.retrieval.vector_store import VectorStore
from app.schemas.models import IngestResponse
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService

logger = get_logger(__name__)


class AppContainer:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        reranker: CrossEncoderReranker | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.settings = settings
        self.messages: list[str] = []
        self._ingest_lock = threading.Lock()

        self.embedder: Embedder = embedder or SentenceTransformerEmbedder(
            settings.embedding_model,
            query_prefix=settings.embedding_query_prefix,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
        )
        self.store = VectorStore(
            settings.vector_store_dir,
            dimension=self.embedder.dimension,
            embedding_model=self.embedder.model_name,
            config_signature={
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "min_chunk_chars": settings.min_chunk_chars,
            },
        )
        self.keyword_index = KeywordIndex()
        if reranker is None and settings.enable_reranker:
            reranker = CrossEncoderReranker(settings.reranker_model, device=settings.embedding_device)
        self.retriever = Retriever(
            self.store,
            self.embedder,
            keyword_index=self.keyword_index,
            reranker=reranker,
            config=RetrievalConfig(
                top_k=settings.top_k,
                candidate_k=settings.candidate_k,
                similarity_threshold=settings.similarity_threshold,
                enable_hybrid=settings.enable_hybrid_search,
                enable_rerank=settings.enable_reranker and reranker is not None,
                rerank_threshold=settings.rerank_threshold,
            ),
        )
        self.ingestion = IngestionService(
            self.store,
            self.embedder,
            settings.raw_data_dir,
            settings.processed_data_dir,
            settings.chunk_size,
            settings.chunk_overlap,
            settings.min_chunk_chars,
        )
        self.llm = llm or self._build_llm()
        self.rag = RAGService(
            self.retriever,
            self.llm,
            max_history_turns=settings.max_history_turns,
            query_log_path=settings.log_dir / "queries.jsonl" if settings.enable_query_log else None,
        )

    def _build_llm(self) -> LLMProvider:
        s = self.settings
        if s.llm_provider == "anthropic":
            if s.llm_available:
                return AnthropicProvider(
                    api_key=s.anthropic_api_key.get_secret_value() if s.anthropic_api_key else None,
                    model=s.llm_model,
                    effort=s.llm_effort,
                    max_tokens=s.llm_max_tokens,
                    timeout=s.llm_timeout_seconds,
                    enable_refusal_fallback=s.llm_enable_refusal_fallback,
                )
            msg = "ANTHROPIC_API_KEY is not set: using the offline extractive answerer (verbatim quotes, no LLM synthesis)."
            logger.warning(msg)
            self.messages.append(msg)
        return ExtractiveProvider(self.embedder)

    @property
    def llm_configured(self) -> bool:
        return self.llm.name != "extractive"

    @property
    def index_ready(self) -> bool:
        return not self.store.is_empty

    def startup(self) -> None:
        """Load the saved index; rebuild it if it is corrupted or stale."""
        try:
            self.store.load()
        except IndexNotReadyError:
            self.messages.append("No index found. Add PDFs to data/raw/ and call POST /ingest.")
        except IndexCorruptedError as exc:
            logger.error("Vector store unusable, rebuilding", extra={"error": str(exc)})
            self.messages.append(f"Index was unusable ({exc}); rebuilt from data/raw/.")
            self.store.clear()
            if self.ingestion.list_pdfs():
                self.ingest(force=True)
        self.keyword_index.build(self.store.chunks)

    def ingest(self, force: bool = False) -> IngestResponse:
        if not self._ingest_lock.acquire(blocking=False):
            raise RuntimeError("An ingestion job is already running")
        try:
            result = self.ingestion.ingest(force=force)
            self.keyword_index.build(self.store.chunks)
            return result
        finally:
            self._ingest_lock.release()
