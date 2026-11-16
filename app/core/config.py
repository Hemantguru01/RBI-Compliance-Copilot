"""Application configuration loaded from environment variables / ``.env``.

All tunables live here so that behaviour can be changed without code edits.
Secrets (API keys) are read from the environment and never logged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Paths -------------------------------------------------------------
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    vector_store_dir: Path = PROJECT_ROOT / "vector_store"
    log_dir: Path = PROJECT_ROOT / "logs"

    # --- Chunking ----------------------------------------------------------
    chunk_size: int = Field(1200, ge=200, le=8000, description="Target chunk size in characters")
    chunk_overlap: int = Field(200, ge=0, le=2000, description="Overlap between chunks in characters")
    min_chunk_chars: int = Field(80, ge=0, description="Chunks shorter than this are merged or dropped")

    # --- Embeddings --------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # BGE models recommend this instruction prefix for *queries* (not passages).
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = Field(32, ge=1, le=512)
    embedding_device: str = "cpu"

    # --- Retrieval ---------------------------------------------------------
    top_k: int = Field(5, ge=1, le=50)
    candidate_k: int = Field(30, ge=1, le=200, description="Candidates fetched before fusion/rerank")
    similarity_threshold: float = Field(
        0.45, ge=-1.0, le=1.0, description="Minimum cosine similarity for a candidate chunk (recall gate)"
    )
    enable_hybrid_search: bool = True
    enable_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_threshold: float = Field(
        -6.0, description="Minimum cross-encoder logit for a chunk to be kept"
    )

    # --- Generation --------------------------------------------------------
    llm_provider: Literal["anthropic", "extractive"] = "anthropic"
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_max_tokens: int = Field(4000, ge=256, le=64000)
    llm_timeout_seconds: float = Field(120.0, gt=0)
    llm_enable_refusal_fallback: bool = True
    max_history_turns: int = Field(3, ge=0, le=20)

    # --- API / frontend ----------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_base_url: str = "http://127.0.0.1:8000"
    log_level: str = "INFO"
    enable_query_log: bool = True

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def llm_available(self) -> bool:
        if self.llm_provider == "extractive":
            return True
        return self.anthropic_api_key is not None and bool(self.anthropic_api_key.get_secret_value())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
