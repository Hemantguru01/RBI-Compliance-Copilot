"""Shared pytest fixtures.

Tests run against *synthetic* PDF fixtures generated on the fly (see
``app/utils/fixtures.py``) and the real embedding / reranker models. External
LLM calls are always mocked.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.core.config import Settings
from app.generation.llm import LLMResult
from app.utils.fixtures import generate_fixture_pdfs


class ScriptedLLM:
    """Fake LLM provider: returns a fixed (or computed) answer and records calls."""

    name = "fake"
    model = "fake-model"

    def __init__(self, answer: str | None = None, error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.calls: list[dict] = []

    def generate(self, question, retrieved, history):
        self.calls.append({"question": question, "retrieved": retrieved, "history": history})
        if self.error:
            raise self.error
        text = self.answer if self.answer is not None else f"The requirement is stated in the source [S1]."
        return LLMResult(text=text, model=self.model, provider=self.name)


@pytest.fixture(scope="session")
def fixture_pdf_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("fixture_pdfs")
    generate_fixture_pdfs(directory)
    return directory


@pytest.fixture(scope="session")
def embedder():
    from app.embeddings.embedder import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder(
        "BAAI/bge-small-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
    )
    try:
        _ = emb.dimension
    except Exception as exc:  # pragma: no cover - only without the cached model and network
        pytest.skip(f"Embedding model unavailable: {exc}")
    return emb


@pytest.fixture(scope="session")
def reranker():
    from app.retrieval.reranker import CrossEncoderReranker

    rr = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    try:
        rr.score("test", ["test passage"])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Reranker model unavailable: {exc}")
    return rr


def make_settings(root: Path, raw_dir: Path, **overrides) -> Settings:
    values = dict(
        raw_data_dir=raw_dir,
        processed_data_dir=root / "processed",
        vector_store_dir=root / "vector_store",
        log_dir=root / "logs",
        llm_provider="extractive",
        anthropic_api_key=None,
        chunk_size=800,
        chunk_overlap=150,
        min_chunk_chars=40,
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def workspace(tmp_path, fixture_pdf_dir) -> tuple[Path, Path]:
    """A private copy of the fixture PDFs that a test may modify."""
    raw = tmp_path / "raw"
    shutil.copytree(fixture_pdf_dir, raw)
    return tmp_path, raw


@pytest.fixture
def make_container(embedder, reranker):
    from app.services.container import AppContainer

    def _make(root: Path, raw: Path, llm=None, **overrides):
        settings = make_settings(root, raw, **overrides)
        container = AppContainer(settings, embedder=embedder, reranker=reranker, llm=llm)
        container.startup()
        return container

    return _make


@pytest.fixture(scope="session")
def indexed_container(tmp_path_factory, fixture_pdf_dir, embedder, reranker):
    """Read-only container with the fixture PDFs already indexed (shared for speed)."""
    from app.services.container import AppContainer

    root = tmp_path_factory.mktemp("indexed")
    settings = make_settings(root, fixture_pdf_dir)
    container = AppContainer(settings, embedder=embedder, reranker=reranker)
    container.startup()
    container.ingest()
    return container
