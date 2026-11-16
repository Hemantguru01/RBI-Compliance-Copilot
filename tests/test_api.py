import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.core.exceptions import LLMGenerationError
from tests.conftest import ScriptedLLM


@pytest.fixture
def client_factory(workspace, make_container):
    root, raw = workspace

    def _make(ingest: bool = True, llm=None):
        container = make_container(root, raw, llm=llm)
        if ingest:
            container.ingest()
        app = create_app(settings=container.settings, container=container)
        return TestClient(app), container

    return _make


def test_health_reports_index_and_llm_state(client_factory):
    client, _ = client_factory()
    with client:
        body = client.get("/health").json()
    assert body["status"] == "ok" and body["index_ready"] and body["documents"] == 3
    assert body["llm_provider"] == "extractive" and body["llm_configured"] is False
    assert body["embedding_model"] == "BAAI/bge-small-en-v1.5"


def test_documents_endpoint(client_factory):
    client, _ = client_factory()
    with client:
        body = client.get("/documents").json()
    assert body["count"] == 3
    doc = next(d for d in body["documents"] if d["source_filename"] == "fixture_cyber_security.pdf")
    assert doc["circular_number"].startswith("RBI/2099-00/903") and doc["publication_date"] == "2099-06-30"
    assert doc["chunk_count"] >= 2 and doc["page_count"] == 2


def test_query_returns_grounded_answer_with_citations(client_factory):
    client, _ = client_factory()
    with client:
        r = client.post("/query", json={"question": "Within how many hours must a cyber incident be reported?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["provider"] == "extractive"
    assert "6 hours" in body["answer"]
    cite = body["citations"][0]
    assert cite["source_filename"] == "fixture_cyber_security.pdf" and cite["page_number"] == 1
    assert cite["circular_number"] and cite["excerpt"]
    assert body["latency"]["total_ms"] > 0 and body["confidence"]["level"] in {"high", "medium", "low"}
    assert "X-Process-Time-Ms" in r.headers


def test_unsupported_question_is_handled_safely(client_factory):
    client, _ = client_factory()
    with client:
        body = client.post("/query", json={"question": "What is the best recipe for chocolate cake?"}).json()
    assert body["status"] == "insufficient_evidence" and body["citations"] == []
    assert "sufficient evidence" in body["answer"]


@pytest.mark.parametrize("payload", [
    {"question": "hi"},
    {"question": "   "},
    {"question": "x" * 1001},
    {"question": "Valid question here", "top_k": 0},
    {"question": "Valid question here", "top_k": 99},
    {},
])
def test_invalid_input_rejected(client_factory, payload):
    client, _ = client_factory()
    with client:
        assert client.post("/query", json=payload).status_code == 422


def test_unknown_document_filter_rejected(client_factory):
    client, _ = client_factory()
    with client:
        r = client.post("/query", json={"question": "complaint timelines", "doc_ids": ["nope"]})
    assert r.status_code == 422 and "nope" in r.json()["detail"]


def test_document_filter_applied(client_factory):
    client, container = client_factory()
    cyber = next(d.doc_id for d in container.store.documents.values() if "cyber" in d.source_filename)
    with client:
        body = client.post("/query", json={"question": "To whom does the Chief Information Security Officer report?", "doc_ids": [cyber]}).json()
    assert body["retrieved"] and all(c["doc_id"] == cyber for c in body["retrieved"])


def test_query_on_empty_index_returns_503(client_factory):
    client, container = client_factory(ingest=False)
    container.store.clear()
    with client:
        r = client.post("/query", json={"question": "Any question at all?"})
    assert r.status_code == 503


def test_llm_failure_maps_to_502(client_factory):
    client, _ = client_factory(llm=ScriptedLLM(error=LLMGenerationError("LLM API error 500")))
    with client:
        r = client.post("/query", json={"question": "How quickly must complaints be resolved?"})
    assert r.status_code == 502 and r.json()["error_type"] == "LLMGenerationError"


def test_ingest_and_reindex_endpoints(client_factory):
    client, _ = client_factory(ingest=False)
    with client:
        first = client.post("/ingest", json={}).json()
        second = client.post("/ingest").json()
        rebuilt = client.post("/reindex").json()
    assert len(first["added"]) == 3 and len(second["unchanged"]) == 3 and len(rebuilt["added"]) == 3


def test_upload_validation_and_success(client_factory, fixture_pdf_dir):
    client, container = client_factory(ingest=False)
    with client:
        assert client.post("/documents/upload", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 400
        assert client.post("/documents/upload", files={"file": ("x.pdf", b"not a pdf", "application/pdf")}).status_code == 400
        data = (fixture_pdf_dir / "fixture_cyber_security.pdf").read_bytes()
        r = client.post("/documents/upload", files={"file": ("../../evil name.pdf", data, "application/pdf")})
    assert r.status_code == 201
    assert r.json()["filename"] == "evil_name.pdf"
    assert (container.settings.raw_data_dir / "evil_name.pdf").exists()


def test_pdf_download_for_citation_verification(client_factory):
    client, container = client_factory()
    doc_id = next(iter(container.store.documents))
    with client:
        r = client.get(f"/documents/{doc_id}/pdf")
        missing = client.get("/documents/unknown/pdf")
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
    assert missing.status_code == 404


def test_openapi_docs_available(client_factory):
    client, _ = client_factory()
    with client:
        spec = client.get("/openapi.json").json()
    assert {"/health", "/documents", "/query", "/ingest", "/reindex"} <= set(spec["paths"])
