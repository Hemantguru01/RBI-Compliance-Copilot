from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest

from app.core.exceptions import LLMConfigurationError, LLMGenerationError
from app.generation.llm import AnthropicProvider, ExtractiveProvider
from app.generation.prompts import INSUFFICIENT_EVIDENCE_TAG, SYSTEM_PROMPT, build_messages
from app.schemas.models import ChatTurn, QueryRequest
from app.services.rag_service import RAGService
from tests.conftest import ScriptedLLM
from tests.test_citations import RETRIEVED


def _response(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason, model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


def _provider(client) -> AnthropicProvider:
    return AnthropicProvider(api_key=None, client=client)


def test_prompt_contains_sources_with_metadata_and_question():
    msgs = build_messages("How fast must complaints be resolved?", RETRIEVED)
    content = msgs[-1]["content"]
    assert msgs[-1]["role"] == "user"
    assert "[S1] Directions 1 | Circular/Ref: RBI/2099-00/1" in content and "Page: 4" in content
    assert "[S2] Directions 2" in content and "Page: 7" in content
    assert "within 30 days of receipt" in content
    assert content.rstrip().endswith("markers.")
    assert "Question: How fast must complaints be resolved?" in content


def test_system_prompt_enforces_grounding_rules():
    for phrase in ("Never invent", "INSUFFICIENT_EVIDENCE", "Interpretation (not a regulatory requirement)", "[S1]"):
        assert phrase in SYSTEM_PROMPT


def test_history_is_included_and_alternates():
    history = [ChatTurn(role="user", content="What about KYC?"), ChatTurn(role="assistant", content="KYC answer")]
    msgs = build_messages("And for NBFCs?", RETRIEVED, history)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]


def test_anthropic_request_is_well_formed():
    client = MagicMock()
    client.beta.messages.create.return_value = _response("Within 30 days [S1].")
    result = _provider(client).generate("q?", RETRIEVED, [])
    kwargs = client.beta.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["system"] == SYSTEM_PROMPT
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["fallbacks"] == "default" and kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert "thinking" not in kwargs and "temperature" not in kwargs
    assert result.text == "Within 30 days [S1]." and result.output_tokens == 20


def test_refusal_and_empty_responses_raise():
    client = MagicMock()
    client.beta.messages.create.return_value = _response("", stop_reason="refusal")
    with pytest.raises(LLMGenerationError, match="declined"):
        _provider(client).generate("q", RETRIEVED, [])
    client.beta.messages.create.return_value = _response("   ")
    with pytest.raises(LLMGenerationError, match="Empty"):
        _provider(client).generate("q", RETRIEVED, [])


def test_api_errors_are_mapped():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client = MagicMock()
    client.beta.messages.create.side_effect = anthropic.AuthenticationError(
        "bad key", response=httpx.Response(401, request=req), body=None)
    with pytest.raises(LLMConfigurationError):
        _provider(client).generate("q", RETRIEVED, [])
    client.beta.messages.create.side_effect = anthropic.APIConnectionError(request=req)
    with pytest.raises(LLMGenerationError, match="reach"):
        _provider(client).generate("q", RETRIEVED, [])


def test_missing_api_key_is_a_configuration_error():
    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(api_key=None)


def test_extractive_answer_only_quotes_source_sentences(embedder):
    result = ExtractiveProvider(embedder).generate("How fast must a complaint be resolved?", RETRIEVED, [])
    lines = [ln for ln in result.text.splitlines() if ln.startswith("- ")]
    assert lines
    source_text = " ".join(r.chunk.text for r in RETRIEVED)
    for ln in lines:
        sentence = ln[2:].rsplit(" [S", 1)[0]
        assert sentence in source_text
    assert "[S1]" in result.text


def _rag(indexed_container, llm):
    return RAGService(indexed_container.retriever, llm)


def test_rag_passes_retrieved_evidence_to_llm(indexed_container):
    llm = ScriptedLLM("Complaints must be resolved within the prescribed timeline [S1].")
    resp = _rag(indexed_container, llm).answer(QueryRequest(question="How quickly must complaints be resolved?"))
    assert resp.status == "answered"
    passed = llm.calls[0]["retrieved"]
    assert any("within 30 days of receipt" in r.chunk.text for r in passed)
    # The citation is built from the metadata of the chunk the LLM was given as S1.
    first = passed[0].chunk
    assert (resp.citations[0].chunk_id, resp.citations[0].page_number, resp.citations[0].source_filename) == (
        first.chunk_id, first.page_number, first.source_filename)
    assert first.source_filename == "fixture_grievance_redress.pdf"
    assert resp.grounding.citation_coverage == 1.0
    assert resp.confidence.level in {"high", "medium"}


def test_rag_without_evidence_never_calls_llm(indexed_container):
    llm = ScriptedLLM("should not be used")
    resp = _rag(indexed_container, llm).answer(QueryRequest(question="Who won the cricket world cup?"))
    assert resp.status == "insufficient_evidence"
    assert llm.calls == [] and resp.citations == [] and resp.confidence.level == "none"


def test_llm_insufficient_evidence_tag_is_respected(indexed_container):
    llm = ScriptedLLM(f"{INSUFFICIENT_EVIDENCE_TAG}\nThe sources do not state the penalty amount.")
    resp = _rag(indexed_container, llm).answer(QueryRequest(question="What penalty applies for late complaint resolution?"))
    assert resp.status == "insufficient_evidence"
    assert "penalty amount" in resp.answer and INSUFFICIENT_EVIDENCE_TAG not in resp.answer
    assert resp.citations == []


def test_hallucinated_citation_lowers_confidence(indexed_container):
    good = _rag(indexed_container, ScriptedLLM("Complaints are resolved within 30 days of receipt [S1].")).answer(
        QueryRequest(question="How quickly must complaints be resolved?"))
    bad = _rag(indexed_container, ScriptedLLM("Complaints are resolved within 30 days of receipt [S1][S42].")).answer(
        QueryRequest(question="How quickly must complaints be resolved?"))
    assert bad.grounding.invalid_citation_markers == ["S42"] and "[S42]" not in bad.answer
    assert bad.confidence.score < good.confidence.score


def test_short_follow_up_uses_history_for_retrieval(indexed_container):
    rag = _rag(indexed_container, ScriptedLLM())
    history = [ChatTurn(role="user", content="What is the cooling-off period for digital loans?"),
               ChatTurn(role="assistant", content="3 days [S1].")]
    resp = rag.answer(QueryRequest(question="Is there a penalty for exiting it?", history=history))
    assert resp.retrieved and resp.retrieved[0].source_filename == "fixture_digital_lending.pdf"


def test_unrelated_new_question_is_not_answered_from_previous_topic(indexed_container):
    llm = ScriptedLLM()
    history = [ChatTurn(role="user", content="Within how many hours must a cyber incident be reported?"),
               ChatTurn(role="assistant", content="6 hours [S1].")]
    resp = _rag(indexed_container, llm).answer(
        QueryRequest(question="What is the best recipe for chocolate cake?", history=history))
    assert resp.status == "insufficient_evidence" and resp.retrieved == [] and llm.calls == []


def test_follow_up_detection():
    from app.services.rag_service import is_follow_up
    assert is_follow_up("What about NBFCs?") and is_follow_up("Does this apply to them?")
    assert not is_follow_up("What is the best recipe for chocolate cake?")


def test_query_log_written(indexed_container, tmp_path):
    log = tmp_path / "q.jsonl"
    rag = RAGService(indexed_container.retriever, ScriptedLLM(), query_log_path=log)
    rag.answer(QueryRequest(question="How quickly must complaints be resolved?"))
    import json
    entry = json.loads(log.read_text().splitlines()[0])
    assert entry["status"] == "answered" and entry["retrieved"] and "total_ms" in entry["latency_ms"]
