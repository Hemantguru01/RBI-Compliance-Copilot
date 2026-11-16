import pytest

from app.retrieval.keyword_search import KeywordIndex, tokenize
from app.retrieval.retriever import reciprocal_rank_fusion

RELEVANCE_CASES = [
    ("How quickly must a bank resolve a customer complaint?", "fixture_grievance_redress.pdf", 2),
    ("Who must an unresolved complaint be escalated to?", "fixture_grievance_redress.pdf", 2),
    ("What compensation is payable for delay in resolving a complaint?", "fixture_grievance_redress.pdf", 3),
    ("What is the cooling-off period for a digital loan?", "fixture_digital_lending.pdf", 2),
    ("Where must borrower data collected by lending apps be stored?", "fixture_digital_lending.pdf", 3),
    ("Within how many hours must a cyber security incident be reported?", "fixture_cyber_security.pdf", 1),
    ("How often must the IT Strategy Committee meet?", "fixture_cyber_security.pdf", 2),
]


@pytest.mark.parametrize("question,filename,page", RELEVANCE_CASES)
def test_top_result_is_the_relevant_document_and_page(indexed_container, question, filename, page):
    results = indexed_container.retriever.retrieve(question, top_k=3)
    assert results, "relevant question must retrieve evidence"
    top = results[0].chunk
    assert top.source_filename == filename
    assert top.page_number == page


@pytest.mark.parametrize("question", [
    "Who won the cricket world cup?",
    "What is the best recipe for chocolate cake?",
    "What is the capital gains tax rate on equity shares?",
])
def test_irrelevant_questions_retrieve_nothing(indexed_container, question):
    assert indexed_container.retriever.retrieve(question) == []


def test_results_carry_scores_and_metadata(indexed_container):
    results = indexed_container.retriever.retrieve("cooling-off period digital loan", top_k=5)
    assert results
    for r in results:
        assert r.semantic_score is not None and r.semantic_score >= indexed_container.retriever.config.similarity_threshold
        assert r.rerank_score is not None and r.rerank_score >= indexed_container.retriever.config.rerank_threshold
        assert r.chunk.title and r.chunk.page_number >= 1 and r.chunk.circular_number
    rerank = [r.rerank_score for r in results]
    assert rerank == sorted(rerank, reverse=True)


def test_top_k_is_respected(indexed_container):
    assert len(indexed_container.retriever.retrieve("bank shall", top_k=2)) <= 2


def test_document_filter(indexed_container):
    store = indexed_container.store
    cyber_id = next(d.doc_id for d in store.documents.values() if "cyber" in d.source_filename)
    results = indexed_container.retriever.retrieve("complaint resolution timeline", doc_ids=[cyber_id])
    assert all(r.chunk.doc_id == cyber_id for r in results)


def test_rrf_rewards_items_ranked_by_both_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 4, 1]])
    assert max(fused, key=fused.get) == 1  # ranks 1 and 3
    assert fused[3] > fused[2] and fused[3] > fused[4]


def test_keyword_index_matches_exact_tokens(indexed_container):
    kw = KeywordIndex()
    kw.build(indexed_container.store.chunks)
    hits = kw.search("Principal Nodal Officer", k=3)
    assert hits
    assert "Principal Nodal Officer" in indexed_container.store.chunks[hits[0][0]].text
    assert kw.search("zzzz qqqq", k=3) == []


def test_tokenizer_keeps_regulatory_tokens():
    assert tokenize("Para 2.1 of RBI/2025-26/12 — the SMA-2 threshold") == ["para", "2.1", "rbi/2025-26/12", "sma-2", "threshold"]
