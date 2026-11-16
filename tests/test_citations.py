from app.generation.citations import (
    best_excerpt,
    extract_markers,
    grounding_metrics,
    resolve_citations,
)
from app.schemas.models import Chunk, RetrievedChunk


def _rc(i: int, text: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=f"d{i}:p{page}:c{i}", doc_id=f"d{i}", title=f"Directions {i}", source_filename=f"f{i}.pdf",
                    page_number=page, circular_number=f"RBI/2099-00/{i}", publication_date="2099-01-01",
                    section="Chapter II", chunk_index=i, text=text, source_url=f"https://example.org/{i}.pdf"),
        semantic_score=0.8, rerank_score=3.0,
    )


RETRIEVED = [
    _rc(1, "A bank shall resolve every complaint within 30 days of receipt.", 4),
    _rc(2, "The Principal Nodal Officer shall be not below the rank of General Manager.", 7),
]


def test_marker_extraction_handles_groups_and_order():
    assert extract_markers("x [S2] y [S1, S3] z [S2][S4]") == ["S2", "S1", "S3", "S4"]


def test_citations_use_chunk_metadata_not_llm_text():
    answer = "Complaints must be resolved within 30 days [S1]. The nodal officer is a General Manager or above [S2]."
    cleaned, citations, invalid = resolve_citations(answer, RETRIEVED)
    assert invalid == [] and cleaned == answer
    assert [c.source_id for c in citations] == ["S1", "S2"]
    c1 = citations[0]
    assert (c1.title, c1.circular_number, c1.page_number, c1.chunk_id) == ("Directions 1", "RBI/2099-00/1", 4, "d1:p4:c1")
    assert c1.excerpt in RETRIEVED[0].chunk.text
    assert citations[1].page_number == 7


def test_hallucinated_markers_are_removed_and_reported():
    answer = "Complaints must be resolved within 30 days [S1][S9]. Another claim [S7]."
    cleaned, citations, invalid = resolve_citations(answer, RETRIEVED)
    assert invalid == ["S9", "S7"]
    assert "[S9]" not in cleaned and "[S7]" not in cleaned and "[S1]" in cleaned
    assert [c.source_id for c in citations] == ["S1"]


def test_uncited_retrieved_chunks_are_not_citations():
    _, citations, _ = resolve_citations("Only the first source matters [S1].", RETRIEVED)
    assert [c.source_id for c in citations] == ["S1"]


def test_best_excerpt_picks_supporting_sentences():
    filler = " ".join(f"Unrelated sentence {i} about branch timings." for i in range(30))
    text = f"{filler} The bank shall report every cyber incident within 6 hours of detection. {filler}"
    excerpt = best_excerpt(text, ["cyber incidents must be reported within 6 hours"], max_chars=300)
    assert "within 6 hours of detection" in excerpt
    assert len(excerpt) <= 300 + 4


def test_grounding_metrics():
    answer = "Regulatory requirements:\n- Complaints must be resolved within 30 days [S1].\n- Banks should also train staff on soft skills."
    cleaned, citations, invalid = resolve_citations(answer, RETRIEVED)
    g = grounding_metrics(cleaned, citations, RETRIEVED, invalid)
    assert g.sentences_total == 2 and g.sentences_with_citation == 1
    assert g.citation_coverage == 0.5
    assert 0 < g.lexical_support < 1


def test_marker_after_full_stop_counts_as_citation():
    answer = "- A bank shall resolve every complaint within 30 days of receipt. [S1]"
    cleaned, citations, invalid = resolve_citations(answer, RETRIEVED)
    g = grounding_metrics(cleaned, citations, RETRIEVED, invalid)
    assert g.citation_coverage == 1.0 and g.lexical_support == 1.0
