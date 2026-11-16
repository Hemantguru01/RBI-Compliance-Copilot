from pathlib import Path

from app.evaluation.evaluate import evaluate, load_cases

GOLD = Path(__file__).resolve().parents[1] / "data" / "eval" / "fixture_questions.jsonl"


def test_fixture_evaluation_meets_quality_bar(indexed_container):
    report = evaluate(indexed_container.rag, load_cases(GOLD), top_k=5)
    assert report.in_scope == 8 and report.out_of_scope == 3
    assert report.hit_at_k == 1.0, report.failures
    assert report.page_hit_at_k == 1.0
    assert report.mrr >= 0.9
    assert report.refusal_accuracy == 1.0
    assert report.false_refusals == 0
    assert report.mean_citation_coverage == 1.0


def test_gold_file_parser_skips_comments(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text('// comment\n{"question": "q one?", "expected_source": "abc"}\n{"question": "q2?", "expected_source": null}\n')
    cases = load_cases(p)
    assert [c.expected_source for c in cases] == ["abc", None]
