"""Run the retrieval / grounding evaluation against the current index.

Usage:
    python scripts/evaluate.py data/eval/rbi_questions.jsonl [--top-k 5] [--out logs/eval.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.evaluation.evaluate import evaluate, load_cases  # noqa: E402
from app.services.container import AppContainer  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gold", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    setup_logging("WARNING")
    container = AppContainer(get_settings())
    container.startup()
    if not container.index_ready:
        sys.exit("Index is empty. Run scripts/ingest.py first.")
    report = evaluate(container.rag, load_cases(args.gold), top_k=args.top_k)
    text = json.dumps(report.to_dict(), indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
