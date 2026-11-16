"""Ask a question from the command line (no API server needed).

Usage:
    python scripts/ask.py "What is the timeline for resolving complaints?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.schemas.models import QueryRequest  # noqa: E402
from app.services.container import AppContainer  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    setup_logging("WARNING")
    container = AppContainer(get_settings())
    container.startup()
    if not container.index_ready:
        sys.exit("Index is empty. Run scripts/ingest.py first.")
    r = container.rag.answer(QueryRequest(question=args.question, top_k=args.top_k))
    print(f"\n[{r.status}] provider={r.provider} model={r.model} confidence={r.confidence.level} ({r.confidence.score})\n")
    print(r.answer)
    print("\nCitations:")
    for c in r.citations:
        print(f"  [{c.source_id}] {c.title} | {c.circular_number or 'ref n/a'} | page {c.page_number}")
        print(f"       \"{c.excerpt[:240]}\"")
    if not r.citations:
        print("  (none)")
    print("\nRetrieved:")
    for c in r.retrieved:
        print(f"  {c.source_id} score={c.relevance_score} p.{c.page_number} {c.title[:80]}")
    print(f"\nLatency: {r.latency.model_dump()}")
