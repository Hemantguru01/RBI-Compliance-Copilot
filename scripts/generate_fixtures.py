"""Write the synthetic test PDFs (NOT RBI publications) to a directory.

Usage:
    python scripts/generate_fixtures.py [--out data/fixtures]

Then try the full pipeline on them without touching data/raw/:
    RAW_DATA_DIR=data/fixtures VECTOR_STORE_DIR=vector_store_fixtures \
    PROCESSED_DATA_DIR=data/processed_fixtures python scripts/ingest.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.fixtures import generate_fixture_pdfs  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/fixtures"))
    args = parser.parse_args()
    for path in generate_fixture_pdfs(args.out):
        print(f"wrote {path}")
