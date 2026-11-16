"""Ingest PDFs from RAW_DATA_DIR (default data/raw/) into the FAISS index.

Usage:
    python scripts/ingest.py            # incremental: only new/changed files
    python scripts/ingest.py --rebuild  # discard the index and rebuild
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.services.container import AppContainer  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="rebuild the index from scratch")
    args = parser.parse_args()
    settings = get_settings()
    setup_logging("WARNING")
    container = AppContainer(settings)
    container.startup()
    result = container.ingest(force=args.rebuild)
    print(json.dumps(result.model_dump(), indent=2))
    if not result.total_documents:
        print(f"\nNo documents indexed. Put PDFs in {settings.raw_data_dir} and run again.", file=sys.stderr)
        sys.exit(1)
