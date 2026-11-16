"""Download the official RBI source PDFs listed in ``data/sources.json``.

Every document comes from ``rbidocs.rbi.org.in`` (the RBI's official document
host), linked from the RBI Master Directions page. The manifest is written to
``data/raw/manifest.json`` so ingestion can attach the original source URL to
every chunk (citations can then be verified against the original PDF).

Usage:
    python scripts/download_rbi_documents.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "sources.json"
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = RAW_DIR / "manifest.json"
ALLOWED_HOST = "rbidocs.rbi.org.in"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; rbi-compliance-copilot/1.0)"}


def slugify(title: str, max_len: int = 90) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    return slug[:max_len].rstrip("_")


def download(url: str, dest: Path, retries: int = 4) -> bytes:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            content = resp.content
            if not content.startswith(b"%PDF"):
                raise ValueError(f"Response from {url} is not a PDF")
            dest.write_bytes(content)
            return content
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                raise
            wait = 2**attempt
            print(f"  attempt {attempt} failed ({exc}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    failures = 0

    for i, doc in enumerate(sources["documents"], start=1):
        url = doc["url"]
        if ALLOWED_HOST not in url:
            print(f"[{i}] SKIP non-RBI host: {url}", file=sys.stderr)
            continue
        filename = f"{i:02d}_{slugify(doc['title'])}.pdf"
        dest = RAW_DIR / filename
        try:
            if dest.exists() and not args.force:
                content = dest.read_bytes()
                status = "exists"
            else:
                content = download(url, dest)
                status = "downloaded"
            manifest.append(
                {
                    "filename": filename,
                    "title": doc["title"],
                    "source_url": url,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                }
            )
            print(f"[{i:02d}] {status:10s} {filename} ({len(content) // 1024} KB)")
        except Exception as exc:  # noqa: BLE001 - report and continue with other files
            failures += 1
            print(f"[{i:02d}] FAILED {url}: {exc}", file=sys.stderr)

    MANIFEST.write_text(
        json.dumps({"source_page": sources["source"], "documents": manifest}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n{len(manifest)} documents available, {failures} failures. Manifest: {MANIFEST}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
