# data/raw/ — source RBI PDFs

Put the official RBI PDFs here. They are **not committed** to git (see `.gitignore`).

## Option A — download the 24 curated Directions (run on your own machine)

```bash
python scripts/download_rbi_documents.py
```

This downloads the 24 documents listed in `data/sources.json` from
`rbidocs.rbi.org.in` and writes `manifest.json` here. Ingestion attaches
each document's official title and source URL from that manifest.
(The cloud environment this project was built in was blocked by RBI's
firewall, so the script has not been run successfully there.)

## Option B — copy your own PDFs

Copy any RBI circular, notification, master direction or guideline PDFs into
this folder. Titles, circular numbers and dates are read from the PDF text.

## Then index them

```bash
python scripts/ingest.py          # incremental
python scripts/ingest.py --rebuild  # from scratch
```

or `POST /ingest` / the "Re-scan data/raw/" button in the UI.

Notes:
* Scanned (image-only) PDFs have no text layer and are skipped. OCR is not enabled.
* Duplicate files (same bytes or same text) are skipped automatically.
* Deleting a PDF here and re-ingesting removes it from the index.
