"""Synthetic PDF fixtures for tests and local smoke runs.

IMPORTANT: these documents are fabricated for software testing. They are NOT
RBI publications and their content is not regulation. Every page carries a
banner saying so, the titles start with "[SYNTHETIC TEST FIXTURE]", and the
reference numbers use the impossible year 2099 so they cannot be mistaken for
real circulars. Never place them in ``data/raw/`` alongside real documents.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

BANNER = "SYNTHETIC TEST FIXTURE - NOT AN RBI PUBLICATION - FOR SOFTWARE TESTING ONLY"

FIXTURE_DOCS: dict[str, dict] = {
    "fixture_grievance_redress.pdf": {
        "title": "[SYNTHETIC TEST FIXTURE] Test Bank (Customer Grievance Redress) Directions, 2099",
        "ref": "RBI/2099-00/901",
        "dept": "TEST.CGR.REC.No.01/99.01.001/2099-00",
        "date": "January 15, 2099",
        "pages": [
            [
                "Test Bank (Customer Grievance Redress) Directions, 2099",
                "Chapter I - Preliminary",
                "1. These directions apply to every test bank that accepts deposits from the public.",
                "2. Definitions. In these directions, a complaint means a representation in writing or through "
                "electronic means alleging deficiency in service on the part of the bank.",
            ],
            [
                "Chapter II - Grievance Redress Mechanism",
                "3. Every bank shall put in place a Board-approved grievance redress policy. The policy shall be "
                "displayed on the website of the bank and at every branch.",
                "4. Timelines. A bank shall acknowledge every complaint within 3 working days of receipt. "
                "A bank shall resolve every complaint within 30 days of receipt. Where a complaint is not resolved "
                "within 30 days, the bank shall escalate it to the Principal Nodal Officer.",
                "5. The Principal Nodal Officer shall be an officer not below the rank of General Manager.",
            ],
            [
                "Chapter III - Reporting",
                "6. Every bank shall submit a quarterly statement of complaints received, resolved and pending "
                "to its Board of Directors within 15 days of the end of each quarter.",
                "7. Compensation. Where a delay in resolving a complaint is attributable to the bank, the bank shall "
                "pay the complainant compensation of 100 rupees for each day of delay.",
            ],
        ],
    },
    "fixture_digital_lending.pdf": {
        "title": "[SYNTHETIC TEST FIXTURE] Test Bank (Digital Lending) Directions, 2099",
        "ref": "RBI/2099-00/902",
        "dept": "TEST.DL.REC.No.02/99.02.002/2099-00",
        "date": "March 2, 2099",
        "pages": [
            [
                "Test Bank (Digital Lending) Directions, 2099",
                "Chapter I - Scope",
                "1. These directions apply to loans sourced through digital lending applications operated by a "
                "test bank or by a lending service provider engaged by it.",
            ],
            [
                "Chapter II - Disclosure Requirements",
                "2. Key Fact Statement. Before the execution of a loan contract, the bank shall provide the borrower "
                "with a Key Fact Statement in a standardised format. The Key Fact Statement shall state the "
                "annual percentage rate, the recovery mechanism and the details of the grievance redress officer.",
                "3. Cooling-off period. The borrower shall be given a cooling-off period of not less than 3 days "
                "during which the borrower may exit the loan by paying the principal and the proportionate "
                "annual percentage rate without any penalty.",
            ],
            [
                "Chapter III - Data Protection",
                "4. Digital lending applications shall collect only data that is needed for the loan and shall "
                "obtain the explicit prior consent of the borrower. Data shall be stored only on servers located "
                "in India.",
                "5. Disbursal and repayment shall be made directly between the bank account of the borrower and the "
                "bank, without any pass-through account of a lending service provider.",
            ],
        ],
    },
    "fixture_cyber_security.pdf": {
        "title": "[SYNTHETIC TEST FIXTURE] Test Bank (Cyber Incident Reporting) Directions, 2099",
        "ref": "RBI/2099-00/903",
        "dept": "TEST.CSITE.REC.No.03/99.03.003/2099-00",
        "date": "June 30, 2099",
        "pages": [
            [
                "Test Bank (Cyber Incident Reporting) Directions, 2099",
                "Chapter I - Incident Reporting",
                "1. A bank shall report every cyber security incident to the supervisor within 6 hours of "
                "detection, using the incident reporting format.",
                "2. A bank shall maintain a Cyber Crisis Management Plan approved by its Board and shall test it "
                "at least once a year.",
            ],
            [
                "Chapter II - Governance",
                "3. The bank shall appoint a Chief Information Security Officer who shall report directly to the "
                "Executive Director in charge of risk management.",
                "4. The IT Strategy Committee of the Board shall meet at least once every quarter.",
            ],
        ],
    },
}


def _write_pdf(path: Path, title: str, page_lines: list[list[str]], header: list[str]) -> None:
    doc = pymupdf.open()
    for page_no, lines in enumerate(page_lines, start=1):
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text((40, 30), BANNER, fontsize=8)
        y = 60.0
        if page_no == 1:
            for h in header:
                page.insert_text((40, y), h, fontsize=10)
                y += 16
            y += 10
        body = "\n\n".join(lines)
        rect = pymupdf.Rect(40, y, 555, 800)
        page.insert_textbox(rect, body, fontsize=10.5, lineheight=1.3)
        page.insert_text((290, 825), str(page_no), fontsize=9)
    doc.set_metadata({"title": title, "subject": BANNER})
    doc.save(path)
    doc.close()


def generate_fixture_pdfs(directory: Path) -> list[Path]:
    """Write the synthetic fixture PDFs into ``directory`` and return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, spec in FIXTURE_DOCS.items():
        path = directory / filename
        header = [spec["ref"], spec["dept"], spec["date"], spec["title"]]
        _write_pdf(path, spec["title"], spec["pages"], header)
        paths.append(path)
    return paths


def write_blank_pdf(path: Path, pages: int = 2) -> Path:
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()
    doc.save(path)
    doc.close()
    return path


def write_corrupted_pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.7\n this is not really a pdf \x00\x01\x02 trailer garbage")
    return path
