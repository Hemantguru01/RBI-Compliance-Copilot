from app.ingestion.metadata import (
    find_circular_number,
    find_publication_date,
    find_title,
    title_from_filename,
)


def test_finds_rbi_and_department_reference():
    text = "RBI/2025-26/123\nDOR.CRE.REC.No.45/21.04.048/2025-26\nNovember 28, 2025"
    assert find_circular_number(text) == "RBI/2025-26/123; DOR.CRE.REC.No.45/21.04.048/2025-26"


def test_finds_department_reference_alone():
    assert find_circular_number("Ref DoR.MCS.REC.No.39/01.01.001/2025-26 dated") == "DoR.MCS.REC.No.39/01.01.001/2025-26"


def test_no_reference_returns_none_rather_than_guessing():
    assert find_circular_number("This page mentions no reference number at all.") is None


def test_dates_in_both_formats_and_earliest_wins():
    assert find_publication_date("Issued on November 28, 2025 and amended later") == "2025-11-28"
    assert find_publication_date("dated 4th September 2020") == "2020-09-04"
    assert find_publication_date("July 1, 2026 ... amended 3 March 2027") == "2026-07-01"
    assert find_publication_date("February 30, 2025") is None
    assert find_publication_date("no date here") is None


def test_title_detection():
    text = "Reserve Bank of India (Commercial Banks – Know Your Customer) Directions, 2025\nChapter I"
    assert find_title(text) == "Reserve Bank of India (Commercial Banks – Know Your Customer) Directions, 2025"
    assert find_title("random text") is None
    assert title_from_filename("02_Reserve_Bank_of_India_KYC.pdf") == "Reserve Bank of India KYC"
