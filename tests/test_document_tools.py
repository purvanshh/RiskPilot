import os

import pytest

from src.tools.document_tools import (
    detect_document_type,
    extract_fields,
    parse_document,
    validate_fields,
)


def test_parse_real_pdf():
    """Verify that PyPDF2 extracts text correctly from a real synthetic PDF."""
    pdf_path = "data/synthetic_docs/APP-001-id.pdf"
    assert os.path.exists(pdf_path), "Synthetic PDF must exist for the test."

    text = parse_document(pdf_path)
    assert "Alice Johnson" in text
    assert "ID Proof" in text


def test_parse_document_size_limit(tmp_path):
    """Verify that parse_document rejects files exceeding the 10MB limit."""
    large_file = tmp_path / "large_file.pdf"

    # Write exactly 10.1 MB
    with open(large_file, "wb") as f:
        f.write(b"\0" * (10 * 1024 * 1024 + 1024))

    with pytest.raises(ValueError, match="exceeds maximum size limit"):
        parse_document(str(large_file))


def test_parse_document_invalid_extension(tmp_path):
    """Verify that parse_document rejects files with invalid extensions."""
    invalid_file = tmp_path / "test.exe"
    invalid_file.touch()

    with pytest.raises(ValueError, match="Invalid file type"):
        parse_document(str(invalid_file))


def test_detect_document_type():
    """Verify document type detection from filename and content heuristics."""
    assert detect_document_type("alice_id_proof.pdf", "") == "id_proof"
    assert detect_document_type("monthly_statement.png", "") == "bank_statement"
    assert detect_document_type("employment_confirmation.pdf", "") == "employment_letter"
    assert detect_document_type("pay_slip_may.txt", "") == "pay_slip"

    # Fallback to content heuristics
    assert detect_document_type("document.pdf", "Monthly deposit balance") == "bank_statement"
    assert detect_document_type("document.png", "gross pay period") == "pay_slip"


def test_extract_fields_fallback():
    """Verify regex-based field extraction for synthetic documents."""
    from src.tools.document_tools import extract_fields_fallback

    text_id = "Identity Document\nName: Alice Johnson\nDOB: 12/10/1990\nID Number: ID-1234"
    res_id = extract_fields_fallback(text_id, "id_proof")
    assert res_id["extracted_fields"]["name"] == "Alice Johnson"
    assert res_id["extracted_fields"]["dob"] == "12/10/1990"
    assert res_id["confidence"] == 0.90

    text_statement = "Monthly deposit: $6,666. Balance: $12,500. Monthly debt: $1,200."
    res_statement = extract_fields_fallback(text_statement, "bank_statement")
    assert res_statement["extracted_fields"]["income_monthly"] == 6666
    assert res_statement["extracted_fields"]["monthly_debt"] == 1200

    text_employment = (
        "Employment confirmation: Evan has been employed at DesignStudio for 11 months."
    )
    res_employment = extract_fields_fallback(text_employment, "employment_letter")
    assert res_employment["extracted_fields"]["employer"] == "DesignStudio"
    assert res_employment["extracted_fields"]["employment_months"] == 11


def test_extract_fields_general():
    """Verify general field extraction entrypoint is robust."""
    text_id = "Identity Document\nName: Alice Johnson\nDOB: 12/10/1990\nID Number: ID-1234"
    res_id = extract_fields(text_id, "id_proof")
    assert res_id["extracted_fields"]["name"] == "Alice Johnson"
    assert "dob" in res_id["extracted_fields"]
    assert res_id["confidence"] > 0.5


def test_validate_fields():
    """Verify complete fields validation checks."""
    assert validate_fields({"name": "Alice Johnson"}) is True
    assert validate_fields({"income_monthly": 5000}) is True
    assert validate_fields({"employer": "TechCorp"}) is False  # Missing name/income
