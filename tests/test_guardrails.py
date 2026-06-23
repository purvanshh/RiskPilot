import json

from src.graph.state import ArbitratorOutput
from src.guardrails.input_validation import validate_application_input
from src.guardrails.output_validation import validate_system_recommendation


def test_input_guardrail_missing_data():
    """Tests input guardrail catches missing required field elements."""
    app_data = {
        "name": "John Doe",
        # Missing income, loan amount, etc.
    }
    docs = [
        {"document_type": "id_proof"},
        {"document_type": "pay_slip"},
        {"document_type": "bank_statement"},
    ]

    is_valid, errors = validate_application_input(app_data, docs)
    assert is_valid is False
    assert any("income" in e for e in errors)


def test_input_guardrail_insufficient_docs():
    """Tests input guardrail catches insufficient document count."""
    app_data = {
        "name": "John Doe",
        "income": 50000,
        "monthly_debt": 1000,
        "loan_amount": 100000,
        "property_value": 150000,
        "employment_months": 12,
    }
    # Only 2 documents provided
    docs = [{"document_type": "id_proof"}, {"document_type": "pay_slip"}]

    is_valid, errors = validate_application_input(app_data, docs)
    assert is_valid is False
    assert any("documents" in e or "KYC" in e for e in errors)


def test_input_guardrail_invalid_file_type():
    """Tests input guardrail catches invalid document formats (e.g. .txt)."""
    app_data = {
        "name": "John Doe",
        "income": 50000,
        "monthly_debt": 1000,
        "loan_amount": 100000,
        "property_value": 150000,
        "employment_months": 12,
    }
    docs = [
        {"document_type": "id_proof", "filename": "id.pdf"},
        {
            "document_type": "pay_slip",
            "filename": "pay.txt",
        },  # TXT is not allowed under 8.1
        {"document_type": "bank_statement", "filename": "bank.png"},
    ]

    is_valid, errors = validate_application_input(app_data, docs)
    assert is_valid is False
    assert any("Invalid file type" in e and "pay.txt" in e for e in errors)


def test_input_guardrail_file_size_exceeded(tmp_path):
    """Tests input guardrail catches documents exceeding the size limit."""
    app_data = {
        "name": "John Doe",
        "income": 50000,
        "monthly_debt": 1000,
        "loan_amount": 100000,
        "property_value": 150000,
        "employment_months": 12,
    }

    # Test via size metadata key
    docs_meta = [
        {
            "document_type": "id_proof",
            "filename": "id.pdf",
            "file_size": 10 * 1024 * 1024 + 10,
        },
        {"document_type": "pay_slip", "filename": "pay.jpg"},
        {"document_type": "bank_statement", "filename": "bank.png"},
    ]
    is_valid, errors = validate_application_input(app_data, docs_meta)
    assert is_valid is False
    assert any("exceeds maximum size limit" in e for e in errors)

    # Test via file on disk size check
    large_file = tmp_path / "large_statement.pdf"
    with open(large_file, "wb") as f:
        f.write(b"\0" * (10 * 1024 * 1024 + 100))

    docs_disk = [
        {"document_type": "id_proof", "filename": "id.pdf"},
        {"document_type": "pay_slip", "filename": "pay.jpg"},
        {"document_type": "bank_statement", "filename": str(large_file)},
    ]
    is_valid_disk, errors_disk = validate_application_input(app_data, docs_disk)
    assert is_valid_disk is False
    assert any("exceeds maximum size limit" in e for e in errors_disk)


def test_input_guardrail_invalid_data_values():
    """Tests input guardrail catches invalid/negative field values."""
    app_data = {
        "name": "John Doe",
        "income": -50000,  # Negative income should fail validation
        "monthly_debt": 1000,
        "loan_amount": 100000,
        "property_value": 150000,
        "employment_months": -5,  # Negative months should fail validation
    }
    docs = [
        {"document_type": "id_proof", "filename": "id.pdf"},
        {"document_type": "pay_slip", "filename": "pay.jpg"},
        {"document_type": "bank_statement", "filename": "bank.png"},
    ]

    is_valid, errors = validate_application_input(app_data, docs)
    assert is_valid is False
    assert any("income" in e for e in errors)
    assert any("employment_months" in e for e in errors)


def test_input_guardrail_audit_logging(tmp_path, monkeypatch):
    """Tests input guardrail logs violations to the audit log if application_id is provided."""
    log_path = tmp_path / "test_input_audit.jsonl"
    monkeypatch.setenv("RISKPILOT_AUDIT_LOG", str(log_path))

    # Invalid application data (missing income, negative months)
    app_data = {
        "name": "John Doe",
        "monthly_debt": 1000,
        "loan_amount": 100000,
        "property_value": 150000,
        "employment_months": -5,
    }
    # Only 2 documents
    docs = [
        {"document_type": "id_proof", "filename": "id.pdf"},
        {"document_type": "pay_slip", "filename": "pay.jpg"},
    ]

    is_valid, errors = validate_application_input(app_data, docs, application_id="APP-ERR-TEST")
    assert is_valid is False
    assert log_path.exists(), "Audit log should have been created"

    # Verify that all errors are recorded in the log
    with open(log_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    assert len(entries) >= len(errors)
    for entry in entries:
        assert entry["application_id"] == "APP-ERR-TEST"
        assert entry["event_type"] == "guardrail_flag"
        assert entry["guardrail_type"] == "input"
        assert any(entry["message"] == err for err in errors)


def test_output_guardrail_low_confidence():
    """Tests output guardrail triggers review on low confidence recommendation."""
    arb_out = ArbitratorOutput(
        recommendation="approve",
        confidence_score=0.55,  # Low confidence (< 0.60)
        agent_agreement="partial",
        summary="Approve with low confidence",
        risk_flags=["Borderline credit score"],
    )

    requires_review, flags = validate_system_recommendation(arb_out)
    assert requires_review is True
    assert any("Confidence" in f or "confidence" in f.lower() for f in flags)
