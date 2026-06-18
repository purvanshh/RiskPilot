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
