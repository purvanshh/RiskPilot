"""
test_fraud_flag.py — Test Case 6: Fraud Flag Detection (Member B — WI-6)

PRD §9.1 Test Case 6 (Bonus):
  Input: Name mismatch between ID document and pay slip
  Expected: KYC agent flags fraud → Immediate human_review
             Credit and Policy agents are BYPASSED
             Final status: under_review (awaiting officer decision)

Tests in this file:
  1. Routing logic: fraud_flag=True → route_after_kyc returns "human_review"
  2. Routing logic: missing_critical_docs=True → route_after_kyc returns "retry"
  3. Routing logic: normal KYC → route_after_kyc returns "credit"
  4. State fixture: APP-006 has expected fraud_flag in kyc_output
  5. Credit node is NOT reached when fraud is flagged (mocked graph)
  6. Audit logger captures fraud flag event
  7. Output guardrail with fraud-flagged low-confidence arbitrator
  8. Full JSON test data has APP-006 with fraud_expected=True
"""

import json
from pathlib import Path

from src.graph.edges import route_after_kyc
from src.graph.state import ArbitratorOutput, LoanApplicationState
from src.guardrails.audit_logger import AuditLogger
from src.guardrails.output_validation import validate_system_recommendation

# ---------------------------------------------------------------------------
# Test 1: route_after_kyc routes fraud_flag=True to human_review
# ---------------------------------------------------------------------------


def test_fraud_flag_routes_to_human_review(state_fraud_flag):
    """
    Core routing test: when kyc_output.fraud_flag is True,
    route_after_kyc must return 'human_review' — bypassing credit + policy.
    """
    # Confirm fraud flag is set in the fixture
    assert state_fraud_flag.kyc_output.get("fraud_flag") is True

    route = route_after_kyc(state_fraud_flag)
    assert route == "human_review", f"Expected 'human_review' for fraud flag, got '{route}'"


# ---------------------------------------------------------------------------
# Test 2: route_after_kyc routes missing_critical_docs=True to retry
# ---------------------------------------------------------------------------


def test_missing_docs_routes_to_retry(state_fraud_flag, mock_kyc_output_missing_docs):
    """
    Missing critical docs should route to 'retry', not 'human_review'.
    Verifies the routing priority is: missing_docs → retry BEFORE fraud → human_review.
    """
    state = LoanApplicationState(
        application_id="APP-004",
        applicant_data={
            "name": "Diana Prince",
            "income": 90000,
            "monthly_debt": 1500,
            "loan_amount": 300000,
            "property_value": 400000,
            "employment_months": 48,
        },
        kyc_output=mock_kyc_output_missing_docs,
    )

    route = route_after_kyc(state)
    assert route == "retry", f"Expected 'retry' for missing docs, got '{route}'"


# ---------------------------------------------------------------------------
# Test 3: route_after_kyc routes normal KYC to credit
# ---------------------------------------------------------------------------


def test_normal_kyc_routes_to_credit(mock_kyc_output):
    """Normal KYC (no fraud, no missing docs) should route to 'credit'."""
    state = LoanApplicationState(
        application_id="APP-001",
        applicant_data={
            "name": "Alice Johnson",
            "income": 80000,
            "monthly_debt": 1200,
            "loan_amount": 200000,
            "property_value": 280000,
            "employment_months": 36,
        },
        kyc_output=mock_kyc_output,
    )

    route = route_after_kyc(state)
    assert route == "credit", f"Expected 'credit' for valid KYC, got '{route}'"


# ---------------------------------------------------------------------------
# Test 4: route_after_kyc with no KYC output routes to human_review (safety)
# ---------------------------------------------------------------------------


def test_no_kyc_output_routes_to_human_review_for_safety():
    """If kyc_output is None (node failed), route_after_kyc must fail-safe to human_review."""
    state = LoanApplicationState(
        application_id="APP-TEST-NULL",
        applicant_data={"income": 50000},
        kyc_output=None,
    )

    route = route_after_kyc(state)
    assert route == "human_review", "Missing kyc_output should default to human_review for safety"


# ---------------------------------------------------------------------------
# Test 5: state_fraud_flag fixture has correct APP-006 structure
# ---------------------------------------------------------------------------


def test_fraud_flag_state_fixture_structure(state_fraud_flag):
    """Verify the conftest APP-006 fraud state fixture has correct structure."""
    state = state_fraud_flag

    assert state.application_id == "APP-006"
    assert state.kyc_output is not None
    assert state.kyc_output.get("fraud_flag") is True
    assert state.kyc_output.get("kyc_status") == "fraud_detected"
    assert state.kyc_output.get("confidence", 1.0) < 0.5  # low confidence

    # Documents: should have id_proof (valid) and pay_slip (invalid — mismatch)
    assert len(state.documents) >= 2
    id_doc = next((d for d in state.documents if d.document_type == "id_proof"), None)
    pay_slip = next((d for d in state.documents if d.document_type == "pay_slip"), None)

    assert id_doc is not None
    assert pay_slip is not None
    assert pay_slip.validation_status == "invalid"
    assert pay_slip.confidence < 0.5


# ---------------------------------------------------------------------------
# Test 6: Credit node is never called when fraud is flagged
# ---------------------------------------------------------------------------


def test_credit_node_not_called_on_fraud(state_fraud_flag):
    """
    When fraud_flag=True, routing goes to human_review before credit node.
    This test verifies the credit node would be bypassed in graph execution
    by asserting route_after_kyc never returns 'credit' for a fraud state.
    """
    route = route_after_kyc(state_fraud_flag)
    assert route != "credit", "Credit node must NOT be reached when fraud is flagged"

    # Also verify the credit_output remains None (credit never ran)
    assert (
        state_fraud_flag.credit_output is None
    ), "credit_output must be None when fraud detection routes to human_review"


# ---------------------------------------------------------------------------
# Test 7: Audit logger captures fraud flag event
# ---------------------------------------------------------------------------


def test_audit_logger_records_fraud_flag(tmp_path, monkeypatch):
    """
    AuditLogger.log_guardrail_flag() should write a fraud-related entry
    to the audit log file.
    """
    log_path = tmp_path / "test_audit.jsonl"
    monkeypatch.setenv("RISKPILOT_AUDIT_LOG", str(log_path))

    audit = AuditLogger(application_id="APP-006", trace_id="trace-test-001")
    audit.log_guardrail_flag(
        "output",
        "Fraud detected: Name mismatch between ID ('Frank Forger') and "
        "pay slip ('Francis Forgett').",
    )
    audit.log_decision(
        decision="under_review",
        flags=["Fraud detected: name mismatch"],
        officer_id=None,
    )

    assert log_path.exists(), "Audit log file should be created"

    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    import json

    fraud_entry = json.loads(lines[0])
    decision_entry = json.loads(lines[1])

    assert fraud_entry["event_type"] == "guardrail_flag"
    assert fraud_entry["application_id"] == "APP-006"
    assert "Fraud" in fraud_entry["message"]

    assert decision_entry["event_type"] == "decision"
    assert decision_entry["decision"] == "under_review"


# ---------------------------------------------------------------------------
# Test 8: Output guardrail fires for fraud-derived low-confidence arbitrator
# ---------------------------------------------------------------------------


def test_output_guardrail_on_fraud_derived_arbitrator_output():
    """
    After fraud detection, if arbitrator is invoked (hypothetical), its
    confidence should be very low (<0.6) and the guardrail should fire.
    """
    fraud_arbitrator = ArbitratorOutput(
        recommendation="review_required",
        confidence_score=0.20,  # very low — fraud case
        agent_agreement="conflict",
        summary="Fraud flag detected by KYC. All other checks inconclusive.",
        risk_flags=["Fraud flag: name mismatch", "KYC confidence: 0.30"],
    )

    requires_review, flags = validate_system_recommendation(
        fraud_arbitrator,
        application_id="APP-006",
    )

    assert requires_review is True
    assert any("confidence" in f.lower() for f in flags)
    assert any("conflict" in f.lower() or "disagreement" in f.lower() for f in flags)
    assert any("HITL" in f or "Human-in-the-Loop" in f for f in flags)


# ---------------------------------------------------------------------------
# Test 9: Test applications JSON contains APP-006 with fraud_expected=True
# ---------------------------------------------------------------------------


def test_test_applications_json_contains_app006():
    """
    Verify test_applications.json has been updated with APP-006 Fraud Flag test case.
    """
    json_path = Path(__file__).parent.parent / "data" / "test_applications.json"
    assert json_path.exists(), f"test_applications.json not found at {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        applications = json.load(f)

    app006 = next((a for a in applications if a["application_id"] == "APP-006"), None)

    assert app006 is not None, "APP-006 (Fraud Flag) must be in test_applications.json"
    assert app006.get("fraud_expected") is True, "APP-006 must have fraud_expected=True"
    assert app006["expected_recommendation"] == "review_required"

    # Validate document structure — should have at least 3 docs
    assert len(app006["documents"]) >= 3

    # At least one document should be invalid (the mismatched pay slip)
    invalid_docs = [d for d in app006["documents"] if d["validation_status"] == "invalid"]
    assert len(invalid_docs) >= 1, "APP-006 must have at least one invalid document"


# ---------------------------------------------------------------------------
# Test 10: Fraud name mismatch detection logic (unit test on the signal)
# ---------------------------------------------------------------------------


def test_fraud_name_mismatch_signal():
    """
    Verifies that a name mismatch across documents produces
    the expected fraud signal pattern used by the KYC agent.
    """
    # This tests the data pattern that KYC agents detect
    id_name = "Frank Forger"
    pay_slip_name = "Francis Forgett"

    # Names are clearly different — simulate what KYC fraud detection checks
    names_match = id_name.lower().strip() == pay_slip_name.lower().strip()
    assert names_match is False, "Names should NOT match (this IS the fraud case)"

    # Verify both names appear in the APP-006 documents
    json_path = Path(__file__).parent.parent / "data" / "test_applications.json"
    with open(json_path, "r", encoding="utf-8") as f:
        applications = json.load(f)

    app006 = next(a for a in applications if a["application_id"] == "APP-006")

    # ID proof name
    id_doc = next(d for d in app006["documents"] if d["document_type"] == "id_proof")
    slip_doc = next(d for d in app006["documents"] if d["document_type"] == "pay_slip")

    assert (
        id_doc["extracted_fields"]["name"] != slip_doc["extracted_fields"]["name"]
    ), "ID and pay slip names must differ to constitute a fraud signal"
