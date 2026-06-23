import time
from typing import Any, Dict

from src.graph.state import (
    LoanApplicationState,
    graceful_fallback,
    timeout_resilience,
    validate_state,
)


@validate_state
@graceful_fallback("kyc")
def dummy_kyc_error_node(state: LoanApplicationState) -> Dict[str, Any]:
    raise RuntimeError("Simulated KYC agent failure")


@validate_state
@graceful_fallback("credit")
def dummy_credit_error_node(state: LoanApplicationState) -> Dict[str, Any]:
    raise RuntimeError("Simulated Credit agent failure")


@validate_state
@graceful_fallback("policy")
def dummy_policy_error_node(state: LoanApplicationState) -> Dict[str, Any]:
    raise RuntimeError("Simulated Policy agent failure")


@validate_state
@graceful_fallback("arbitrator")
def dummy_arbitrator_error_node(state: LoanApplicationState) -> Dict[str, Any]:
    raise RuntimeError("Simulated Arbitrator agent failure")


@validate_state
@graceful_fallback("credit")
@timeout_resilience(0.05)
def dummy_timeout_node(state: LoanApplicationState) -> Dict[str, Any]:
    time.sleep(0.1)
    return {}


def test_kyc_graceful_fallback():
    """Verify that a failure in a KYC-type node is handled gracefully."""
    state = LoanApplicationState(
        application_id="APP-RES-001",
        applicant_data={"name": "Alice Johnson", "income": 75000},
        documents=[],
    )
    result = dummy_kyc_error_node(state)
    assert "kyc_output" in result
    kyc_out = result["kyc_output"]
    assert kyc_out["status"] == "failed"
    assert kyc_out["confidence"] == 0.0
    assert any("Simulated KYC agent failure" in err for err in result["error_log"])


def test_credit_graceful_fallback():
    """Verify that a failure in a Credit-type node is handled gracefully."""
    state = LoanApplicationState(
        application_id="APP-RES-002",
        applicant_data={"name": "Bob Smith", "income": 50000},
        documents=[],
    )
    result = dummy_credit_error_node(state)
    assert "credit_output" in result
    credit_out = result["credit_output"]
    assert credit_out.credit_score == 300
    assert credit_out.risk_category == "very_high"
    assert credit_out.confidence_score == 0.0
    assert any("Simulated Credit agent failure" in err for err in result["error_log"])


def test_policy_graceful_fallback():
    """Verify that a failure in a Policy-type node is handled gracefully."""
    state = LoanApplicationState(
        application_id="APP-RES-003",
        applicant_data={"name": "Charlie Brown"},
        documents=[],
    )
    result = dummy_policy_error_node(state)
    assert "policy_output" in result
    policy_out = result["policy_output"]
    assert policy_out.policy_passed is False
    assert len(policy_out.violations) == 1
    assert "Simulated Policy agent failure" in policy_out.violations[0]
    assert any("Simulated Policy agent failure" in err for err in result["error_log"])


def test_arbitrator_graceful_fallback():
    """Verify that a failure in an Arbitrator-type node is handled gracefully."""
    state = LoanApplicationState(
        application_id="APP-RES-004",
        applicant_data={"name": "Diana Prince"},
        documents=[],
    )
    result = dummy_arbitrator_error_node(state)
    assert "arbitrator_output" in result
    arb_out = result["arbitrator_output"]
    assert arb_out.recommendation == "review_required"
    assert arb_out.confidence_score == 0.0
    assert any("Simulated Arbitrator agent failure" in err for err in result["error_log"])


def test_timeout_resilience_trigger():
    """Verify that execution exceeding timeout limits triggers a timeout error and fallback."""
    state = LoanApplicationState(
        application_id="APP-RES-005",
        applicant_data={"name": "Evan Wright"},
        documents=[],
    )
    result = dummy_timeout_node(state)
    assert "credit_output" in result
    credit_out = result["credit_output"]
    assert credit_out.credit_score == 300
    assert credit_out.confidence_score == 0.0
    assert any("timed out after 0.05 seconds" in err for err in result["error_log"])
