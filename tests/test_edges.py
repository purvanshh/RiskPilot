"""Phase 9 — Routing edge tests (Member D).

Verifies the existing conditional routing functions in src/graph/edges.py with
dummy state. No agents are run; we set kyc_output directly and assert the route.
"""

from src.graph.edges import route_after_arbitrator, route_after_kyc
from src.graph.state import (
    ArbitratorOutput,
    CreditRiskOutput,
    LoanApplicationState,
    PolicyCheckOutput,
)


def make_state(kyc_output=None):
    return LoanApplicationState(
        application_id="APP-EDGE",
        applicant_data={"name": "Edge Test"},
        kyc_output=kyc_output,
    )


# --- route_after_kyc ---------------------------------------------------------


def test_route_after_kyc_missing_docs_goes_to_retry():
    state = make_state({"missing_critical_docs": True, "fraud_flag": False})
    assert route_after_kyc(state) == "retry"


def test_route_after_kyc_fraud_goes_to_human_review():
    state = make_state({"missing_critical_docs": False, "fraud_flag": True})
    assert route_after_kyc(state) == "human_review"


def test_route_after_kyc_clean_goes_to_credit():
    state = make_state({"missing_critical_docs": False, "fraud_flag": False})
    assert route_after_kyc(state) == "credit"


def test_route_after_kyc_no_output_defaults_to_human_review():
    """Missing KYC output should fail safe to human review, not crash."""
    state = make_state(kyc_output=None)
    assert route_after_kyc(state) == "human_review"


def test_route_after_kyc_missing_docs_takes_priority_over_fraud():
    """When both flags are set, the retry (missing docs) branch is checked first."""
    state = make_state({"missing_critical_docs": True, "fraud_flag": True})
    assert route_after_kyc(state) == "retry"


# --- route_after_arbitrator --------------------------------------------------


def _full_state(recommendation):
    return LoanApplicationState(
        application_id="APP-ARB",
        applicant_data={},
        kyc_output={"missing_critical_docs": False, "fraud_flag": False},
        credit_output=CreditRiskOutput(
            credit_score=700,
            risk_category="low",
            dti_ratio=0.3,
            default_probability=0.1,
            confidence_score=0.9,
            reasoning="x",
        ),
        policy_output=PolicyCheckOutput(
            policy_passed=True,
            violations=[],
            ltv_ratio=0.6,
            min_credit_requirement_met=True,
            max_dti_threshold=0.45,
            retrieved_policy_chunks=[],
            reasoning="x",
        ),
        arbitrator_output=ArbitratorOutput(
            recommendation=recommendation,
            confidence_score=0.9,
            agent_agreement="unanimous",
            summary="x",
            risk_flags=[],
        ),
    )


def test_route_after_arbitrator_always_human_review_on_approve():
    assert route_after_arbitrator(_full_state("approve")) == "human_review"


def test_route_after_arbitrator_always_human_review_on_deny():
    assert route_after_arbitrator(_full_state("deny")) == "human_review"


def test_route_after_arbitrator_always_human_review_on_review_required():
    assert route_after_arbitrator(_full_state("review_required")) == "human_review"
