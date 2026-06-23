"""Phase 12 — Tests for the human_review_ui bridge tool (Member D)."""

import pytest

from src.graph.state import ArbitratorOutput, HumanDecision, LoanApplicationState
from src.tools.human_review_tool import human_review_ui, summarize_for_officer


def make_state():
    return LoanApplicationState(
        application_id="APP-HITL",
        applicant_data={"name": "HITL Test"},
        kyc_output={"missing_critical_docs": False, "fraud_flag": False, "confidence": 0.9},
        arbitrator_output=ArbitratorOutput(
            recommendation="review_required",
            confidence_score=0.7,
            agent_agreement="partial",
            summary="Borderline application.",
            risk_flags=["Borderline credit score (Tier 2)"],
        ),
    )


def test_programmatic_approve_returns_human_decision():
    update = human_review_ui(make_state(), decision="approve", officer_id="OFF-1")
    hd = update["human_decision"]
    assert isinstance(hd, HumanDecision)
    assert hd.decision == "approve"
    assert hd.officer_id == "OFF-1"
    assert hd.timestamp  # populated


def test_override_requires_reason():
    with pytest.raises(ValueError, match="override_reason is required"):
        human_review_ui(make_state(), decision="override_deny")


def test_override_with_reason_ok():
    update = human_review_ui(
        make_state(),
        decision="override_approve",
        override_reason="Strong compensating factors.",
    )
    assert update["human_decision"].override_reason == "Strong compensating factors."


def test_invalid_decision_rejected():
    with pytest.raises(ValueError, match="Invalid decision"):
        human_review_ui(make_state(), decision="maybe")


def test_accepts_dict_state():
    update = human_review_ui(make_state().to_dict(), decision="deny")
    assert update["human_decision"].decision == "deny"


def test_summary_contains_agent_sections():
    text = summarize_for_officer(make_state())
    assert "KYC" in text
    assert "Arbitrator" in text
    assert "APP-HITL" in text
