"""Phase 8, Task 6 — Arbitrator dummy-input tests (Member D).

Feeds hand-built KYC / Credit / Policy outputs into the arbitrator and asserts the
expected recommendation, agreement level, risk flags, and the new confidence-weighted
vote. Runs in complete isolation from Members A/B/C — no parsing, scoring, or RAG needed.
"""

from src.agents.arbitrator_agent import arbitrator_node, compute_weighted_vote
from src.graph.state import CreditRiskOutput, LoanApplicationState, PolicyCheckOutput


# --- Helpers to build dummy agent outputs ------------------------------------


def make_kyc(confidence=0.9, fraud=False, missing=False):
    return {
        "missing_critical_docs": missing,
        "fraud_flag": fraud,
        "confidence": confidence,
        "status": "valid" if not (fraud or missing) else "needs_review",
    }


def make_credit(score=720, risk="low", confidence=0.85, dti=0.30):
    return CreditRiskOutput(
        credit_score=score,
        risk_category=risk,
        dti_ratio=dti,
        default_probability=round(1 - (score - 300) / 550, 4),
        confidence_score=confidence,
        reasoning="dummy credit output",
    )


def make_policy(passed=True, violations=None, ltv=0.70, chunks=None):
    return PolicyCheckOutput(
        policy_passed=passed,
        violations=violations or [],
        ltv_ratio=ltv,
        min_credit_requirement_met=passed,
        max_dti_threshold=0.45,
        retrieved_policy_chunks=chunks or [],
        reasoning="dummy policy output",
    )


def build_state(kyc, credit, policy, app_id="APP-DUMMY"):
    return LoanApplicationState(
        application_id=app_id,
        applicant_data={"name": "Test User"},
        kyc_output=kyc,
        credit_output=credit,
        policy_output=policy,
    )


def run(kyc, credit, policy):
    state = build_state(kyc, credit, policy)
    return arbitrator_node(state)["arbitrator_output"]


# --- compute_weighted_vote unit tests ----------------------------------------


def test_weighted_vote_all_positive_leans_approve():
    vote = compute_weighted_vote(
        kyc_confidence=0.9, kyc_score=1.0,
        credit_confidence=0.85, credit_score_lean=1.0,
        policy_confidence=0.9, policy_score_lean=1.0,
    )
    assert vote["weighted_score"] > 0.9
    assert vote["implied_recommendation"] == "approve"


def test_weighted_vote_all_negative_leans_deny():
    vote = compute_weighted_vote(
        kyc_confidence=0.9, kyc_score=-1.0,
        credit_confidence=0.85, credit_score_lean=-1.0,
        policy_confidence=0.9, policy_score_lean=-1.0,
    )
    assert vote["weighted_score"] < -0.9
    assert vote["implied_recommendation"] == "deny"


def test_weighted_vote_stays_in_range():
    """Result must remain in [-1, 1] regardless of confidence magnitudes."""
    vote = compute_weighted_vote(0.3, 1.0, 0.3, 1.0, 0.3, -1.0)
    assert -1.0 <= vote["weighted_score"] <= 1.0


def test_weighted_vote_zero_confidence_is_neutral():
    vote = compute_weighted_vote(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    assert vote["weighted_score"] == 0.0
    assert vote["implied_recommendation"] == "review_required"


def test_weighted_vote_low_confidence_agent_has_less_say():
    """A high-confidence approve should outweigh a low-confidence deny."""
    vote = compute_weighted_vote(
        kyc_confidence=0.95, kyc_score=1.0,
        credit_confidence=0.95, credit_score_lean=1.0,
        policy_confidence=0.1, policy_score_lean=-1.0,
    )
    assert vote["weighted_score"] > 0
    assert vote["implied_recommendation"] == "approve"


# --- arbitrator_node end-to-end with dummy data ------------------------------


def test_clean_application_approves():
    arb = run(make_kyc(), make_credit(score=760, risk="low"), make_policy(passed=True))
    assert arb.recommendation == "approve"
    assert arb.agent_agreement == "unanimous"
    assert arb.risk_flags == []
    assert "Weighted vote score" in arb.summary


def test_fraud_flag_forces_review():
    arb = run(make_kyc(fraud=True), make_credit(), make_policy())
    assert arb.recommendation == "review_required"
    assert arb.agent_agreement == "conflict"
    assert any("Fraud" in f for f in arb.risk_flags)


def test_missing_docs_forces_review():
    arb = run(make_kyc(missing=True), make_credit(), make_policy())
    assert arb.recommendation == "review_required"
    assert any("Missing" in f for f in arb.risk_flags)


def test_very_high_risk_denies():
    arb = run(
        make_kyc(),
        make_credit(score=480, risk="very_high", confidence=0.8, dti=0.55),
        make_policy(passed=True),
    )
    assert arb.recommendation == "deny"
    assert any("VERY_HIGH" in f or "very_high" in f.lower() for f in arb.risk_flags)


def test_high_credit_with_policy_violation_is_conflict():
    arb = run(
        make_kyc(),
        make_credit(score=730, risk="low"),
        make_policy(passed=False, violations=["LTV exceeds 80% limit"], ltv=0.85),
    )
    assert arb.recommendation == "review_required"
    assert arb.agent_agreement == "conflict"
    assert any("Policy" in f or "LTV" in f for f in arb.risk_flags)


def test_borderline_credit_routes_to_review():
    arb = run(make_kyc(), make_credit(score=670, risk="medium"), make_policy(passed=True))
    assert arb.recommendation == "review_required"
    assert any("Borderline" in f for f in arb.risk_flags)


def test_summary_always_populated():
    arb = run(make_kyc(), make_credit(), make_policy())
    assert isinstance(arb.summary, str) and len(arb.summary) > 0
    assert "Arbitrator Recommendation" in arb.summary
