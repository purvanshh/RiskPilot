from src.agents.arbitrator_agent import arbitrator_node
from src.graph.state import CreditRiskOutput, LoanApplicationState, PolicyCheckOutput


def test_arbitrator_unanimous_approval():
    """Tests arbitrator returns APPROVE when all agents align favorably."""
    state = LoanApplicationState(
        application_id="APP-001",
        applicant_data={},
        kyc_output={
            "missing_critical_docs": False,
            "fraud_flag": False,
            "confidence": 1.0,
        },
        credit_output=CreditRiskOutput(
            credit_score=780,
            risk_category="low",
            dti_ratio=0.20,
            default_probability=0.02,
            confidence_score=0.92,
            reasoning="Excellent credit",
        ),
        policy_output=PolicyCheckOutput(
            policy_passed=True,
            violations=[],
            ltv_ratio=0.65,
            min_credit_requirement_met=True,
            max_dti_threshold=0.45,
            retrieved_policy_chunks=[],
            reasoning="All policies satisfied",
        ),
    )

    result = arbitrator_node(state)
    arb_out = result["arbitrator_output"]

    assert arb_out.recommendation == "approve"
    assert arb_out.agent_agreement == "unanimous"
    assert arb_out.confidence_score >= 0.90


def test_arbitrator_conflict_resolution():
    """Tests arbitrator catches conflicts (e.g. high credit score but policy violations)."""
    state = LoanApplicationState(
        application_id="APP-003",
        applicant_data={},
        kyc_output={
            "missing_critical_docs": False,
            "fraud_flag": False,
            "confidence": 1.0,
        },
        credit_output=CreditRiskOutput(
            credit_score=710,  # Good credit
            risk_category="low",
            dti_ratio=0.35,
            default_probability=0.08,
            confidence_score=0.88,
            reasoning="Good credit score",
        ),
        policy_output=PolicyCheckOutput(
            policy_passed=False,  # Policy violation (LTV 82% > 80%)
            violations=["LTV exceeds standard 80% limit."],
            ltv_ratio=0.82,
            min_credit_requirement_met=True,
            max_dti_threshold=0.45,
            retrieved_policy_chunks=[],
            reasoning="LTV policy violation",
        ),
    )

    result = arbitrator_node(state)
    arb_out = result["arbitrator_output"]

    assert arb_out.recommendation == "review_required"
    assert arb_out.agent_agreement == "conflict"
    assert any("LTV" in f or "Policy" in f for f in arb_out.risk_flags)
