import pytest

from src.agents.credit_agent import credit_node
from src.graph.state import LoanApplicationState


def test_credit_assessment_low_risk():
    """Tests credit scoring for a high-income, low-debt applicant."""
    state = LoanApplicationState(
        application_id="APP-001",
        applicant_data={
            "name": "Alice Johnson",
            "income": 80000,
            "monthly_debt": 1200,
            "employment_months": 36,
        },
    )

    result = credit_node(state)
    credit_out = result["credit_output"]

    # Assertions based on scoring formula
    # income / 12 = 6666.67
    # dti = 1200 / 6666.67 = 0.18
    # base_score = 300 + (80000/1000)*10 + 36*2 - 0.18*200 = 300+800+72-36 = 1136 -> capped at 850
    assert credit_out.credit_score == 850
    assert credit_out.risk_category == "low"
    assert credit_out.dti_ratio == pytest.approx(0.18, abs=0.01)


def test_credit_assessment_high_risk():
    """Tests credit scoring for a low-income, high-debt applicant."""
    state = LoanApplicationState(
        application_id="APP-002",
        applicant_data={
            "name": "Bob Smith",
            "income": 30000,
            "monthly_debt": 1500,
            "employment_months": 6,
        },
    )

    result = credit_node(state)
    credit_out = result["credit_output"]

    # income / 12 = 2500
    # dti = 1500 / 2500 = 0.60
    # base_score = 300 + 30 * 10 + 6 * 2 - 0.60 * 200 = 300 + 300 + 12 - 120 = 492
    assert credit_out.credit_score == 492
    assert credit_out.risk_category == "very_high"
    assert credit_out.dti_ratio == 0.60
