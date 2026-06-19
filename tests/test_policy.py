import pytest

from src.agents.policy_agent import policy_node
from src.graph.state import CreditRiskOutput, LoanApplicationState


def test_policy_checks_passed():
    """Tests policy validation passes for a clean application."""
    state = LoanApplicationState(
        application_id="APP-001",
        applicant_data={
            "loan_amount": 200000,
            "property_value": 280000,
            "employment_months": 24,
        },
        credit_output=CreditRiskOutput(
            credit_score=750,
            risk_category="low",
            dti_ratio=0.25,
            default_probability=0.05,
            confidence_score=0.91,
            reasoning="Low risk profile",
        ),
    )

    result = policy_node(state)
    policy_out = result["policy_output"]

    # LTV = 200k / 280k = 71.4% (well below 80% limit)
    # Credit score 750 (above 650 limit)
    # DTI 25% (below 45% limit)
    # Employment 24 months (above 12 months limit)
    assert policy_out.policy_passed is True
    assert len(policy_out.violations) == 0
    assert policy_out.ltv_ratio == pytest.approx(0.7143, abs=0.0001)


def test_policy_checks_failed_dti_and_employment():
    """Tests policy validation catches DTI and employment duration violations."""
    state = LoanApplicationState(
        application_id="APP-002",
        applicant_data={
            "loan_amount": 250000,
            "property_value": 270000,
            "employment_months": 8,  # Violation: < 12 months
        },
        credit_output=CreditRiskOutput(
            credit_score=580,
            risk_category="very_high",
            dti_ratio=0.55,  # Violation: > 45%
            default_probability=0.80,
            confidence_score=0.62,
            reasoning="High risk profile",
        ),
    )

    result = policy_node(state)
    policy_out = result["policy_output"]

    assert policy_out.policy_passed is False
    # Verify at least the DTI and employment violations are captured
    assert any("DTI" in v for v in policy_out.violations)
    assert any("Employment" in v for v in policy_out.violations)
