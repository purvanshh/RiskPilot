import logging
from typing import Any, Dict

from src.graph.state import CreditRiskOutput, LoanApplicationState, validate_state
from src.tools.credit_tools import calculate_credit_score, dti_calculator, risk_classifier

logger = logging.getLogger(__name__)


@validate_state
def credit_node(state: LoanApplicationState) -> Dict[str, Any]:
    """
    Credit Risk Agent Node
    Scores applicant creditworthiness using scoring functions and DTI calculations.
    """
    logger.info(f"Starting credit risk assessment for application {state.application_id}")

    error_log = list(state.error_log)
    try:
        # Retrieve values from applicant data or verified KYC data
        income = state.applicant_data.get("income", 50000)
        monthly_debt = state.applicant_data.get("monthly_debt", 1000)
        employment_months = state.applicant_data.get("employment_months", 12)

        # Override with KYC verified values if available
        if state.kyc_output and state.kyc_output.get("verified_fields"):
            verified_fields = state.kyc_output["verified_fields"]
            income = verified_fields.get("income") or income

        # Calculate DTI
        dti = dti_calculator(monthly_debt, income)

        # Calculate credit score
        credit_score = calculate_credit_score(income, monthly_debt, employment_months)

        # Map score to risk category
        risk_category = risk_classifier(credit_score)

        # Compute probability of default (simple mapping for boilerplate)
        default_prob = 1.0 - ((credit_score - 300) / 550.0)
        default_prob = min(1.0, max(0.0, default_prob))

        reasoning = (
            f"Calculated Credit Score: {credit_score} based on "
            f"annual income of ${income:.2f}, monthly debt of ${monthly_debt:.2f}, "
            f"and employment tenure of {employment_months} months. "
            f"DTI ratio is {dti:.2%}, "
            f"leading to a risk classification of {risk_category.upper()}."
        )

        credit_result = CreditRiskOutput(
            credit_score=credit_score,
            risk_category=risk_category,
            dti_ratio=round(dti, 4),
            default_probability=round(default_prob, 4),
            reasoning=reasoning,
        )

    except Exception as e:
        logger.error(f"Error in Credit Risk Agent: {str(e)}")
        error_log.append(f"Credit Agent error: {str(e)}")
        # Fallback values
        credit_result = CreditRiskOutput(
            credit_score=600,
            risk_category="high",
            dti_ratio=0.5,
            default_probability=0.5,
            reasoning=f"Error occurred: {str(e)}. Fallback credit score assigned.",
        )

    return {"credit_output": credit_result, "error_log": error_log}
