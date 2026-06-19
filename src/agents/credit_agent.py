import logging
from typing import Any, Dict

from src.graph.state import CreditRiskOutput, LoanApplicationState, validate_state
from src.tools.credit_tools import (
    calculate_confidence_score,
    calculate_credit_score,
    calculate_default_probability,
    dti_calculator,
    risk_classifier,
)

logger = logging.getLogger(__name__)


@validate_state
def credit_node(state: LoanApplicationState) -> Dict[str, Any]:
    """
    Credit Risk Agent Node.

    Responsibilities:
    - Calculate DTI ratio from applicant financial data
    - Compute credit score using PRD deterministic formula
    - Classify risk category (low / medium / high / very_high)
    - Estimate default probability
    - Produce a confidence score reflecting input data quality
    - All outputs stored in CreditRiskOutput for downstream agents
    """
    logger.info(f"[CreditAgent] Starting credit risk assessment for application {state.application_id}")

    error_log = list(state.error_log)
    try:
        # --- Step 1: Resolve income, debt, and tenure ---
        # Primary source: applicant_data (pre-validated by KYC input guardrails)
        income = state.applicant_data.get("income", 0)
        monthly_debt = state.applicant_data.get("monthly_debt", 0)
        employment_months = state.applicant_data.get("employment_months", 0)

        logger.debug(
            f"[CreditAgent] Raw inputs — income={income}, "
            f"monthly_debt={monthly_debt}, employment_months={employment_months}"
        )

        # Override with KYC-verified values if available (higher trust)
        if state.kyc_output and state.kyc_output.get("verified_fields"):
            verified_fields = state.kyc_output["verified_fields"]
            if verified_fields.get("income"):
                income = verified_fields["income"]
                logger.info(f"[CreditAgent] Income overridden by KYC verified value: {income}")

        # --- Step 2: Calculate DTI ---
        dti = dti_calculator(monthly_debt, income)
        logger.info(f"[CreditAgent] DTI ratio: {dti:.4f} ({dti:.2%})")

        # --- Step 3: Calculate credit score (PRD formula) ---
        credit_score = calculate_credit_score(income, monthly_debt, employment_months)
        logger.info(f"[CreditAgent] Computed credit score: {credit_score}")

        # --- Step 4: Classify risk category ---
        risk_category = risk_classifier(credit_score)
        logger.info(f"[CreditAgent] Risk category: {risk_category}")

        # --- Step 5: Default probability ---
        default_prob = calculate_default_probability(credit_score)
        logger.info(f"[CreditAgent] Default probability: {default_prob:.4f}")

        # --- Step 6: Confidence score (data quality based) ---
        confidence = calculate_confidence_score(income, monthly_debt, employment_months)
        if confidence < 0.6:
            logger.warning(
                f"[CreditAgent] Low confidence score {confidence:.2f} — "
                "application may require human review."
            )

        # --- Step 7: Build reasoning narrative ---
        monthly_income = max(1.0, income / 12.0)
        reasoning = (
            f"Credit Score: {credit_score} (PRD formula). "
            f"Annual income: ${income:,.2f} (monthly: ${monthly_income:,.2f}). "
            f"Monthly debt obligations: ${monthly_debt:,.2f}. "
            f"DTI ratio: {dti:.2%} — "
            f"{'EXCEEDS 60% hard stop threshold.' if dti > 0.6 else 'within acceptable range.'} "
            f"Employment tenure: {employment_months} months "
            f"({'stable ≥12 months' if employment_months >= 12 else 'short <12 months, risk factor'}). "
            f"Risk classification: {risk_category.upper()}. "
            f"Default probability: {default_prob:.2%}. "
            f"Assessment confidence: {confidence:.2f}."
        )

        credit_result = CreditRiskOutput(
            credit_score=credit_score,
            risk_category=risk_category,
            dti_ratio=round(dti, 4),
            default_probability=default_prob,
            confidence_score=confidence,
            reasoning=reasoning,
        )

        logger.info(
            f"[CreditAgent] Assessment complete — "
            f"score={credit_score}, risk={risk_category}, confidence={confidence:.2f}"
        )

    except Exception as e:
        logger.error(f"[CreditAgent] Error during credit assessment: {str(e)}", exc_info=True)
        error_log.append(f"Credit Agent error: {str(e)}")
        # Fallback: conservative values that force human review
        credit_result = CreditRiskOutput(
            credit_score=300,
            risk_category="very_high",
            dti_ratio=1.0,
            default_probability=1.0,
            confidence_score=0.0,
            reasoning=f"Assessment failed due to error: {str(e)}. Fallback conservative values assigned.",
        )

    return {"credit_output": credit_result, "error_log": error_log}
