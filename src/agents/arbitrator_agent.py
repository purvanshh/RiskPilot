import logging
from typing import Any, Dict, List

from src.graph.state import ArbitratorOutput, LoanApplicationState

logger = logging.getLogger(__name__)


def arbitrator_node(state: LoanApplicationState) -> Dict[str, Any]:
    """
    Arbitrator Agent Node
    Aggregates the decisions from KYC, Credit Risk, and Policy agents,
    evaluates conflicts, and calculates a final recommendation with confidence scoring.
    """
    logger.info(f"Starting arbitration for application {state.application_id}")

    error_log = list(state.error_log)

    # 1. Gather signals
    kyc_ok = True
    kyc_fraud = False
    kyc_missing_docs = False

    if state.kyc_output:
        kyc = state.kyc_output
        kyc_ok = not kyc.get("missing_critical_docs", False) and not kyc.get("fraud_flag", False)
        kyc_fraud = kyc.get("fraud_flag", False)
        kyc_missing_docs = kyc.get("missing_critical_docs", False)

    credit_score = 600
    credit_risk = "medium"
    if state.credit_output:
        credit_score = state.credit_output.credit_score
        credit_risk = state.credit_output.risk_category

    policy_passed = True
    policy_violations: List[str] = []
    if state.policy_output:
        policy_passed = state.policy_output.policy_passed
        policy_violations = state.policy_output.violations

    # 2. Conflict Detection & Recommendation Rules
    risk_flags = []
    recommendation = "approve"
    agreement = "unanimous"
    confidence_score = 0.90

    if kyc_fraud:
        risk_flags.append("KYC Fraud Flagged (Document/Name mismatch)")
        recommendation = "review_required"
        agreement = "conflict"
        confidence_score = 0.95  # Confident that human review is required
    elif kyc_missing_docs:
        risk_flags.append("KYC Missing critical documents")
        recommendation = "review_required"
        agreement = "conflict"
        confidence_score = 0.90

    if credit_risk in ["high", "very_high"]:
        risk_flags.append(f"Credit risk category is {credit_risk.upper()} (Score: {credit_score})")
        if recommendation != "review_required":
            recommendation = "deny"
            confidence_score = 0.85

    if not policy_passed:
        violations_str = ", ".join(policy_violations)
        risk_flags.append(f"Policy check failed: {violations_str}")
        if credit_score >= 700:
            # High credit score but policy violation = Conflict!
            agreement = "conflict"
            recommendation = "review_required"
            confidence_score = 0.80
        else:
            # Policy violation with lower credit score = clear deny
            if recommendation != "review_required":
                recommendation = "deny"
                confidence_score = 0.85

    # Check for borderline cases
    if recommendation == "approve":
        # Check if borderline credit score (between 650 and 699)
        if 650 <= credit_score < 700:
            recommendation = "review_required"
            agreement = "partial"
            confidence_score = 0.70
            risk_flags.append("Borderline credit score (Tier 2)")

    # Calculate agreement level
    if recommendation == "approve" and policy_passed and kyc_ok and credit_risk == "low":
        agreement = "unanimous"
        confidence_score = 0.95
    elif recommendation == "deny" and not policy_passed and credit_risk in ["high", "very_high"]:
        agreement = "unanimous"
        confidence_score = 0.95
    elif agreement != "conflict" and (
        not policy_passed or not kyc_ok or credit_risk in ["medium", "high"]
    ):
        agreement = "partial"

    # Assemble summary
    summary_parts = []
    summary_parts.append(
        f"Arbitrator Recommendation: {recommendation.upper()} (Confidence: {confidence_score:.0%})."
    )
    summary_parts.append(f"Agent Agreement Level: {agreement.upper()}.")
    if risk_flags:
        summary_parts.append(f"Risk Flags: {'; '.join(risk_flags)}.")
    else:
        summary_parts.append("No active risk flags. Clean application profile.")

    summary = " ".join(summary_parts)

    arbitrator_output = ArbitratorOutput(
        recommendation=recommendation,
        confidence_score=confidence_score,
        agent_agreement=agreement,
        summary=summary,
        risk_flags=risk_flags,
    )

    return {"arbitrator_output": arbitrator_output, "error_log": error_log}
