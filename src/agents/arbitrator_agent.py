import logging
from typing import Any, Dict, List

from src.graph.state import (
    ArbitratorOutput,
    LoanApplicationState,
    graceful_fallback,
    timeout_resilience,
    validate_state,
)

logger = logging.getLogger(__name__)


def _agent_lean(recommendation_score: float, confidence: float) -> float:
    """A single agent's weighted contribution to the final vote.

    recommendation_score is in [-1, 1] where +1 leans fully toward APPROVE and
    -1 leans fully toward DENY. We multiply by the agent's confidence so that a
    low-confidence agent has proportionally less say in the outcome.
    """
    return confidence * recommendation_score


def compute_weighted_vote(
    kyc_confidence: float,
    kyc_score: float,
    credit_confidence: float,
    credit_score_lean: float,
    policy_confidence: float,
    policy_score_lean: float,
) -> Dict[str, float]:
    """Confidence-weighted aggregation of the three agents' leanings (Phase 8, Task 3).

    Each agent emits a recommendation_score in [-1, 1]; we weight it by that
    agent's confidence and normalise by the total confidence so the result stays
    in [-1, 1] regardless of how confident the agents are.

        weighted = Sum(confidence_i * score_i) / Sum(confidence_i)

    Returns a dict with the raw weighted score and the implied recommendation.
    This is used as a *tie-breaker* for the approve/deny decision; the hard-stop
    rules (fraud, missing docs, policy violations, high risk) still take
    precedence in arbitrator_node.
    """
    weighted_sum = (
        _agent_lean(kyc_score, kyc_confidence)
        + _agent_lean(credit_score_lean, credit_confidence)
        + _agent_lean(policy_score_lean, policy_confidence)
    )
    total_confidence = kyc_confidence + credit_confidence + policy_confidence

    weighted_score = weighted_sum / total_confidence if total_confidence > 0 else 0.0

    # Map the continuous score to a discrete recommendation.
    if weighted_score >= 0.25:
        implied = "approve"
    elif weighted_score <= -0.25:
        implied = "deny"
    else:
        implied = "review_required"

    return {"weighted_score": weighted_score, "implied_recommendation": implied}


@validate_state
@graceful_fallback("arbitrator")
@timeout_resilience(30.0)
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

    kyc_confidence = float(state.kyc_output.get("confidence", 0.5)) if state.kyc_output else 0.5

    credit_score = 600
    credit_risk = "medium"
    credit_confidence = 0.5
    if state.credit_output:
        credit_score = state.credit_output.credit_score
        credit_risk = state.credit_output.risk_category
        credit_confidence = state.credit_output.confidence_score

    policy_passed = True
    policy_violations: List[str] = []
    policy_confidence = 0.5
    if state.policy_output:
        policy_passed = state.policy_output.policy_passed
        policy_violations = state.policy_output.violations
        # PolicyCheckOutput has no confidence field; derive one. A clear pass or a
        # clear violation with supporting RAG evidence is treated as high-confidence.
        has_evidence = bool(state.policy_output.retrieved_policy_chunks)
        policy_confidence = 0.9 if has_evidence else 0.7

    # --- Confidence-weighted vote (Phase 8, Task 3) ---
    # Translate each agent's signal into a recommendation_score in [-1, 1].
    kyc_lean = -1.0 if (kyc_fraud or kyc_missing_docs) else 1.0
    if credit_risk == "low":
        credit_lean = 1.0
    elif credit_risk == "medium":
        credit_lean = 0.25
    elif credit_risk == "high":
        credit_lean = -0.75
    else:  # very_high
        credit_lean = -1.0
    policy_lean = 1.0 if policy_passed else -1.0

    vote = compute_weighted_vote(
        kyc_confidence=kyc_confidence,
        kyc_score=kyc_lean,
        credit_confidence=credit_confidence,
        credit_score_lean=credit_lean,
        policy_confidence=policy_confidence,
        policy_score_lean=policy_lean,
    )
    weighted_score = vote["weighted_score"]

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
        # Weighted vote as tie-breaker: if the hard rules left us at "approve"
        # but the confidence-weighted vote of all three agents does not support
        # approval, defer to human review rather than auto-approving.
        elif vote["implied_recommendation"] != "approve":
            recommendation = "review_required"
            agreement = "partial"
            confidence_score = 0.72
            risk_flags.append(
                f"Weighted agent vote inconclusive (score {weighted_score:+.2f})"
            )

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
    summary_parts.append(f"Weighted vote score: {weighted_score:+.2f} (range -1 deny .. +1 approve).")
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
