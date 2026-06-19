"""
output_validation.py — Output Guardrails for RiskPilot

Enforces PRD §8.2 and §8.3 safety requirements on agent outputs:
  - Confidence threshold: <0.6 forces human review
  - DTI hard stop: DTI >0.6 auto-flags regardless of other signals
  - Policy conflict detection: agent disagreement forces review
  - HITL prevention: system NEVER auto-communicates with applicant
  - Audit logging: all flags written to logs/audit.jsonl
"""

import logging
from typing import List, Optional, Tuple

from src.graph.state import ArbitratorOutput, CreditRiskOutput

logger = logging.getLogger(__name__)

# PRD §8.2 thresholds
_CONFIDENCE_THRESHOLD = 0.60
_DTI_HARD_STOP = 0.60


def validate_credit_output(
    credit_output: CreditRiskOutput,
    application_id: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """
    Applies output guardrails on the Credit Risk Agent's output.

    Checks:
      1. Confidence score < 0.6 → human review required
      2. DTI ratio > 0.6 → hard stop (PRD §8.2 policy hard stop)

    Args:
        credit_output: CreditRiskOutput from the credit agent node
        application_id: Optional, used for audit log entries

    Returns:
        (requires_review: bool, flags: List[str])
    """
    from src.guardrails.audit_logger import log_guardrail_flag

    flags: List[str] = []

    # Guard 1: Confidence threshold check
    if credit_output.confidence_score < _CONFIDENCE_THRESHOLD:
        msg = (
            f"Credit assessment confidence {credit_output.confidence_score:.2f} "
            f"is below the {_CONFIDENCE_THRESHOLD:.0%} safety threshold. "
            "Human review required."
        )
        flags.append(msg)
        if application_id:
            log_guardrail_flag(application_id, "output", msg)
        logger.warning(f"[OutputGuardrail] {msg}")

    # Guard 2: DTI hard stop
    if credit_output.dti_ratio > _DTI_HARD_STOP:
        msg = (
            f"DTI ratio {credit_output.dti_ratio:.2%} exceeds the "
            f"{_DTI_HARD_STOP:.0%} hard stop threshold. "
            "Application auto-flagged for review regardless of other signals."
        )
        flags.append(msg)
        if application_id:
            log_guardrail_flag(application_id, "output", msg)
        logger.warning(f"[OutputGuardrail] {msg}")

    requires_review = len(flags) > 0
    return requires_review, flags


def validate_system_recommendation(
    arbitrator_output: ArbitratorOutput,
    credit_output: Optional[CreditRiskOutput] = None,
    application_id: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """
    Applies safety guardrails on the Arbitrator's final recommendation output.

    Checks:
      1. Confidence score < 0.6 → review required
      2. Agent disagreement (conflict) → review required
      3. DTI hard stop (if credit_output provided) → review required
      4. Mandatory HITL: system NEVER auto-communicates to applicant

    Args:
        arbitrator_output: ArbitratorOutput from the arbitrator node
        credit_output: Optional CreditRiskOutput for DTI hard stop check
        application_id: Optional, used for audit log entries

    Returns:
        (requires_review_override: bool, flags: List[str])
    """
    from src.guardrails.audit_logger import log_guardrail_flag

    flags: List[str] = []

    # Guard 1: Confidence threshold check
    if arbitrator_output.confidence_score < _CONFIDENCE_THRESHOLD:
        msg = (
            f"Arbitrator confidence score {arbitrator_output.confidence_score:.2f} "
            f"is below the {_CONFIDENCE_THRESHOLD:.0%} safety threshold. Review required."
        )
        flags.append(msg)
        if application_id:
            log_guardrail_flag(application_id, "output", msg)

    # Guard 2: Agent conflict / disagreement detection
    if arbitrator_output.agent_agreement == "conflict":
        msg = (
            "Agent disagreement detected (conflict). "
            "Arbitrator output flagged for conflict resolution by human officer."
        )
        flags.append(msg)
        if application_id:
            log_guardrail_flag(application_id, "output", msg)

    # Guard 3: DTI hard stop (if credit output available for cross-check)
    if credit_output is not None and credit_output.dti_ratio > _DTI_HARD_STOP:
        msg = (
            f"Credit output DTI {credit_output.dti_ratio:.2%} exceeds hard stop threshold "
            f"({_DTI_HARD_STOP:.0%}). System recommendation overridden to 'review_required'."
        )
        flags.append(msg)
        if application_id:
            log_guardrail_flag(application_id, "output", msg)

    # Guard 4: Mandatory HITL — system NEVER communicates directly with applicant (PRD §8.3)
    hitl_msg = (
        "Mandatory Human-in-the-Loop review triggered. "
        "Underwriting outputs must not be sent directly to client. "
        "Loan officer approval required before any communication."
    )
    flags.append(hitl_msg)

    # Requires review override if any non-HITL guard fired
    requires_review_override = (
        arbitrator_output.confidence_score < _CONFIDENCE_THRESHOLD
        or arbitrator_output.agent_agreement == "conflict"
        or (credit_output is not None and credit_output.dti_ratio > _DTI_HARD_STOP)
    )

    return requires_review_override, flags
