import logging
from typing import Literal

from src.graph.state import LoanApplicationState

logger = logging.getLogger(__name__)


def route_after_kyc(
    state: LoanApplicationState,
) -> Literal["credit", "human_review", "retry"]:
    """
    Routes application after KYC step:
    - If missing critical documents -> retry
    - If fraud flag detected -> human_review
    - Otherwise -> credit risk assessment
    """
    kyc = state.kyc_output
    if not kyc:
        logger.warning("KYC output not found. Routing to human review for safety.")
        return "human_review"

    if kyc.get("missing_critical_docs"):
        logger.info("KYC: Missing critical documents. Routing to RETRY node/loop.")
        return "retry"

    if kyc.get("fraud_flag"):
        logger.warning("KYC: Fraud/inconsistency flagged! Routing directly to HUMAN_REVIEW.")
        return "human_review"

    logger.info("KYC: Documentation validated. Routing to CREDIT assessment.")
    return "credit"


def route_after_arbitrator(state: LoanApplicationState) -> Literal["human_review"]:
    """
    Routes application after Arbitrator step:
    - All recommendations (approve, deny, review) must go through HITL review.
    """
    logger.info("Arbitrator completed. Routing to HUMAN_REVIEW node for mandatory officer review.")
    return "human_review"
