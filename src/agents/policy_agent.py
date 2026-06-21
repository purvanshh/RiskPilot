import logging
from typing import Any, Dict

from src.graph.state import LoanApplicationState, PolicyCheckOutput, validate_state
from src.rag.policy_agent import run_policy_check

logger = logging.getLogger(__name__)


@validate_state
def policy_node(state: LoanApplicationState) -> Dict[str, Any]:
    """
    LangGraph adapter for the canonical Policy RAG pipeline.

    All retrieval, evaluation, and reasoning lives in `src.rag.*`. This node
    only translates between graph state and the RAG subsystem and captures
    any subsystem errors into the audit log.
    """
    logger.info("Starting policy compliance checking for application %s", state.application_id)
    error_log = list(state.error_log)

    try:
        policy_result = run_policy_check(state)
    except Exception as exc:
        logger.error("Error in Policy Agent: %s", exc)
        error_log.append(f"Policy Agent error: {exc}")
        policy_result = PolicyCheckOutput(
            policy_passed=False,
            violations=[f"System error in policy checking: {exc}"],
            ltv_ratio=0.0,
            min_credit_requirement_met=False,
            max_dti_threshold=0.45,
            retrieved_policy_chunks=[],
            reasoning=f"System error: {exc}",
            confidence=0.0,
        )

    return {"policy_output": policy_result, "error_log": error_log}
