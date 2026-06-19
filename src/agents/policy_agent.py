import logging
from typing import Any, Dict, List

from src.graph.state import (
    LoanApplicationState,
    PolicyCheckOutput,
    graceful_fallback,
    timeout_resilience,
    validate_state,
)
from src.tools.policy_tools import ltv_calculator, policy_retriever, policy_validator

logger = logging.getLogger(__name__)


@validate_state
@graceful_fallback("policy")
@timeout_resilience(30.0)
def policy_node(state: LoanApplicationState) -> Dict[str, Any]:
    """
    Policy / Eligibility Agent Node
    Checks application against lending policies grounded in RAG policy retrieval.
    """
    logger.info(f"Starting policy compliance checking for application {state.application_id}")

    error_log = list(state.error_log)
    try:
        # Extracted info
        loan_amount = state.applicant_data.get("loan_amount", 0)
        property_value = state.applicant_data.get("property_value", 1)
        employment_months = state.applicant_data.get("employment_months", 0)

        # Credit output info
        credit_score = 300
        dti = 0.0
        if state.credit_output:
            credit_score = state.credit_output.credit_score
            dti = state.credit_output.dti_ratio

        # 1. Compute LTV (Loan-to-Value)
        ltv = ltv_calculator(loan_amount, property_value)

        # 2. Retrieve policy documents (RAG)
        queries = [
            "What is the minimum credit score for a loan?",
            "What is the maximum DTI ratio allowed?",
            "What are the employment stability requirements?",
            f"What is the maximum LTV ratio allowed for loan amount {loan_amount}?",
        ]

        retrieved_chunks: List[str] = []
        for q in queries:
            chunks = policy_retriever(q)
            retrieved_chunks.extend(chunks)

        # Remove duplicates while keeping order
        retrieved_chunks = list(dict.fromkeys(retrieved_chunks))

        # 3. Policy validation (Checks rules based on configuration/policy text)
        validation_results = policy_validator(
            credit_score=credit_score,
            dti=dti,
            ltv=ltv,
            employment_months=employment_months,
            policy_chunks=retrieved_chunks,
        )

        policy_result = PolicyCheckOutput(
            policy_passed=validation_results["passed"],
            violations=validation_results["violations"],
            ltv_ratio=round(ltv, 4),
            min_credit_requirement_met=validation_results["min_credit_met"],
            max_dti_threshold=validation_results["max_dti_threshold"],
            retrieved_policy_chunks=retrieved_chunks[:5],  # Store top 5 chunks
            reasoning=validation_results["reasoning"],
        )

    except Exception as e:
        logger.error(f"Error in Policy Agent: {str(e)}")
        error_log.append(f"Policy Agent error: {str(e)}")
        policy_result = PolicyCheckOutput(
            policy_passed=False,
            violations=[f"System error in policy checking: {str(e)}"],
            ltv_ratio=0.0,
            min_credit_requirement_met=False,
            max_dti_threshold=0.45,
            retrieved_policy_chunks=[],
            reasoning=f"System error: {str(e)}",
        )

    return {"policy_output": policy_result, "error_log": error_log}
