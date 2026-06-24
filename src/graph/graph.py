import logging
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from src.agents.arbitrator_agent import arbitrator_node
from src.agents.credit_agent import credit_node
from src.agents.kyc_agent import kyc_node
from src.agents.policy_agent import policy_node
from src.graph.edges import route_after_arbitrator, route_after_kyc
from src.graph.state import LoanApplicationState, validate_state

logger = logging.getLogger(__name__)


def _populate_missing_agent_outputs(state: LoanApplicationState) -> Dict[str, Any]:
    """Helper to populate credit, policy, and arbitrator outputs if they were bypassed."""
    updates = {}

    if not state.credit_output:
        credit_res = credit_node(state)
        state.credit_output = credit_res.get("credit_output")
        updates["credit_output"] = state.credit_output
        if credit_res.get("error_log"):
            for err in credit_res["error_log"]:
                if err not in state.error_log:
                    state.error_log.append(err)

    if not state.policy_output:
        policy_res = policy_node(state)
        state.policy_output = policy_res.get("policy_output")
        updates["policy_output"] = state.policy_output
        if policy_res.get("error_log"):
            for err in policy_res["error_log"]:
                if err not in state.error_log:
                    state.error_log.append(err)

    if not state.arbitrator_output:
        arbitrator_res = arbitrator_node(state)
        state.arbitrator_output = arbitrator_res.get("arbitrator_output")
        updates["arbitrator_output"] = state.arbitrator_output
        if arbitrator_res.get("error_log"):
            for err in arbitrator_res["error_log"]:
                if err not in state.error_log:
                    state.error_log.append(err)

    updates["error_log"] = state.error_log
    return updates


@validate_state
def retry_node(state: LoanApplicationState) -> Dict[str, Any]:
    """
    Retry node triggered when documents are missing.
    If an officer decision is already present (e.g. officer manually reviewed
    and chose to override), apply it. Otherwise mark as under_review.
    """
    logger.info(f"Application {state.application_id} placed in retry. Awaiting document upload.")

    updated_state = LoanApplicationState.from_dict(state.to_dict())

    bypassed_updates = _populate_missing_agent_outputs(updated_state)
    for k, v in bypassed_updates.items():
        setattr(updated_state, k, v)

    if updated_state.human_decision:
        decision = updated_state.human_decision.decision
        if decision in ["approve", "override_approve"]:
            final_status = "approved"
        else:
            final_status = "denied"
        logger.info(
            f"Officer override found in retry node: {decision}. Final status: {final_status}."
        )
        updated_state.final_status = final_status
    else:
        updated_state.final_status = "under_review"
        if "Application suspended: Missing critical documents." not in updated_state.error_log:
            updated_state.error_log.append("Application suspended: Missing critical documents.")

    return {
        "final_status": updated_state.final_status,
        "error_log": updated_state.error_log,
        "credit_output": updated_state.credit_output,
        "policy_output": updated_state.policy_output,
        "arbitrator_output": updated_state.arbitrator_output,
    }


@validate_state
def human_review_node(state: LoanApplicationState) -> Dict[str, Any]:
    """
    Human Review node.
    If an officer decision is already provided in the state, apply it and finalize the status.
    Otherwise, mark the status as under_review.
    """
    logger.info(f"Running human review node for application {state.application_id}")

    updated_state = LoanApplicationState.from_dict(state.to_dict())

    bypassed_updates = _populate_missing_agent_outputs(updated_state)
    for k, v in bypassed_updates.items():
        setattr(updated_state, k, v)

    if updated_state.human_decision:
        decision = updated_state.human_decision.decision
        if decision in ["approve", "override_approve"]:
            final_status = "approved"
        else:
            final_status = "denied"

        logger.info(f"Officer decision found: {decision}. Final status set to {final_status}.")
        updated_state.final_status = final_status
    else:
        logger.info("No officer decision found. Application marked as under_review.")
        updated_state.final_status = "under_review"

    return {
        "final_status": updated_state.final_status,
        "error_log": updated_state.error_log,
        "credit_output": updated_state.credit_output,
        "policy_output": updated_state.policy_output,
        "arbitrator_output": updated_state.arbitrator_output,
    }


# Build the Graph
builder = StateGraph(LoanApplicationState)

# Add Nodes
builder.add_node("kyc", kyc_node)
builder.add_node("credit", credit_node)
builder.add_node("policy", policy_node)
builder.add_node("arbitrator", arbitrator_node)
builder.add_node("retry", retry_node)
builder.add_node("human_review", human_review_node)

# Set Entry Point
builder.set_entry_point("kyc")

# Add Conditional Edges
builder.add_conditional_edges(
    "kyc",
    route_after_kyc,
    {"credit": "credit", "retry": "retry", "human_review": "human_review"},
)

# Add Static Edges
builder.add_edge("credit", "policy")
builder.add_edge("policy", "arbitrator")

# Add Conditional Edge from Arbitrator to Human Review
builder.add_conditional_edges(
    "arbitrator", route_after_arbitrator, {"human_review": "human_review"}
)

# Connect terminal nodes to END
builder.add_edge("retry", END)
builder.add_edge("human_review", END)

# Compile Graph
graph = builder.compile()
