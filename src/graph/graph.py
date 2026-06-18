import logging
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from src.agents.arbitrator_agent import arbitrator_node
from src.agents.credit_agent import credit_node
from src.agents.kyc_agent import kyc_node
from src.agents.policy_agent import policy_node
from src.graph.edges import route_after_arbitrator, route_after_kyc
from src.graph.state import LoanApplicationState

logger = logging.getLogger(__name__)


def retry_node(state: LoanApplicationState) -> Dict[str, Any]:
    """
    Retry node triggered when documents are missing.
    In production, this would notify the user. For boilerplate, we log and halt.
    """
    logger.info(f"Application {state.application_id} placed in retry. Awaiting document upload.")
    error_log = list(state.error_log)
    error_log.append("Application suspended: Missing critical documents.")

    return {"final_status": "under_review", "error_log": error_log}


def human_review_node(state: LoanApplicationState) -> Dict[str, Any]:
    """
    Human Review node.
    If an officer decision is already provided in the state, apply it and finalize the status.
    Otherwise, mark the status as under_review.
    """
    logger.info(f"Running human review node for application {state.application_id}")

    if state.human_decision:
        decision = state.human_decision.decision
        if decision in ["approve", "override_approve"]:
            final_status = "approved"
        else:
            final_status = "denied"

        logger.info(f"Officer decision found: {decision}. Final status set to {final_status}.")
        return {"final_status": final_status}

    logger.info("No officer decision found. Application marked as under_review.")
    return {"final_status": "under_review"}


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
