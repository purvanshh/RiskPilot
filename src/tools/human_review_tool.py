"""Phase 12 — Human-in-the-loop review tool (Member D).

Bridges the LangGraph flow to a human decision. The graph's `human_review_node`
applies a `HumanDecision` if one is already present on the state; this tool is the
piece that *produces* that decision and returns it as a state update dict, so it can
be merged in before (re-)invoking the graph.

Two modes:
  - Programmatic: pass `decision=...` (and optional officer_id / override_reason).
    Non-blocking; ideal for integration tests and headless demo runs.
  - Interactive CLI: omit `decision` and the function prompts on the console.
    (The Streamlit dashboard is the richer UI equivalent of this same step.)

The function never blocks the graph itself: LangGraph runs synchronously, so the
HITL pause is modelled by collecting the decision here and re-invoking the graph
with it attached, exactly as the officer dashboard does.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.graph.state import HumanDecision, LoanApplicationState

VALID_DECISIONS = ("approve", "deny", "override_approve", "override_deny")


def _state_to_obj(state: Any) -> LoanApplicationState:
    if isinstance(state, LoanApplicationState):
        return state
    if isinstance(state, dict):
        return LoanApplicationState.from_dict(state)
    raise TypeError("state must be a LoanApplicationState or dict")


def summarize_for_officer(state: LoanApplicationState) -> str:
    """Build a concise text briefing of all agent outputs for the reviewer."""
    lines = [f"=== Application {state.application_id} ==="]

    if state.kyc_output:
        kyc = state.kyc_output
        lines.append(
            "KYC      : "
            f"missing_docs={kyc.get('missing_critical_docs')}, "
            f"fraud_flag={kyc.get('fraud_flag')}, "
            f"confidence={kyc.get('confidence')}"
        )
    if state.credit_output:
        c = state.credit_output
        lines.append(
            f"Credit   : score={c.credit_score}, risk={c.risk_category}, "
            f"dti={c.dti_ratio:.2%}, confidence={c.confidence_score}"
        )
    if state.policy_output:
        p = state.policy_output
        lines.append(
            f"Policy   : passed={p.policy_passed}, " f"ltv={p.ltv_ratio}, violations={p.violations}"
        )
    if state.arbitrator_output:
        a = state.arbitrator_output
        lines.append(
            f"Arbitrator: recommendation={a.recommendation}, "
            f"agreement={a.agent_agreement}, confidence={a.confidence_score}"
        )
        lines.append(f"Summary   : {a.summary}")
    return "\n".join(lines)


def human_review_ui(
    state: Any,
    decision: Optional[str] = None,
    officer_id: str = "OFFICER-CLI",
    override_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect an officer decision and return it as a graph state update.

    Returns ``{"human_decision": HumanDecision}`` ready to be merged into the
    state before re-invoking the graph (the human_review_node then finalizes
    ``final_status``).

    If ``decision`` is None, prompts interactively on the console.
    """
    obj = _state_to_obj(state)

    if decision is None:
        # Interactive CLI fallback.
        print(summarize_for_officer(obj))
        print(f"\nValid decisions: {', '.join(VALID_DECISIONS)}")
        decision = input("Officer decision: ").strip()
        entered_id = input(f"Officer ID [{officer_id}]: ").strip()
        if entered_id:
            officer_id = entered_id
        if "override" in decision:
            override_reason = input("Override reason (required): ").strip()

    if decision not in VALID_DECISIONS:
        raise ValueError(f"Invalid decision '{decision}'. Must be one of {VALID_DECISIONS}.")

    if "override" in decision and not (override_reason and override_reason.strip()):
        raise ValueError("override_reason is required when overriding the recommendation.")

    human_decision = HumanDecision(
        officer_id=officer_id,
        decision=decision,
        override_reason=(override_reason.strip() if override_reason else None),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    return {"human_decision": human_decision}
