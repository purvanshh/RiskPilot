"""Phase 13 — Runnable entry point with LangSmith observability (Member D).

Runs the loan-underwriting graph end-to-end on a test application and emits a
LangSmith trace for the run.

Tracing approach: LangGraph auto-instruments to LangSmith whenever the standard
env vars are set (LANGSMITH_TRACING / LANGCHAIN_TRACING_V2 = true), so every node
(kyc, credit, policy, arbitrator, human_review) appears in the trace automatically
— no per-node @traceable decorators needed. We attach a stable run_id so the
trace is addressable and print its URL at the end of the run.

Usage:
    python -m src.main                 # run first test application
    python -m src.main APP-002         # run a specific application id
    python -m src.main --all           # run every test application

Requires (in .env, see .env.example):
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=<your-key>
    LANGSMITH_PROJECT=RiskPilot
"""

import logging
import os
import sys
import uuid

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass

from src.graph.graph import graph
from src.tools.data_loader import build_state_from_app, iter_states, load_test_applications

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("riskpilot.main")


def _tracing_enabled() -> bool:
    """True if either of the LangSmith/LangChain tracing flags is set truthy."""
    flags = (
        os.getenv("LANGSMITH_TRACING", ""),
        os.getenv("LANGCHAIN_TRACING_V2", ""),
    )
    return any(f.strip().lower() in ("1", "true", "yes") for f in flags)


def _normalize_tracing_env() -> None:
    """LangGraph reads LANGCHAIN_* names; mirror the LANGSMITH_* ones if only those are set."""
    mirror = {
        "LANGSMITH_TRACING": "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY": "LANGCHAIN_API_KEY",
        "LANGSMITH_PROJECT": "LANGCHAIN_PROJECT",
        "LANGSMITH_ENDPOINT": "LANGCHAIN_ENDPOINT",
    }
    for src_key, dst_key in mirror.items():
        if os.getenv(src_key) and not os.getenv(dst_key):
            os.environ[dst_key] = os.environ[src_key]


def _print_trace_url(run_id: str) -> None:
    """Best-effort lookup and print of the LangSmith trace URL for this run."""
    if not _tracing_enabled():
        print(
            "[trace] LangSmith tracing is OFF. Set LANGSMITH_TRACING=true and "
            "LANGSMITH_API_KEY in .env to capture a trace URL."
        )
        return
    try:
        from langsmith import Client

        client = Client()
        url = client.get_run_url(run_id=run_id)
        print(f"[trace] View trace: {url}")
    except Exception as e:  # pragma: no cover - network/credentials dependent
        project = os.getenv("LANGSMITH_PROJECT", "RiskPilot")
        print(f"[trace] Trace captured (run_id={run_id}) but URL lookup failed: {e}")
        print(f"        Open the LangSmith '{project}' project to view it.")


def run_application(app: dict) -> dict:
    """Run a single application through the graph, traced, and return the final state."""
    _normalize_tracing_env()

    trace_id = str(uuid.uuid4())
    state = build_state_from_app(app)
    state.trace_id = trace_id

    app_id = app.get("application_id", "UNKNOWN")
    print(f"\n{'=' * 60}\nRunning application {app_id}  (trace_id={trace_id})\n{'=' * 60}")

    # Pass a stable run_id so the trace is addressable for the URL lookup.
    config = {"run_id": trace_id, "run_name": f"loan-underwriting-{app_id}"}
    final_state = graph.invoke(state, config=config)

    arb = final_state.get("arbitrator_output")
    if arb:
        print(f"Recommendation : {arb.recommendation.upper()}")
        print(f"Agreement      : {arb.agent_agreement}")
        print(f"Summary        : {arb.summary}")
    print(f"Final status   : {final_state.get('final_status')}")

    _print_trace_url(trace_id)
    return final_state


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--all" in argv:
        for state in iter_states():
            run_application(state.to_dict() if hasattr(state, "to_dict") else state)
        return

    apps = load_test_applications()
    if not apps:
        print("No test applications found in data/test_applications.json")
        return

    if argv:
        target = argv[0]
        match = next((a for a in apps if a.get("application_id") == target), None)
        if match is None:
            print(
                f"Application '{target}' not found. Available: "
                f"{', '.join(a.get('application_id', '?') for a in apps)}"
            )
            return
        run_application(match)
    else:
        run_application(apps[0])


if __name__ == "__main__":
    main()
