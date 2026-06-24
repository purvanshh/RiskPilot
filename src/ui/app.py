import datetime
import logging
import os
import sys

from flask import Flask, jsonify, request, send_from_directory

# Add project root to sys.path to resolve 'src' imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.graph.graph import graph  # noqa: E402
from src.graph.state import HumanDecision, LoanApplicationState  # noqa: E402
from src.tools.data_loader import build_state_from_app, load_test_applications  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("riskpilot.app")

# Suppress verbose third-party logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

app = Flask(__name__, static_folder="static")

TEST_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "test_applications.json")

# In-memory store of pipeline output states keyed by application_id.
# Used by the decision endpoint to apply human decisions to the *exact*
# agent outputs the officer reviewed, preventing state desync.
_PIPELINE_STATE: dict[str, dict] = {}


def serialize_value(val):
    if hasattr(val, "to_dict"):
        return val.to_dict()
    elif isinstance(val, list):
        return [serialize_value(x) for x in val]
    elif isinstance(val, dict):
        return {k: serialize_value(v) for k, v in val.items()}
    return val


def serialize_state(state):
    if hasattr(state, "to_dict"):
        return state.to_dict()
    res = {}
    for k, v in state.items():
        res[k] = serialize_value(v)
    return res


@app.route("/")
def index():
    return send_from_directory(os.path.join(_HERE, "templates"), "index.html")


@app.route("/api/applications")
def get_applications():
    try:
        apps = load_test_applications(TEST_DATA_PATH)
        return jsonify(apps)
    except Exception as e:
        logger.error(f"Error loading applications: {e}")
        return jsonify({"error": "Failed to load applications."}), 500


def _parse_json_body() -> tuple[dict | None, tuple | None]:
    """Parse and validate the JSON request body.

    Returns:
        (parsed_dict, None) on success.
        (None, error_response_tuple) on failure — caller should return the tuple immediately.
    """
    data = request.get_json(silent=True)
    if data is None:
        return None, (jsonify({"error": "Request body must be a valid JSON object."}), 400)
    if not isinstance(data, dict):
        return None, (
            jsonify({"error": "Request body must be a JSON object, not a primitive or array."}),
            400,
        )
    return data, None


def _str_field(data: dict, key: str, default: str = "") -> str:
    """Safely extract a string field, coercing non-string values."""
    val = data.get(key, default)
    return str(val).strip() if val is not None else default


@app.route("/api/underwrite/<app_id>", methods=["POST"])
def underwrite_application(app_id):
    data, err = _parse_json_body()
    if err:
        return err

    try:
        fast_mode = _str_field(data, "fast_mode", "false").lower() in ("true", "1", "yes")

        apps = load_test_applications(TEST_DATA_PATH)
        app_data = next((a for a in apps if a.get("application_id") == app_id), None)
        if not app_data:
            return jsonify({"error": f"Application {app_id} not found"}), 404

        # Build initial state from test data (resolves synthetic PDF paths if present)
        initial_state = build_state_from_app(app_data, use_pdf_paths=not fast_mode)

        # Run graph
        final_state_dict = graph.invoke(initial_state)
        serialized = serialize_state(final_state_dict)

        # Persist the pipeline output state for the decision endpoint
        _PIPELINE_STATE[app_id] = serialized

        return jsonify(serialized)
    except Exception as e:
        logger.error(f"Error executing underwrite for {app_id}: {e}", exc_info=True)
        return jsonify({"error": "Internal pipeline error. See server logs."}), 500


@app.route("/api/decision/<app_id>", methods=["POST"])
def submit_decision(app_id):
    data, err = _parse_json_body()
    if err:
        return err

    try:
        officer_id = _str_field(data, "officer_id")
        decision = _str_field(data, "decision")
        override_reason = data.get("override_reason")

        # --- Input validation ---
        if not decision:
            return jsonify({"error": "'decision' field is required."}), 400

        valid_decisions = {"approve", "deny", "override_approve", "override_deny"}
        if decision not in valid_decisions:
            return (
                jsonify(
                    {
                        "error": (
                            f"Invalid decision '{decision}'. "
                            f"Must be one of: {sorted(valid_decisions)}"
                        )
                    }
                ),
                400,
            )

        if not officer_id:
            return jsonify({"error": "'officer_id' must be a non-empty string."}), 400

        # Enforce that the pipeline must have run before a decision is accepted
        if app_id not in _PIPELINE_STATE:
            return (
                jsonify({"error": "Pipeline must be run before a decision can be submitted."}),
                400,
            )

        # Use the stored pipeline state to preserve agent outputs exactly as the
        # officer reviewed them, then apply the human decision.
        previous_state_dict = _PIPELINE_STATE[app_id].copy()

        # Reconstruct LoanApplicationState from the stored state dict
        previous_state = LoanApplicationState.from_dict(previous_state_dict)

        # Inject the human decision while preserving all computed agent outputs
        previous_state.human_decision = HumanDecision(
            officer_id=officer_id,
            decision=decision,
            override_reason=override_reason
            if (override_reason and override_reason.strip())
            else None,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        # Run only the human_review node to apply the decision,
        # keeping all prior agent outputs intact.
        from src.graph.graph import human_review_node

        state_updates = human_review_node(previous_state)
        merged = previous_state.to_dict()
        merged.update(state_updates)
        merged["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        serialized = serialize_state(merged)
        return jsonify(serialized)
    except Exception as e:
        logger.error(f"Error submitting decision for {app_id}: {e}", exc_info=True)
        return jsonify({"error": "Internal error processing decision. See server logs."}), 500


if __name__ == "__main__":
    # Ensure template and static dirs exist
    os.makedirs(os.path.join(_HERE, "templates"), exist_ok=True)
    os.makedirs(os.path.join(_HERE, "static"), exist_ok=True)
    app.run(host="0.0.0.0", port=8501, debug=True)
