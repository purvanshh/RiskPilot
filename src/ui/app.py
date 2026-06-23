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
from src.graph.state import HumanDecision  # noqa: E402
from src.tools.data_loader import build_state_from_app, load_test_applications  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("riskpilot.app")

# Suppress verbose third-party logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

app = Flask(__name__, static_folder="static")

TEST_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "test_applications.json")


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
        return jsonify({"error": str(e)}), 500


@app.route("/api/underwrite/<app_id>", methods=["POST"])
def underwrite_application(app_id):
    try:
        apps = load_test_applications(TEST_DATA_PATH)
        app_data = next((a for a in apps if a.get("application_id") == app_id), None)
        if not app_data:
            return jsonify({"error": f"Application {app_id} not found"}), 404

        # Build initial state from test data (resolves synthetic PDF paths if present)
        initial_state = build_state_from_app(app_data, use_pdf_paths=True)

        # Run graph
        final_state_dict = graph.invoke(initial_state)
        serialized = serialize_state(final_state_dict)
        return jsonify(serialized)
    except Exception as e:
        logger.error(f"Error executing underwrite for {app_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/decision/<app_id>", methods=["POST"])
def submit_decision(app_id):
    try:
        data = request.json or {}
        officer_id = data.get("officer_id", "OFFICER-01")
        decision = data.get("decision")
        override_reason = data.get("override_reason")

        if not decision:
            return jsonify({"error": "decision is required"}), 400

        apps = load_test_applications(TEST_DATA_PATH)
        app_data = next((a for a in apps if a.get("application_id") == app_id), None)
        if not app_data:
            return jsonify({"error": f"Application {app_id} not found"}), 404

        initial_state = build_state_from_app(app_data, use_pdf_paths=True)

        # Inject decision
        import datetime

        initial_state.human_decision = HumanDecision(
            officer_id=officer_id,
            decision=decision,
            override_reason=override_reason,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        # Re-run graph
        final_state_dict = graph.invoke(initial_state)
        serialized = serialize_state(final_state_dict)
        return jsonify(serialized)
    except Exception as e:
        logger.error(f"Error submitting decision for {app_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Ensure template and static dirs exist
    os.makedirs(os.path.join(_HERE, "templates"), exist_ok=True)
    os.makedirs(os.path.join(_HERE, "static"), exist_ok=True)
    app.run(host="0.0.0.0", port=8501, debug=True)
