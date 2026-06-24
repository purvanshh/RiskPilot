import datetime
import logging
import os
import sys
import threading
from collections import defaultdict
from functools import wraps
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Add project root to sys.path to resolve 'src' imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_PROJECT_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from app_config import (  # noqa: E402
    ALLOWED_ORIGIN,
    API_KEYS,
    DEBUG,
    HOST,
    MAX_CONTENT_LENGTH,
    PORT,
    SECRET_KEY,
)

from src.graph.graph import graph, human_review_node  # noqa: E402
from src.graph.state import HumanDecision, LoanApplicationState  # noqa: E402
from src.tools.data_loader import build_state_from_app, load_test_applications  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("riskpilot.app")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

app = Flask(__name__, static_folder="static")
app.config["SECRET_KEY"] = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ---------------------------------------------------------------------------
# CORS — restrict to the configured origin if set
# ---------------------------------------------------------------------------
if ALLOWED_ORIGIN:
    CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGIN}})

# ---------------------------------------------------------------------------
# Rate limiting (in-memory storage; swap to Redis via RATELIMIT_STORAGE_URI
# for multi-process deployments)
# ---------------------------------------------------------------------------
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    headers_enabled=True,
)
limiter.init_app(app)

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def require_api_key(f):
    """Decorator that enforces X-API-Key header authentication.

    When API_KEYS is empty (not configured), authentication is skipped.
    In TESTING mode, authentication is always skipped.
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        if app.config.get("TESTING"):
            return f(*args, **kwargs)
        if not API_KEYS:
            return f(*args, **kwargs)
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key not in API_KEYS:
            return jsonify({"error": "Missing or invalid API key."}), 401
        return f(*args, **kwargs)

    return decorated


# ---------------------------------------------------------------------------
# Concurrency — per-application locks to prevent race conditions on
# _PIPELINE_STATE without serialising all requests globally.
# ---------------------------------------------------------------------------
_app_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)


# ---------------------------------------------------------------------------
# In-memory pipeline output store
# ---------------------------------------------------------------------------
_PIPELINE_STATE: Dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "test_applications.json")


def serialize_value(val: Any) -> Any:
    if hasattr(val, "to_dict"):
        return val.to_dict()
    if isinstance(val, list):
        return [serialize_value(x) for x in val]
    if isinstance(val, dict):
        return {k: serialize_value(v) for k, v in val.items()}
    return val


def serialize_state(state: Any) -> dict:
    if hasattr(state, "to_dict"):
        return state.to_dict()
    return {k: serialize_value(v) for k, v in state.items()}


def _parse_json_body() -> tuple[dict | None, tuple | None]:
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
    val = data.get(key, default)
    return str(val).strip() if val is not None else default


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return send_from_directory(os.path.join(_HERE, "templates"), "index.html")


@app.route("/api/applications")
@require_api_key
def get_applications():
    try:
        apps = load_test_applications(TEST_DATA_PATH)
        return jsonify(apps)
    except Exception as e:
        logger.error(f"Error loading applications: {e}")
        return jsonify({"error": "Failed to load applications."}), 500


@app.route("/api/underwrite/<app_id>", methods=["POST"])
@require_api_key
def underwrite_application(app_id: str):
    data, err = _parse_json_body()
    if err:
        return err

    try:
        fast_mode = _str_field(data, "fast_mode", "false").lower() in ("true", "1", "yes")

        apps = load_test_applications(TEST_DATA_PATH)
        app_data = next((a for a in apps if a.get("application_id") == app_id), None)
        if not app_data:
            return jsonify({"error": f"Application {app_id} not found"}), 404

        initial_state = build_state_from_app(app_data, use_pdf_paths=not fast_mode)

        # Run the full LangGraph pipeline (expensive — done outside the lock
        # so different applications can be processed concurrently).
        final_state_dict = graph.invoke(initial_state)
        serialized = serialize_state(final_state_dict)

        # Protect the write to _PIPELINE_STATE with a per-app lock.
        lock = _app_locks[app_id]
        with lock:
            _PIPELINE_STATE[app_id] = serialized

        return jsonify(serialized)
    except Exception as e:
        logger.error(f"Error executing underwrite for {app_id}: {e}", exc_info=True)
        return jsonify({"error": "Internal pipeline error. See server logs."}), 500


@app.route("/api/decision/<app_id>", methods=["POST"])
@require_api_key
@limiter.limit("10 per minute")
def submit_decision(app_id: str):
    data, err = _parse_json_body()
    if err:
        return err

    try:
        officer_id = _str_field(data, "officer_id")
        decision = _str_field(data, "decision")
        override_reason = data.get("override_reason")

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

        # ---- Locked critical section for this app_id ----
        lock = _app_locks[app_id]
        with lock:
            if app_id not in _PIPELINE_STATE:
                return (
                    jsonify({"error": "Pipeline must be run before a decision can be submitted."}),
                    400,
                )

            stored = _PIPELINE_STATE[app_id]

            # Reject if a final decision has already been recorded.
            if stored.get("final_status") in ("approved", "denied"):
                return (
                    jsonify(
                        {
                            "error": (
                                f"A decision for application {app_id} has already been "
                                "submitted. Resubmission is not allowed."
                            )
                        }
                    ),
                    409,
                )

            previous_state_dict = stored.copy()

            previous_state = LoanApplicationState.from_dict(previous_state_dict)
            previous_state.human_decision = HumanDecision(
                officer_id=officer_id,
                decision=decision,
                override_reason=override_reason
                if (override_reason and override_reason.strip())
                else None,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )

            state_updates = human_review_node(previous_state)
            merged = previous_state.to_dict()
            merged.update(state_updates)
            merged["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            serialized = serialize_state(merged)

            _PIPELINE_STATE[app_id] = serialized

        return jsonify(serialized)
    except Exception as e:
        logger.error(f"Error submitting decision for {app_id}: {e}", exc_info=True)
        return jsonify({"error": "Internal error processing decision. See server logs."}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(os.path.join(_HERE, "templates"), exist_ok=True)
    os.makedirs(os.path.join(_HERE, "static"), exist_ok=True)
    app.run(host=HOST, port=PORT, debug=DEBUG)
