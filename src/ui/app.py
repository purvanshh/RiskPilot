import datetime
import logging
import os
import shutil
import sys
import threading
import time
import uuid
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
    UPLOAD_DIR,
)

from src.graph.graph import graph, human_review_node  # noqa: E402
from src.graph.state import ExtractedDocument  # noqa: E402
from src.graph.state import HumanDecision, LoanApplicationState  # noqa: E402
from src.guardrails.input_validation import validate_application_input  # noqa: E402
from src.tools.data_loader import build_state_from_app, load_test_applications  # noqa: E402
from src.tools.document_tools import detect_document_type, parse_document  # noqa: E402

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
        start_time = time.time()

        apps = load_test_applications(TEST_DATA_PATH)
        app_data = next((a for a in apps if a.get("application_id") == app_id), None)
        if not app_data:
            return jsonify({"error": f"Application {app_id} not found"}), 404

        initial_state = build_state_from_app(app_data, use_pdf_paths=not fast_mode)

        # Run the full LangGraph pipeline (expensive — done outside the lock
        # so different applications can be processed concurrently).
        final_state_dict = graph.invoke(initial_state)
        serialized = serialize_state(final_state_dict)

        # Artificial delay when fast mode is OFF — ensure minimum 5s elapsed
        if not fast_mode:
            elapsed = time.time() - start_time
            remaining = 5.0 - elapsed
            if remaining > 0:
                time.sleep(remaining)

        # Protect the write to _PIPELINE_STATE with a per-app lock.
        lock = _app_locks[app_id]
        with lock:
            _PIPELINE_STATE[app_id] = serialized

        return jsonify(serialized)
    except Exception as e:
        logger.error(f"Error executing underwrite for {app_id}: {e}", exc_info=True)
        return jsonify({"error": "Internal pipeline error. See server logs."}), 500


@app.route("/api/underwrite/upload", methods=["POST"])
@require_api_key
def underwrite_upload():
    """Accept uploaded documents + applicant data and run the full pipeline."""
    try:
        fast_mode = request.form.get("fast_mode", "false").lower() in ("true", "1", "yes")
        start_time = time.time()

        # Parse applicant data from form fields
        try:
            applicant_data = {
                "name": request.form.get("name", "").strip(),
                "income": float(request.form.get("income", 0)),
                "monthly_debt": float(request.form.get("monthly_debt", 0)),
                "loan_amount": float(request.form.get("loan_amount", 0)),
                "property_value": float(request.form.get("property_value", 0)),
                "employment_months": int(request.form.get("employment_months", 0)),
            }

            # Validate that required fields are provided
            if not applicant_data["name"]:
                return jsonify({"error": "Applicant name is required"}), 400
            if applicant_data["income"] <= 0:
                return jsonify({"error": "Annual income must be greater than 0"}), 400
            if applicant_data["loan_amount"] <= 0:
                return jsonify({"error": "Loan amount must be greater than 0"}), 400

            logger.info(
                f"Applicant data parsed: name={applicant_data['name']}, income={applicant_data['income']}"
            )
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid applicant data: {str(e)}"}), 400

        # Collect uploaded files
        uploaded_files = request.files.getlist("documents")
        logger.info(f"Received {len(uploaded_files)} files in upload request")
        if not uploaded_files or len(uploaded_files) < 1:
            return jsonify({"error": "No files uploaded."}), 400

        # Create upload directory
        app_id = f"UPLOAD-{uuid.uuid4().hex[:8].upper()}"
        upload_subdir = os.path.join(UPLOAD_DIR, app_id)
        os.makedirs(upload_subdir, exist_ok=True)

        # Also accept explicit doc types from form
        doc_types_raw = request.form.getlist("doc_types")

        documents = []
        for i, f in enumerate(uploaded_files):
            if not f.filename:
                logger.warning(f"Skipping file at index {i} - no filename")
                continue

            # Save file
            safe_name = f"{i}_{f.filename}"
            save_path = os.path.join(upload_subdir, safe_name)
            logger.info(f"Saving file {i}: {f.filename} to {save_path}")
            f.save(save_path)

            # Determine doc type
            if i < len(doc_types_raw) and doc_types_raw[i]:
                doc_type = doc_types_raw[i]
            else:
                # Try to parse content for type detection
                try:
                    content = parse_document(save_path)
                except Exception:
                    content = ""
                doc_type = detect_document_type(f.filename, content)

            # Parse the document text
            try:
                extracted_text = parse_document(save_path)
            except Exception as e:
                extracted_text = f"Error parsing: {str(e)}"

            documents.append(
                ExtractedDocument(
                    document_type=doc_type,
                    extracted_text=extracted_text,
                    validation_status="valid",
                    confidence=0.85,
                    extracted_fields={},
                )
            )

        # Run input validation guardrails
        is_valid, errors = validate_application_input(
            applicant_data, [d.to_dict() for d in documents], app_id
        )

        # Build state and run pipeline
        initial_state = LoanApplicationState(
            application_id=app_id,
            applicant_data=applicant_data,
            documents=documents,
        )

        final_state_dict = graph.invoke(initial_state)
        serialized = serialize_state(final_state_dict)

        # Include guardrail validation results in response
        serialized["guardrail_validation"] = {
            "is_valid": is_valid,
            "errors": errors,
        }
        serialized["application_id"] = app_id

        # Artificial delay when fast mode is OFF
        if not fast_mode:
            elapsed = time.time() - start_time
            remaining = 5.0 - elapsed
            if remaining > 0:
                time.sleep(remaining)

        # Store result
        lock = _app_locks[app_id]
        with lock:
            _PIPELINE_STATE[app_id] = serialized

        return jsonify(serialized)
    except Exception as e:
        logger.error(f"Error processing upload: {e}", exc_info=True)
        return jsonify({"error": f"Pipeline error: {str(e)}"}), 500


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
            kyc_out = stored.get("kyc_output", {})
            arb_out = stored.get("arbitrator_output", {})

            # Guardrail: detect risky applications that require override decisions
            kyc_confidence_low = kyc_out.get("confidence", 1.0) < 0.5
            kyc_has_flags = kyc_out.get("fraud_flag", False) or kyc_out.get(
                "missing_critical_docs", False
            )
            arb_review_required = (
                arb_out.get("recommendation", "") == "review_required"
                if isinstance(arb_out, dict)
                else getattr(arb_out, "recommendation", "") == "review_required"
            )
            is_risky = kyc_confidence_low or kyc_has_flags or arb_review_required

            if is_risky and decision not in ("override_approve", "override_deny"):
                return (
                    jsonify(
                        {
                            "error": (
                                "This application has risk flags (low KYC confidence, fraud alert, "
                                "missing documents, or arbitrator requires review). You must use an "
                                "OVERRIDE decision (override_approve or override_deny) with a "
                                "mandatory justification."
                            )
                        }
                    ),
                    400,
                )

            if is_risky and not (override_reason and override_reason.strip()):
                return (
                    jsonify(
                        {
                            "error": (
                                "Override justification is REQUIRED for applications with risk flags. "
                                "Please provide the rationale for your override decision."
                            )
                        }
                    ),
                    400,
                )

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
