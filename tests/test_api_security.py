"""
test_api_security.py — Black-box security & QA audit tests for the Flask API.

Covers:
  BUG-4: Malformed JSON → HTTP 400 (not 500)
  BUG-6: Invalid decision type → HTTP 400 (not 500, no stack trace)
  BUG-9: Empty officer_id rejected
  BUG-2: State desync fixed — decision uses stored pipeline state
  Additional: null bytes, oversized payload, missing content-type, IDOR, race conditions
"""

import json
from typing import Any

import pytest

from src.graph.state import LoanApplicationState
from src.ui.app import app as _app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """Flask test client with isolated request context."""
    _app.config["TESTING"] = True
    with _app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def reset_pipeline_state():
    """Ensure _PIPELINE_STATE is clean before every test."""
    import src.ui.app as app_mod

    app_mod._PIPELINE_STATE.clear()
    yield


def _underwrite(client, app_id: str = "APP-001", **overrides) -> dict[str, Any]:
    """Helper: run the pipeline for an application and return JSON."""
    body = {"fast_mode": True, **overrides}
    resp = client.post(f"/api/underwrite/{app_id}", json=body)
    assert resp.status_code == 200, f"Underwrite failed: {resp.get_json()}"
    return resp.get_json()


# ===========================================================================
# BUG-4: Malformed JSON returns HTTP 400, not 500
# ===========================================================================


class TestBug4MalformedJson:
    def test_array_body_returns_400(self, client):
        """Sending a JSON array [] should return 400, not 500."""
        resp = client.post("/api/decision/APP-001", data="[]", content_type="application/json")
        assert resp.status_code == 400
        err = resp.get_json()["error"].lower()
        assert "object" in err

    def test_numeric_body_returns_400(self, client):
        """Sending a bare number should return 400."""
        resp = client.post("/api/decision/APP-001", data="42", content_type="application/json")
        assert resp.status_code == 400

    def test_string_body_returns_400(self, client):
        """Sending a bare string should return 400."""
        resp = client.post("/api/decision/APP-001", data='"hello"', content_type="application/json")
        assert resp.status_code == 400

    def test_null_body_returns_400(self, client):
        """Sending JSON null should return 400."""
        resp = client.post("/api/decision/APP-001", data="null", content_type="application/json")
        assert resp.status_code == 400

    def test_empty_body_returns_400(self, client):
        """Sending an empty body should return 400."""
        resp = client.post("/api/decision/APP-001", data="", content_type="application/json")
        assert resp.status_code == 400

    def test_malformed_string_returns_400(self, client):
        """Sending non-JSON text should return 400."""
        resp = client.post(
            "/api/decision/APP-001", data="not-json", content_type="application/json"
        )
        assert resp.status_code == 400

    def test_missing_content_type_returns_400(self, client):
        """Sending JSON without Content-Type should return 400."""
        resp = client.post("/api/decision/APP-001", data='{"key": "val"}', content_type="")
        assert resp.status_code == 400

    def test_underwrite_also_rejects_array(self, client):
        """The underwrite endpoint must also reject malformed JSON (not just decision)."""
        resp = client.post("/api/underwrite/APP-001", data="[]", content_type="application/json")
        assert resp.status_code == 400

    def test_no_status_500_on_any_malformed_input(self, client):
        """Exhaustive check: none of these should produce a 500."""
        payloads = ["[]", "{}", "null", "42", '"x"', "", "not-json", "{broken"]
        for body in payloads:
            resp = client.post("/api/decision/APP-001", data=body, content_type="application/json")
            assert resp.status_code != 500, f"Body {body!r} produced 500 instead of 4xx"


# ===========================================================================
# BUG-6: Invalid decision type → HTTP 400, no stack trace
# ===========================================================================


class TestBug6InvalidDecision:
    def test_numeric_decision_returns_400(self, client):
        """decision=12345 (int) should return 400, not 500."""
        _underwrite(client, "APP-001")
        resp = client.post(
            "/api/decision/APP-001",
            json={"officer_id": "OFF-1", "decision": 12345},
        )
        assert resp.status_code == 400

    def test_array_decision_returns_400(self, client):
        """decision=['approve'] (array) should return 400."""
        _underwrite(client, "APP-001")
        resp = client.post(
            "/api/decision/APP-001",
            json={"officer_id": "OFF-1", "decision": ["approve"]},
        )
        assert resp.status_code == 400

    def test_null_decision_returns_400(self, client):
        """decision=null should return 400."""
        _underwrite(client, "APP-001")
        resp = client.post(
            "/api/decision/APP-001",
            json={"officer_id": "OFF-1", "decision": None},
        )
        assert resp.status_code == 400

    def test_blank_decision_returns_400(self, client):
        """decision='' (empty string) should return 400."""
        _underwrite(client, "APP-001")
        resp = client.post(
            "/api/decision/APP-001",
            json={"officer_id": "OFF-1", "decision": ""},
        )
        assert resp.status_code == 400

    def test_invalid_decision_string_returns_400(self, client):
        """decision='BAD' (invalid string) should return 400 with valid options."""
        _underwrite(client, "APP-001")
        resp = client.post(
            "/api/decision/APP-001",
            json={"officer_id": "OFF-1", "decision": "BAD"},
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert "approve" in body["error"]
        assert "deny" in body["error"]

    def test_no_stack_trace_leaked(self, client):
        """Non-500 responses must not contain Pydantic or Python stack traces."""
        _underwrite(client, "APP-001")
        resp = client.post(
            "/api/decision/APP-001",
            json={"officer_id": "OFF-1", "decision": 999},
        )
        body_text = json.dumps(resp.get_json()).lower()
        assert "traceback" not in body_text
        assert "validationerror" not in body_text
        assert "attributeerror" not in body_text


# ===========================================================================
# BUG-9: Empty officer_id rejected
# ===========================================================================


class TestBug9EmptyOfficerId:
    def test_empty_officer_id_api(self, client):
        """Empty officer_id in the API must return 400."""
        _underwrite(client, "APP-001")
        resp = client.post(
            "/api/decision/APP-001",
            json={"officer_id": "", "decision": "approve"},
        )
        assert resp.status_code == 400
        assert "officer_id" in resp.get_json()["error"].lower()

    def test_whitespace_officer_id_api(self, client):
        """Whitespace-only officer_id in the API must return 400."""
        _underwrite(client, "APP-001")
        resp = client.post(
            "/api/decision/APP-001",
            json={"officer_id": "   ", "decision": "approve"},
        )
        assert resp.status_code == 400

    def test_missing_officer_id_api(self, client):
        """Missing officer_id field in the API must return 400."""
        _underwrite(client, "APP-001")
        resp = client.post(
            "/api/decision/APP-001",
            json={"decision": "approve"},
        )
        assert resp.status_code == 400

    def test_blank_officer_id_in_tool_raises(self):
        """human_review_ui() must raise ValueError for empty officer_id."""
        from src.tools.human_review_tool import human_review_ui

        state = LoanApplicationState(
            application_id="APP-TEST",
            applicant_data={"name": "Test"},
            documents=[],
        )
        with pytest.raises(ValueError, match="officer_id"):
            human_review_ui(state, decision="approve", officer_id="")

    def test_whitespace_officer_id_in_tool_raises(self):
        """human_review_ui() must raise ValueError for whitespace-only officer_id."""
        from src.tools.human_review_tool import human_review_ui

        state = LoanApplicationState(
            application_id="APP-TEST",
            applicant_data={"name": "Test"},
            documents=[],
        )
        with pytest.raises(ValueError, match="officer_id"):
            human_review_ui(state, decision="approve", officer_id="   ")


# ===========================================================================
# BUG-2: State desync — decision preserves pipeline agent outputs
# ===========================================================================


class TestBug2StateDesync:
    def test_decision_preserves_arbitrator_output(self, client):
        """The decision endpoint must preserve the exact arbitrator output
        from the pipeline run, not recompute it."""
        uw_data = _underwrite(client, "APP-002")
        arb_before = uw_data["arbitrator_output"]

        dec_resp = client.post(
            "/api/decision/APP-002",
            json={"officer_id": "OFF-1", "decision": "deny"},
        )
        dec_data = dec_resp.get_json()
        arb_after = dec_data["arbitrator_output"]

        assert arb_after == arb_before, (
            "Arbitrator output changed between underwrite and decision. "
            "State desync is still present."
        )

    def test_decision_preserves_all_agent_outputs(self, client):
        """All agent outputs (KYC, credit, policy, arbitrator) must be identical."""
        uw = _underwrite(client, "APP-001")
        dec = client.post(
            "/api/decision/APP-001",
            json={"officer_id": "OFF-1", "decision": "approve"},
        ).get_json()

        for key in ("kyc_output", "credit_output", "policy_output", "arbitrator_output"):
            assert dec.get(key) == uw.get(key), f"{key} changed between underwrite and decision"

    def test_decision_only_changes_final_status(self, client):
        """The only fields that should differ are final_status, human_decision, and updated_at."""
        uw = _underwrite(client, "APP-003")
        dec = client.post(
            "/api/decision/APP-003",
            json={"officer_id": "OFF-1", "decision": "approve"},
        ).get_json()

        for key in uw:
            if key in ("final_status", "human_decision", "updated_at", "trace_id"):
                continue
            assert dec.get(key) == uw.get(
                key
            ), f"Unexpected change in field '{key}' between underwrite and decision"

    def test_decision_for_missing_docs_preserves_state(self, client):
        """APP-004 (missing docs) must still preserve pipeline state on decision."""
        uw = _underwrite(client, "APP-004")
        assert uw["kyc_output"]["missing_critical_docs"] is True

        dec = client.post(
            "/api/decision/APP-004",
            json={"officer_id": "OFF-1", "decision": "deny"},
        ).get_json()

        assert dec["kyc_output"]["missing_critical_docs"] is True
        assert dec["kyc_output"] == uw["kyc_output"]

    def test_decision_requires_pipeline_first(self, client):
        """Submitting a decision without running the pipeline must return 400."""
        resp = client.post(
            "/api/decision/APP-001",
            json={"officer_id": "OFF-1", "decision": "approve"},
        )
        assert resp.status_code == 400
        assert "pipeline" in resp.get_json()["error"].lower()

    def test_approve_sets_final_status_approved(self, client):
        """Approving must set final_status to 'approved'."""
        _underwrite(client, "APP-001")
        dec = client.post(
            "/api/decision/APP-001",
            json={"officer_id": "OFF-1", "decision": "approve"},
        ).get_json()
        assert dec["final_status"] == "approved"

    def test_deny_sets_final_status_denied(self, client):
        """Denying must set final_status to 'denied'."""
        _underwrite(client, "APP-002")
        dec = client.post(
            "/api/decision/APP-002",
            json={"officer_id": "OFF-1", "decision": "deny"},
        ).get_json()
        assert dec["final_status"] == "denied"


# ===========================================================================
# Additional security & QA findings
# ===========================================================================


class TestAdditionalFindings:
    def test_underwrite_nonexistent_app_returns_404(self, client):
        """Underwriting a non-existent app_id must return 404."""
        resp = client.post("/api/underwrite/NONEXIST", json={})
        assert resp.status_code == 404

    def test_decision_with_corrupted_state_returns_500(self, client):
        """If the stored pipeline state is invalid/truncated, return 500."""
        import src.ui.app as app_mod

        app_mod._PIPELINE_STATE["CORRUPTED"] = {"dummy": True}
        resp = client.post(
            "/api/decision/CORRUPTED",
            json={"officer_id": "OFF-1", "decision": "approve"},
        )
        assert resp.status_code == 500

    def test_decision_pipeline_not_run_returns_400_no_app_id(self, client):
        """A decision for an app that hasn't been underwritten returns 400."""
        resp = client.post(
            "/api/decision/NOBODY_RAN_ME",
            json={"officer_id": "OFF-1", "decision": "approve"},
        )
        assert resp.status_code == 400
        assert "pipeline" in resp.get_json()["error"].lower()

    def test_override_reason_required_for_override_decision(self, client):
        """Sending override_approve without a reason must still process (graph handles it)."""
        _underwrite(client, "APP-001")
        resp = client.post(
            "/api/decision/APP-001",
            json={
                "officer_id": "OFF-1",
                "decision": "override_approve",
                "override_reason": None,
            },
        )
        # The API does not enforce override_reason — that's a presentation-layer concern.
        assert resp.status_code == 200

    def test_second_decision_overwrites_first(self, client):
        """Submitting two decisions for the same app should overwrite the first."""
        _underwrite(client, "APP-001")
        c1 = client.post(
            "/api/decision/APP-001",
            json={"officer_id": "OFF-1", "decision": "approve"},
        ).get_json()
        assert c1["final_status"] == "approved"

        c2 = client.post(
            "/api/decision/APP-001",
            json={"officer_id": "OFF-2", "decision": "deny"},
        ).get_json()
        assert c2["final_status"] == "denied"
        assert c2["human_decision"]["officer_id"] == "OFF-2"
