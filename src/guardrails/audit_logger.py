"""
audit_logger.py — Structured Audit Logger for RiskPilot Guardrails

Writes every agent output, guardrail flag, and final decision to
logs/audit.jsonl in JSON Lines format for full audit trail compliance.

PRD §8.3: "All agent outputs, retrievals, and decisions logged."
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default log location — can be overridden via RISKPILOT_AUDIT_LOG env var
_DEFAULT_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "audit.jsonl"


def _get_log_path() -> Path:
    """Resolves the audit log file path (env override supported)."""
    env_path = os.environ.get("RISKPILOT_AUDIT_LOG")
    return Path(env_path) if env_path else _DEFAULT_LOG_PATH


def _write_entry(entry: Dict[str, Any]) -> None:
    """Appends a single JSON entry to the audit log file."""
    log_path = _get_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:
        logger.error(f"[AuditLogger] Failed to write audit entry: {e}")


def _now_iso() -> str:
    """Returns current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class AuditLogger:
    """
    Structured audit logger for the RiskPilot multi-agent system.

    Usage:
        audit = AuditLogger(application_id="APP-001")
        audit.log_agent_output("credit", credit_output.to_dict())
        audit.log_guardrail_flag("output", "DTI exceeds 60% hard stop")
        audit.log_decision("under_review", ["Low confidence", "DTI hard stop"])
    """

    def __init__(self, application_id: str, trace_id: Optional[str] = None):
        self.application_id = application_id
        self.trace_id = trace_id

    def _base_entry(self, event_type: str) -> Dict[str, Any]:
        return {
            "timestamp": _now_iso(),
            "application_id": self.application_id,
            "trace_id": self.trace_id,
            "event_type": event_type,
        }

    def log_agent_output(self, agent_name: str, output: Dict[str, Any]) -> None:
        """
        Logs a raw agent output dict to the audit trail.

        Args:
            agent_name: One of 'kyc', 'credit', 'policy', 'arbitrator'
            output: The agent's output dict (Pydantic .to_dict() is fine)
        """
        entry = self._base_entry("agent_output")
        entry["agent"] = agent_name
        entry["output"] = output
        _write_entry(entry)
        logger.debug(f"[AuditLogger] Logged {agent_name} output for {self.application_id}")

    def log_guardrail_flag(self, guardrail_type: str, message: str) -> None:
        """
        Logs a guardrail flag (input or output violation).

        Args:
            guardrail_type: 'input' or 'output'
            message: Human-readable description of the flag
        """
        entry = self._base_entry("guardrail_flag")
        entry["guardrail_type"] = guardrail_type
        entry["message"] = message
        _write_entry(entry)
        logger.warning(f"[AuditLogger] Guardrail [{guardrail_type}] flagged: {message}")

    def log_decision(
        self,
        decision: str,
        flags: List[str],
        officer_id: Optional[str] = None,
        override_reason: Optional[str] = None,
    ) -> None:
        """
        Logs the final system or human decision with all active flags.

        Args:
            decision: 'approved', 'denied', 'under_review', 'review_required'
            flags: List of active guardrail/risk flag messages
            officer_id: Loan officer ID if a human decision was made
            override_reason: Reason string if officer used override
        """
        entry = self._base_entry("decision")
        entry["decision"] = decision
        entry["flags"] = flags
        entry["officer_id"] = officer_id
        entry["override_reason"] = override_reason
        _write_entry(entry)
        logger.info(
            f"[AuditLogger] Decision logged: {decision} | flags={len(flags)} "
            f"| officer={officer_id or 'system'}"
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions (for quick use without instantiating)
# ---------------------------------------------------------------------------

def log_agent_output(application_id: str, agent_name: str, output: Dict[str, Any]) -> None:
    """Convenience wrapper: log a single agent output without an AuditLogger instance."""
    AuditLogger(application_id).log_agent_output(agent_name, output)


def log_guardrail_flag(application_id: str, guardrail_type: str, message: str) -> None:
    """Convenience wrapper: log a guardrail flag without an AuditLogger instance."""
    AuditLogger(application_id).log_guardrail_flag(guardrail_type, message)


def log_decision(application_id: str, decision: str, flags: List[str]) -> None:
    """Convenience wrapper: log a decision without an AuditLogger instance."""
    AuditLogger(application_id).log_decision(decision, flags)
