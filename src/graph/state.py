import concurrent.futures
import logging
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SerializableModel(BaseModel):
    def to_dict(self) -> Dict[str, Any]:
        """Serializes the model to a dictionary compatible with standard JSON."""
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()

    def state_to_dict(self) -> Dict[str, Any]:
        """Explicit alias of to_dict() for PRD compliance and audit trail exports."""
        return self.to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Deserializes and validates a model from a dictionary."""
        if hasattr(cls, "model_validate"):
            return cls.model_validate(data)
        return cls.parse_obj(data)

    @classmethod
    def validate_state_dict(cls, data: Dict[str, Any]) -> bool:
        """
        Validates a raw dictionary against this model's schema.
        Returns True if valid, raises ValidationError if not.
        Useful in tests and guardrails without constructing a full object.
        """
        cls.from_dict(data)
        return True


class ExtractedDocument(SerializableModel):
    document_type: Literal["id_proof", "bank_statement", "pay_slip", "employment_letter"]
    extracted_text: str
    validation_status: Literal["valid", "invalid", "needs_review"]
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_fields: Dict[str, Any]  # e.g., {"name": "John Doe", "income": 50000}


class CreditRiskOutput(SerializableModel):
    credit_score: int = Field(ge=300, le=850)
    risk_category: Literal["low", "medium", "high", "very_high"]
    dti_ratio: float = Field(ge=0.0, le=1.0)
    default_probability: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the credit assessment (0-1). <0.6 triggers human review.",
    )
    reasoning: str


class PolicyCheckOutput(SerializableModel):
    policy_passed: bool
    violations: List[str] = []
    ltv_ratio: Optional[float] = None
    min_credit_requirement_met: bool
    max_dti_threshold: float
    retrieved_policy_chunks: List[str]  # RAG evidence
    reasoning: str


class ArbitratorOutput(SerializableModel):
    recommendation: Literal["approve", "deny", "review_required"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    agent_agreement: Literal["unanimous", "partial", "conflict"]
    summary: str
    risk_flags: List[str] = []


class HumanDecision(SerializableModel):
    officer_id: str
    decision: Literal["approve", "deny", "override_approve", "override_deny"]
    override_reason: Optional[str] = None
    timestamp: str


class LoanApplicationState(SerializableModel):
    # --- Required Identity Fields ---
    application_id: str = Field(description="Unique identifier for the loan application.")
    trace_id: Optional[str] = Field(
        default=None,
        description="LangSmith / observability trace ID for end-to-end run tracking.",
    )

    # --- Applicant & Document Data ---
    applicant_data: Dict[str, Any]
    documents: List[ExtractedDocument] = []

    # --- Agent Outputs (populated sequentially by each node) ---
    kyc_output: Optional[Dict[str, Any]] = None
    credit_output: Optional[CreditRiskOutput] = None
    policy_output: Optional[PolicyCheckOutput] = None
    arbitrator_output: Optional[ArbitratorOutput] = None

    # --- Decision Fields ---
    human_decision: Optional[HumanDecision] = None
    final_status: Optional[Literal["approved", "denied", "under_review"]] = None

    # --- Audit & Metadata ---
    error_log: List[str] = Field(
        default_factory=list,
        description="Captures errors and warnings from every agent node.",
    )
    state_version: str = Field(
        default="1.0.0",
        description="Schema version for forward-compatibility checks.",
    )
    updated_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp of the last state mutation, used in audit logs.",
    )

    def stamp(self) -> "LoanApplicationState":
        """Returns a copy of the state with updated_at set to now (UTC)."""
        return self.model_copy(update={"updated_at": datetime.now(timezone.utc).isoformat()})


def validate_state(func):
    """
    Decorator for agent nodes to validate state against the schema.
    - Accepts both dict and LoanApplicationState inputs.
    - Validates output dict can be merged back into a valid state.
    - Stamps updated_at on every successful node execution.
    """

    @wraps(func)
    def wrapper(state: Any, *args, **kwargs):
        # Validate Input
        if isinstance(state, dict):
            try:
                validated_state = LoanApplicationState.from_dict(state)
            except Exception as e:
                raise ValueError(f"Input to {func.__name__} violates state schema: {str(e)}") from e
        elif isinstance(state, LoanApplicationState):
            validated_state = state
        else:
            raise TypeError(
                f"State input to {func.__name__} must be a dict or LoanApplicationState."
            )

        # Run Node Function
        result = func(validated_state, *args, **kwargs)

        # Validate Output
        if not isinstance(result, dict):
            raise TypeError(f"Node function {func.__name__} must return a dictionary of updates.")

        try:
            # Check schema validation after applying the updates
            merged_dict = validated_state.to_dict()
            merged_dict.update(result)
            # Stamp updated_at on every validated output
            merged_dict["updated_at"] = datetime.now(timezone.utc).isoformat()
            LoanApplicationState.from_dict(merged_dict)
        except Exception as e:
            raise ValueError(
                f"Output updates from {func.__name__} violate state schema: {str(e)}"
            ) from e

        return result

    return wrapper


def timeout_resilience(seconds: float = 30.0):
    """
    Decorator to enforce a timeout on node execution.
    Raises TimeoutError if execution takes longer than specified seconds.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError as e:
                    raise TimeoutError(
                        f"Agent node '{func.__name__}' timed out after {seconds} seconds."
                    ) from e

        return wrapper

    return decorator


def graceful_fallback(fallback_type: str):
    """
    Decorator to catch any exceptions (including TimeoutError) during node execution,
    log the error to state.error_log, and return a conservative fallback update dictionary.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(state: Any, *args, **kwargs):
            error_log = list(state.error_log) if hasattr(state, "error_log") else []
            try:
                return func(state, *args, **kwargs)
            except Exception as e:
                # Propagate validation errors meant to halt ingestion
                if isinstance(e, ValueError) and (
                    "Validation Error" in str(e) or "No documents" in str(e)
                ):
                    raise e

                logger.error(f"Error in node {func.__name__}: {str(e)}", exc_info=True)
                error_log.append(f"Node '{func.__name__}' error: {str(e)}")

                if fallback_type == "kyc":
                    kyc_output = {
                        "status": "failed",
                        "missing_critical_docs": True,
                        "missing_docs_list": ["id_proof", "bank_statement", "pay_slip"],
                        "fraud_flag": False,
                        "confidence": 0.0,
                        "verified_fields": {
                            "name": (
                                state.applicant_data.get("name") if state.applicant_data else None
                            ),
                            "income": (
                                state.applicant_data.get("income") if state.applicant_data else None
                            ),
                            "employer": None,
                        },
                    }
                    return {
                        "kyc_output": kyc_output,
                        "error_log": error_log,
                    }
                elif fallback_type == "credit":
                    credit_result = CreditRiskOutput(
                        credit_score=300,
                        risk_category="very_high",
                        dti_ratio=1.0,
                        default_probability=1.0,
                        confidence_score=0.0,
                        reasoning=(
                            f"Assessment failed due to error: {str(e)}. "
                            "Fallback values assigned."
                        ),
                    )
                    return {"credit_output": credit_result, "error_log": error_log}
                elif fallback_type == "policy":
                    policy_result = PolicyCheckOutput(
                        policy_passed=False,
                        violations=[f"System error in policy checking: {str(e)}"],
                        ltv_ratio=0.0,
                        min_credit_requirement_met=False,
                        max_dti_threshold=0.45,
                        retrieved_policy_chunks=[],
                        reasoning=f"System error: {str(e)}",
                    )
                    return {"policy_output": policy_result, "error_log": error_log}
                elif fallback_type == "arbitrator":
                    arbitrator_output = ArbitratorOutput(
                        recommendation="review_required",
                        confidence_score=0.0,
                        agent_agreement="conflict",
                        summary=f"Arbitration failed due to system error: {str(e)}",
                        risk_flags=[f"System error in arbitration: {str(e)}"],
                    )
                    return {
                        "arbitrator_output": arbitrator_output,
                        "error_log": error_log,
                    }
                else:
                    return {"error_log": error_log}

        return wrapper

    return decorator
