from functools import wraps
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SerializableModel(BaseModel):
    def to_dict(self) -> Dict[str, Any]:
        """Serializes the model to a dictionary compatible with standard JSON."""
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Deserializes and validates a model from a dictionary."""
        if hasattr(cls, "model_validate"):
            return cls.model_validate(data)
        return cls.parse_obj(data)


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
    application_id: str
    applicant_data: Dict[str, Any]
    documents: List[ExtractedDocument] = []
    kyc_output: Optional[Dict[str, Any]] = None
    credit_output: Optional[CreditRiskOutput] = None
    policy_output: Optional[PolicyCheckOutput] = None
    arbitrator_output: Optional[ArbitratorOutput] = None
    human_decision: Optional[HumanDecision] = None
    final_status: Optional[Literal["approved", "denied", "under_review"]] = None
    error_log: List[str] = []
    trace_id: Optional[str] = None
    state_version: str = Field(default="1.0.0")


def validate_state(func):
    """
    Decorator for agent nodes to validate state against the schema.
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
            LoanApplicationState.from_dict(merged_dict)
        except Exception as e:
            raise ValueError(
                f"Output updates from {func.__name__} violate state schema: {str(e)}"
            ) from e

        return result

    return wrapper
