from typing import Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field

class ExtractedDocument(BaseModel):
    document_type: Literal["id_proof", "bank_statement", "pay_slip", "employment_letter"]
    extracted_text: str
    validation_status: Literal["valid", "invalid", "needs_review"]
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_fields: Dict[str, Any]  # e.g., {"name": "John Doe", "income": 50000}

class CreditRiskOutput(BaseModel):
    credit_score: int = Field(ge=300, le=850)
    risk_category: Literal["low", "medium", "high", "very_high"]
    dti_ratio: float = Field(ge=0.0, le=1.0)
    default_probability: float = Field(ge=0.0, le=1.0)
    reasoning: str

class PolicyCheckOutput(BaseModel):
    policy_passed: bool
    violations: List[str] = []
    ltv_ratio: Optional[float] = None
    min_credit_requirement_met: bool
    max_dti_threshold: float
    retrieved_policy_chunks: List[str]  # RAG evidence
    reasoning: str

class ArbitratorOutput(BaseModel):
    recommendation: Literal["approve", "deny", "review_required"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    agent_agreement: Literal["unanimous", "partial", "conflict"]
    summary: str
    risk_flags: List[str] = []

class HumanDecision(BaseModel):
    officer_id: str
    decision: Literal["approve", "deny", "override_approve", "override_deny"]
    override_reason: Optional[str] = None
    timestamp: str

class LoanApplicationState(BaseModel):
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
