"""
conftest.py — Shared pytest fixtures for RiskPilot integration tests (Member B — WI-5)

Provides:
  - Mock fixtures for KYC, Policy, and Arbitrator agent outputs
  - Parameterised base_application_state factory
  - Graph-level fixture that patches non-credit agents and runs real credit_node
  - Complete state builder for end-to-end test scenarios

These fixtures let Member B test the credit agent + guardrails in isolation
without depending on A's document parsing, C's RAG pipeline, or D's arbitrator.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.graph.state import (
    ArbitratorOutput,
    CreditRiskOutput,
    ExtractedDocument,
    HumanDecision,
    LoanApplicationState,
    PolicyCheckOutput,
)


# ---------------------------------------------------------------------------
# Primitive Mock Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_kyc_output() -> Dict[str, Any]:
    """Returns a valid, passing KYC output dict with verified income."""
    return {
        "kyc_status": "verified",
        "verified_fields": {
            "name": "Test Applicant",
            "income": 80000,
            "employer": "TestCorp",
        },
        "confidence": 0.92,
        "missing_critical_docs": False,
        "fraud_flag": False,
        "document_results": [
            {"document_type": "id_proof", "status": "valid", "confidence": 0.95},
            {"document_type": "pay_slip", "status": "valid", "confidence": 0.92},
            {"document_type": "bank_statement", "status": "valid", "confidence": 0.90},
        ],
    }


@pytest.fixture
def mock_kyc_output_fraud() -> Dict[str, Any]:
    """Returns a KYC output that flags a fraud signal (name mismatch)."""
    return {
        "kyc_status": "fraud_detected",
        "verified_fields": {},
        "confidence": 0.3,
        "missing_critical_docs": False,
        "fraud_flag": True,
        "fraud_reason": "Name mismatch: 'John Doe' on ID vs 'Jonathan Doe' on pay slip.",
        "document_results": [
            {"document_type": "id_proof", "status": "valid", "confidence": 0.95},
            {"document_type": "pay_slip", "status": "invalid", "confidence": 0.30},
        ],
    }


@pytest.fixture
def mock_kyc_output_missing_docs() -> Dict[str, Any]:
    """Returns a KYC output that flags missing critical documents."""
    return {
        "kyc_status": "incomplete",
        "verified_fields": {},
        "confidence": 0.5,
        "missing_critical_docs": True,
        "fraud_flag": False,
        "document_results": [
            {"document_type": "id_proof", "status": "valid", "confidence": 0.95},
            {"document_type": "pay_slip", "status": "valid", "confidence": 0.90},
        ],
    }


@pytest.fixture
def mock_policy_output_pass() -> PolicyCheckOutput:
    """Returns a PolicyCheckOutput that passes all checks."""
    return PolicyCheckOutput(
        policy_passed=True,
        violations=[],
        ltv_ratio=0.71,
        min_credit_requirement_met=True,
        max_dti_threshold=0.43,
        retrieved_policy_chunks=[
            "Minimum credit score for standard loan: 620.",
            "Maximum DTI ratio: 43% for conforming loans.",
            "Maximum LTV ratio: 80% for standard purchase loans.",
        ],
        reasoning="All policy checks passed. LTV 71% < 80% limit. DTI within threshold.",
    )


@pytest.fixture
def mock_policy_output_fail() -> PolicyCheckOutput:
    """Returns a PolicyCheckOutput with violations (high LTV + DTI)."""
    return PolicyCheckOutput(
        policy_passed=False,
        violations=[
            "DTI ratio 0.55 exceeds maximum allowed 0.43.",
            "LTV ratio 0.93 exceeds maximum allowed 0.80.",
        ],
        ltv_ratio=0.93,
        min_credit_requirement_met=False,
        max_dti_threshold=0.43,
        retrieved_policy_chunks=[
            "Maximum DTI ratio: 43% for conforming loans.",
            "Maximum LTV ratio: 80% for standard purchase loans.",
        ],
        reasoning="Policy FAILED. DTI and LTV both exceed thresholds.",
    )


@pytest.fixture
def mock_arbitrator_approve() -> ArbitratorOutput:
    """Returns an ArbitratorOutput recommending approval at high confidence."""
    return ArbitratorOutput(
        recommendation="approve",
        confidence_score=0.91,
        agent_agreement="unanimous",
        summary="All agents agree. Credit strong, policy passed, KYC verified. Recommend approval.",
        risk_flags=[],
    )


@pytest.fixture
def mock_arbitrator_deny() -> ArbitratorOutput:
    """Returns an ArbitratorOutput recommending denial at high confidence."""
    return ArbitratorOutput(
        recommendation="deny",
        confidence_score=0.88,
        agent_agreement="unanimous",
        summary="Multiple policy violations and high DTI. Deny.",
        risk_flags=["DTI exceeds limit", "LTV exceeds limit"],
    )


@pytest.fixture
def mock_arbitrator_conflict() -> ArbitratorOutput:
    """Returns an ArbitratorOutput signalling agent conflict → review required."""
    return ArbitratorOutput(
        recommendation="review_required",
        confidence_score=0.55,
        agent_agreement="conflict",
        summary="Credit score strong but policy violations detected. Human review required.",
        risk_flags=["Policy violation with strong credit score"],
    )


# ---------------------------------------------------------------------------
# State Factories
# ---------------------------------------------------------------------------

@pytest.fixture
def base_documents() -> list:
    """Three valid documents suitable for any standard test application."""
    return [
        ExtractedDocument(
            document_type="id_proof",
            extracted_text="ID: Test Applicant",
            validation_status="valid",
            confidence=0.95,
            extracted_fields={"name": "Test Applicant"},
        ),
        ExtractedDocument(
            document_type="pay_slip",
            extracted_text="Pay: $6,666/mo",
            validation_status="valid",
            confidence=0.92,
            extracted_fields={"income_monthly": 6666},
        ),
        ExtractedDocument(
            document_type="bank_statement",
            extracted_text="Deposit: $6,666/mo",
            validation_status="valid",
            confidence=0.90,
            extracted_fields={"income_monthly": 6666},
        ),
    ]


def make_application_state(
    income: float,
    monthly_debt: float,
    employment_months: int,
    loan_amount: float = 200000,
    property_value: float = 280000,
    app_id: str = "APP-TEST",
    kyc_output: Dict[str, Any] = None,
    documents: list = None,
) -> LoanApplicationState:
    """
    Factory function to build a fully valid LoanApplicationState.

    Useful in parametrise decorators and ad-hoc fixture construction.
    Not a fixture itself — call it directly.
    """
    docs = documents or []
    return LoanApplicationState(
        application_id=app_id,
        applicant_data={
            "name": "Test Applicant",
            "income": income,
            "monthly_debt": monthly_debt,
            "loan_amount": loan_amount,
            "property_value": property_value,
            "employment_months": employment_months,
        },
        documents=docs,
        kyc_output=kyc_output,
    )


@pytest.fixture
def state_clean_approval(mock_kyc_output, base_documents) -> LoanApplicationState:
    """Full state for Test Case 1: Clean Approval (Alice Johnson profile)."""
    return make_application_state(
        income=80000,
        monthly_debt=1200,
        employment_months=36,
        loan_amount=200000,
        property_value=280000,
        app_id="APP-001",
        kyc_output=mock_kyc_output,
        documents=base_documents,
    )


@pytest.fixture
def state_clean_denial(mock_kyc_output, base_documents) -> LoanApplicationState:
    """Full state for Test Case 2: Clean Denial (Bob Smith profile)."""
    return make_application_state(
        income=30000,
        monthly_debt=1375,
        employment_months=8,
        loan_amount=250000,
        property_value=270000,
        app_id="APP-002",
        kyc_output=mock_kyc_output,
        documents=base_documents,
    )


@pytest.fixture
def state_borderline(mock_kyc_output, base_documents) -> LoanApplicationState:
    """Full state for Test Case 3: Borderline (Charlie Brown profile)."""
    return make_application_state(
        income=60000,
        monthly_debt=1750,
        employment_months=24,
        loan_amount=205000,
        property_value=250000,
        app_id="APP-003",
        kyc_output=mock_kyc_output,
        documents=base_documents,
    )


@pytest.fixture
def state_policy_edge_case(mock_kyc_output, base_documents) -> LoanApplicationState:
    """Full state for Test Case 5: Policy Edge (Evan Wright — 11 months employment)."""
    return make_application_state(
        income=100000,
        monthly_debt=3166,
        employment_months=11,
        loan_amount=237000,
        property_value=300000,
        app_id="APP-005",
        kyc_output=mock_kyc_output,
        documents=base_documents,
    )


@pytest.fixture
def state_fraud_flag(mock_kyc_output_fraud) -> LoanApplicationState:
    """Full state for Test Case 6: Fraud Flag — name mismatch detected by KYC."""
    docs = [
        ExtractedDocument(
            document_type="id_proof",
            extracted_text="ID: John Doe",
            validation_status="valid",
            confidence=0.95,
            extracted_fields={"name": "John Doe"},
        ),
        ExtractedDocument(
            document_type="pay_slip",
            extracted_text="Name: Jonathan Doe, Pay: $7,000/mo",
            validation_status="invalid",
            confidence=0.30,
            extracted_fields={"name": "Jonathan Doe", "income_monthly": 7000},
        ),
        ExtractedDocument(
            document_type="bank_statement",
            extracted_text="Deposit: $7,000/mo",
            validation_status="valid",
            confidence=0.85,
            extracted_fields={"income_monthly": 7000},
        ),
    ]
    return LoanApplicationState(
        application_id="APP-006",
        applicant_data={
            "name": "John Doe",
            "income": 84000,
            "monthly_debt": 1500,
            "loan_amount": 220000,
            "property_value": 300000,
            "employment_months": 18,
        },
        documents=docs,
        kyc_output=mock_kyc_output_fraud,
    )
