import logging
from typing import List

from src.graph.state import LoanApplicationState, PolicyCheckOutput
from src.rag.evaluator import evaluate_policy
from src.rag.retriever import PolicyRetriever, RetrievedPolicyChunk

logger = logging.getLogger(__name__)

DEFAULT_QUERIES = [
    "Minimum credit score requirement for loan approval.",
    "Maximum debt-to-income ratio allowed for standard approvals.",
    "Maximum loan-to-value ratio for mortgage loans.",
    "Employment tenure requirements for underwriting.",
    "Income verification documentation requirements.",
    "General loan eligibility and underwriting guidelines.",
]


def _deduplicate_chunks(
    chunks: List[RetrievedPolicyChunk],
) -> List[RetrievedPolicyChunk]:
    seen_texts = set()
    unique_chunks: List[RetrievedPolicyChunk] = []
    for chunk in chunks:
        normalized = chunk.text.strip()
        if normalized not in seen_texts:
            seen_texts.add(normalized)
            unique_chunks.append(chunk)
    return unique_chunks


def run_policy_check(state: LoanApplicationState) -> PolicyCheckOutput:
    loan_amount = state.applicant_data.get("loan_amount", 0.0)
    property_value = state.applicant_data.get("property_value", 1.0)
    employment_months = state.applicant_data.get("employment_months", 0)
    income_document_count = sum(
        1
        for doc in (state.documents or [])
        if doc.document_type in {"bank_statement", "pay_slip", "employment_letter"}
    )

    credit_score = 300
    dti = 0.0
    if state.credit_output is not None:
        credit_score = state.credit_output.credit_score
        dti = state.credit_output.dti_ratio

    retriever = PolicyRetriever()
    retrieved_chunks: List[RetrievedPolicyChunk] = []
    for query in DEFAULT_QUERIES:
        retrieved_chunks.extend(retriever.retrieve(query))

    retrieved_chunks = _deduplicate_chunks(retrieved_chunks)
    validation = evaluate_policy(
        credit_score=credit_score,
        dti=dti,
        ltv=loan_amount / property_value if property_value > 0 else 0.0,
        employment_months=employment_months,
        policy_chunks=retrieved_chunks,
        income_document_count=income_document_count,
    )

    return PolicyCheckOutput(
        policy_passed=validation.policy_passed,
        violations=validation.violations,
        ltv_ratio=round(validation.ltv_ratio, 4),
        min_credit_requirement_met=validation.min_credit_requirement_met,
        max_dti_threshold=validation.max_dti_threshold,
        retrieved_policy_chunks=validation.retrieved_policy_chunks,
        retrieved_sources=validation.retrieved_sources,
        citations=validation.citations,
        reasoning=validation.reasoning,
        confidence=validation.confidence,
    )
