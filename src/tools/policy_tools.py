"""
Compatibility adapter for the legacy policy-tools surface.

The canonical RAG implementation lives in `src.rag.retriever`,
`src.rag.evaluator`, and `src.rag.policy_agent`. This module retains the
historical public function names so any teammate/external code that imports
from `src.tools.policy_tools` keeps working, but contains no business logic
of its own — every call is delegated to the canonical implementation.
"""

from typing import Any, Dict, List, Optional

from src.rag.evaluator import evaluate_policy
from src.rag.retriever import PolicyRetriever, RetrievedPolicyChunk

__all__ = [
    "ltv_calculator",
    "policy_retriever",
    "policy_validator",
    "RetrievedPolicyChunk",
]


def ltv_calculator(loan_amount: float, property_value: float) -> float:
    """Pure utility — computes the loan-to-value ratio. No policy logic."""
    if property_value <= 0:
        return 0.0
    return loan_amount / property_value


def policy_retriever(
    query: str,
    top_k: int = 3,
    similarity_threshold: float = 0.5,
    policy_docs_dir: str = "./data/policy_docs",
    persist_dir: str = "./data/chroma_db",
    collection_name: str = "lending_policy",
) -> List[RetrievedPolicyChunk]:
    """Adapter — delegates to the canonical `PolicyRetriever`."""
    retriever = PolicyRetriever(
        policy_docs_dir=policy_docs_dir,
        persist_dir=persist_dir,
        collection_name=collection_name,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )
    return retriever.retrieve(query)


def policy_validator(
    credit_score: int,
    dti: float,
    ltv: float,
    employment_months: int,
    policy_chunks: List[RetrievedPolicyChunk],
    income_document_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Adapter — delegates evaluation to `evaluate_policy` and reshapes the
    result into the historical dict shape expected by legacy callers:
    {passed, violations, min_credit_met, max_dti_threshold, reasoning}.

    For grounded outputs (citations, sources, confidence), call
    `src.rag.evaluator.evaluate_policy` directly.
    """
    result = evaluate_policy(
        credit_score=credit_score,
        dti=dti,
        ltv=ltv,
        employment_months=employment_months,
        policy_chunks=policy_chunks,
        income_document_count=income_document_count,
    )
    return {
        "passed": result.policy_passed,
        "violations": result.violations,
        "min_credit_met": result.min_credit_requirement_met,
        "max_dti_threshold": result.max_dti_threshold,
        "reasoning": result.reasoning,
    }
