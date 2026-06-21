"""
Retrieval-quality tests for the Policy RAG pipeline.

Verifies that natural-language policy queries route to the correct source
document via the canonical `PolicyRetriever`. Kept intentionally simple —
no evaluation framework, no LLM judging — just "did the expected file
appear at the top of the retrieved results?".

The fixture builds the retriever once per module so we only pay the
sentence-transformer load and Chroma indexing cost once.
"""

import pytest

from src.rag.evaluator import NO_SUPPORTING_CITATION, evaluate_policy
from src.rag.retriever import PolicyRetriever

# (natural-language query, expected source_file)
QUERY_CASES = [
    ("minimum credit score required for loan approval", "credit_policy.md"),
    ("maximum debt-to-income ratio allowed", "dti_policy.md"),
    ("maximum loan-to-value ratio", "ltv_policy.md"),
    ("employment tenure requirements for borrowers", "employment_policy.md"),
    ("income verification documents required", "income_verification.md"),
    ("general exceptions and manual override authority", "general_guidelines.md"),
]


@pytest.fixture(scope="module")
def retriever() -> PolicyRetriever:
    return PolicyRetriever()


@pytest.mark.parametrize("query,expected_file", QUERY_CASES)
def test_expected_doc_appears_in_top_k(retriever, query, expected_file):
    """The expected source document must appear among the top-k retrieved chunks."""
    chunks = retriever.retrieve(query)
    assert chunks, f"Query returned no chunks: {query!r}"
    sources = [c.metadata.get("source_file") for c in chunks]
    assert (
        expected_file in sources
    ), f"Query {query!r}: expected {expected_file} in top-{len(chunks)}, got {sources}"


def test_precision_at_1_meets_target(retriever):
    """Aggregate Precision@1 across all query domains must be ≥ 0.80."""
    correct = 0
    misses = []
    for query, expected_file in QUERY_CASES:
        chunks = retriever.retrieve(query)
        top_source = chunks[0].metadata.get("source_file") if chunks else None
        if top_source == expected_file:
            correct += 1
        else:
            misses.append((query, expected_file, top_source))

    precision_at_1 = correct / len(QUERY_CASES)
    assert precision_at_1 >= 0.80, (
        f"Precision@1 = {precision_at_1:.2f} (target ≥ 0.80). " f"Misses: {misses}"
    )


def test_citations_originate_from_retrieved_chunks(retriever):
    """
    Every non-sentinel citation produced by the evaluator must reference a
    chunk that was actually retrieved. The evaluator must never fabricate
    document names, sections, or chunk numbers.
    """
    # Pull a broad retrieval set so the evaluator has plausible context.
    chunks = []
    for query, _ in QUERY_CASES:
        chunks.extend(retriever.retrieve(query))

    # Trigger every violation branch.
    result = evaluate_policy(
        credit_score=500,
        dti=0.60,
        ltv=0.95,
        employment_months=3,
        policy_chunks=chunks,
        income_document_count=1,
    )

    real_citations = {c.citation() for c in chunks}
    for violation in result.violations:
        assert "Citation: " in violation, f"Violation missing citation: {violation}"
        cited = violation.split("Citation: ", 1)[1].rstrip(".")
        assert cited in real_citations or cited == NO_SUPPORTING_CITATION, (
            f"Fabricated citation detected: {cited!r} not in retrieved set "
            f"and not the no-supporting sentinel"
        )


def test_no_supporting_sentinel_when_retrieval_empty():
    """With zero retrieved chunks, every violation must carry the sentinel."""
    result = evaluate_policy(
        credit_score=500,
        dti=0.60,
        ltv=0.95,
        employment_months=3,
        policy_chunks=[],
        income_document_count=1,
    )
    for violation in result.violations:
        assert (
            NO_SUPPORTING_CITATION in violation
        ), f"Empty retrieval should yield no-supporting sentinel, got: {violation}"
