from dataclasses import dataclass
from typing import List, Optional

from src.rag.retriever import RetrievedPolicyChunk


@dataclass(frozen=True)
class PolicyEvaluationResult:
    policy_passed: bool
    violations: List[str]
    ltv_ratio: float
    min_credit_requirement_met: bool
    max_dti_threshold: float
    retrieved_policy_chunks: List[str]
    retrieved_sources: List[str]
    citations: List[str]
    reasoning: str
    confidence: float


def _compute_sources(chunks: List[RetrievedPolicyChunk]) -> List[str]:
    sources = []
    for chunk in chunks:
        document = chunk.metadata.get("document") or chunk.metadata.get("source_file")
        if document and document not in sources:
            sources.append(document)
    return sources


def _find_citation(chunks: List[RetrievedPolicyChunk], keywords: List[str], default: str) -> str:
    keyword_set = {word.lower() for word in keywords}
    for chunk in chunks:
        text = chunk.text.lower()
        if all(keyword in text for keyword in keyword_set):
            return chunk.citation()
    return default


def evaluate_policy(
    credit_score: int,
    dti: float,
    ltv: float,
    employment_months: int,
    policy_chunks: List[RetrievedPolicyChunk],
    income_document_count: Optional[int] = None,
) -> PolicyEvaluationResult:
    violations: List[str] = []
    min_credit_met = credit_score >= 650
    max_dti_threshold = 0.45

    if credit_score < 650:
        citation = _find_citation(
            policy_chunks,
            ["credit", "score", "minimum", "650"],
            "Credit Policy | Minimum Credit Requirement | chunk 0",
        )
        violations.append(
            f"Credit score {credit_score} is below the minimum requirement of 650. "
            f"Citation: {citation}"
        )

    if dti > 0.50:
        citation = _find_citation(
            policy_chunks,
            ["dti", "50%", "hard cap"],
            "DTI Policy | Hard Cap DTI | chunk 0",
        )
        violations.append(
            f"DTI ratio {dti:.2%} exceeds the hard maximum of 50%. Citation: {citation}"
        )
    elif dti > max_dti_threshold:
        citation = _find_citation(
            policy_chunks,
            ["dti", "45%", "standard"],
            "DTI Policy | Maximum DTI Thresholds | chunk 0",
        )
        violations.append(
            f"DTI ratio {dti:.2%} exceeds the standard maximum threshold of 45%. "
            f"Citation: {citation}"
        )

    if ltv > 0.85:
        citation = _find_citation(
            policy_chunks,
            ["ltv", "85%", "hard"],
            "LTV Policy | Hard Reject | chunk 0",
        )
        violations.append(
            f"LTV ratio {ltv:.2%} exceeds the hard maximum of 85%. Citation: {citation}"
        )
    elif ltv > 0.80:
        if credit_score <= 720 or dti >= 0.35:
            citation = _find_citation(
                policy_chunks,
                ["ltv", "85%", "exceptions"],
                "LTV Policy | Exceptions | chunk 0",
            )
            violations.append(
                f"LTV ratio {ltv:.2%} exceeds the standard threshold of 80%"
                f" and exception conditions are not met. Citation: {citation}"
            )

    if employment_months < 6:
        citation = _find_citation(
            policy_chunks,
            ["employment", "6 months", "denial"],
            "Employment Stability Policy | Direct Denial | chunk 0",
        )
        violations.append(
            f"Employment tenure of {employment_months} months is below"
            f" the hard minimum of 6 months. Citation: {citation}"
        )
    elif employment_months < 12:
        citation = _find_citation(
            policy_chunks,
            ["employment", "12 months", "review"],
            "Employment Stability Policy | Minimum Stability | chunk 0",
        )
        violations.append(
            f"Employment tenure of {employment_months} months is below"
            f" the standard requirement of 12 months. Citation: {citation}"
        )

    if income_document_count is not None and income_document_count < 2:
        citation = _find_citation(
            policy_chunks,
            ["income", "documents", "bank statement", "pay slip"],
            "Income Verification Policy | Documentation Requirements | chunk 0",
        )
        violations.append(
            f"Income verification requires two documents, but only"
            f" {income_document_count} were available. Citation: {citation}"
        )

    policy_passed = len(violations) == 0
    confidence = 0.92 if policy_passed else max(0.45, 0.8 - 0.05 * len(violations))
    sources = _compute_sources(policy_chunks)
    citations = [chunk.citation() for chunk in policy_chunks[: min(len(policy_chunks), 5)]]
    reasoning = (
        f"Evaluated credit score {credit_score}, DTI {dti:.2%}, LTV {ltv:.2%}, "
        f"employment months {employment_months}, income docs {income_document_count}. "
        f"Result: {'PASSED' if policy_passed else 'FAILED'}."
    )

    return PolicyEvaluationResult(
        policy_passed=policy_passed,
        violations=violations,
        ltv_ratio=ltv,
        min_credit_requirement_met=min_credit_met,
        max_dti_threshold=max_dti_threshold,
        retrieved_policy_chunks=[chunk.text for chunk in policy_chunks[:5]],
        retrieved_sources=sources,
        citations=citations,
        reasoning=reasoning,
        confidence=confidence,
    )
