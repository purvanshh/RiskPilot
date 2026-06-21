from dataclasses import dataclass
from typing import Dict, List, Optional

from src.rag.retriever import RetrievedPolicyChunk

# Sentinel returned when the evaluator cannot ground a violation in any
# retrieved chunk. The evaluator MUST NEVER fabricate document names,
# sections, or chunk numbers — grounded explanations originate only from
# the retrieval set.
NO_SUPPORTING_CITATION = "No supporting policy retrieved"


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


def _find_citation(chunks: List[RetrievedPolicyChunk], keywords: List[str]) -> str:
    """
    Returns the citation of the best-matching retrieved chunk by keyword
    overlap, or NO_SUPPORTING_CITATION when the retrieval set is empty or
    no chunk shares any keyword with the violation topic.

    Never invents a citation. The returned string is either:
      - the literal `citation()` of a chunk present in `chunks`, or
      - the NO_SUPPORTING_CITATION sentinel.
    """
    if not chunks:
        return NO_SUPPORTING_CITATION

    keyword_set = {word.lower() for word in keywords}
    best_chunk = None
    best_score = 0
    for chunk in chunks:
        text = chunk.text.lower()
        score = sum(1 for keyword in keyword_set if keyword in text)
        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_chunk is None or best_score == 0:
        return NO_SUPPORTING_CITATION
    return best_chunk.citation()


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
        citation = _find_citation(policy_chunks, ["credit", "score", "minimum", "650"])
        violations.append(
            f"Credit score {credit_score} is below the minimum requirement of 650. Citation: {citation}"
        )

    if dti > 0.50:
        citation = _find_citation(policy_chunks, ["dti", "50%", "hard cap"])
        violations.append(
            f"DTI ratio {dti:.2%} exceeds the hard maximum of 50%. Citation: {citation}"
        )
    elif dti > max_dti_threshold:
        citation = _find_citation(policy_chunks, ["dti", "45%", "standard"])
        violations.append(
            f"DTI ratio {dti:.2%} exceeds the standard maximum threshold of 45%. Citation: {citation}"
        )

    if ltv > 0.85:
        citation = _find_citation(policy_chunks, ["ltv", "85%", "hard"])
        violations.append(
            f"LTV ratio {ltv:.2%} exceeds the hard maximum of 85%. Citation: {citation}"
        )
    elif ltv > 0.80:
        if credit_score <= 720 or dti >= 0.35:
            citation = _find_citation(policy_chunks, ["ltv", "85%", "exceptions"])
            violations.append(
                f"LTV ratio {ltv:.2%} exceeds the standard threshold of 80% and exception conditions are not met. Citation: {citation}"
            )

    if employment_months < 6:
        citation = _find_citation(policy_chunks, ["employment", "6 months", "denial"])
        violations.append(
            f"Employment tenure of {employment_months} months is below the hard minimum of 6 months. Citation: {citation}"
        )
    elif employment_months < 12:
        citation = _find_citation(policy_chunks, ["employment", "12 months", "review"])
        violations.append(
            f"Employment tenure of {employment_months} months is below the standard requirement of 12 months. Citation: {citation}"
        )

    if income_document_count is not None and income_document_count < 2:
        citation = _find_citation(
            policy_chunks, ["income", "documents", "bank statement", "pay slip"]
        )
        violations.append(
            f"Income verification requires two documents, but only {income_document_count} were available. Citation: {citation}"
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
