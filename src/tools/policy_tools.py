from typing import List, Dict, Any

def ltv_calculator(loan_amount: float, property_value: float) -> float:
    """
    Computes the Loan-to-Value (LTV) ratio.
    """
    if property_value <= 0:
        return 0.0
    return loan_amount / property_value

def policy_retriever(query: str) -> List[str]:
    """
    Mock RAG retriever. Looks up policy guidelines by simple keyword matching
    to simulate semantic search from ChromaDB policy documents.
    """
    # Simple hardcoded chunks representing the markdown files in data/policy_docs/
    policy_chunks = [
        "Credit Score Policy: The absolute minimum credit score required for auto-approval is 650. Any score below 650 must be marked as review_required or deny.",
        "Credit Score Policy Tiers: Tier 1 (Low Risk) >= 720, Tier 2 (Medium Risk) 650-719, Tier 3 (High Risk) 600-649, Tier 4 (Unacceptable Risk) < 600.",
        "Debt-to-Income (DTI) Policy: The maximum DTI allowed for standard approvals is 45% (0.45). Under no circumstances shall an applicant be approved with a DTI exceeding 50% (0.50).",
        "DTI Borderline Policy: DTI ratios between 40% and 45% are considered borderline and should prompt the Arbitrator agent to flag for manual review.",
        "Loan-to-Value (LTV) Policy: The maximum allowable Loan-to-Value (LTV) ratio for standard mortgage/property-backed loans is 80% (0.80).",
        "LTV Exception Policy: LTV ratios up to 85% (0.85) may be approved only if the applicant's credit score is greater than 720 and their DTI is under 35%. Any LTV exceeding 85% is a hard policy violation.",
        "Employment Stability Policy: Applicants must have a minimum of 12 months of continuous employment history with their current employer.",
        "Employment Policy Exceptions: Less than 12 months (e.g., 6-11 months) may be accepted if the applicant was in a similar role previously with no gaps, subject to manual review. Less than 6 months is direct denial.",
        "Income Verification Policy: To verify income, the applicant must provide at least two documents: bank statement showing payroll deposits, pay slip, or employment letter."
    ]
    
    query_words = query.lower().split()
    matched_chunks = []
    
    for chunk in policy_chunks:
        # Simple scoring based on word match
        score = sum(1 for word in query_words if word in chunk.lower())
        if score > 0:
            matched_chunks.append((score, chunk))
            
    # Sort by score descending and return top 3 chunks
    matched_chunks.sort(key=lambda x: x[0], reverse=True)
    results = [chunk for _, chunk in matched_chunks]
    
    return results[:3] if results else policy_chunks[:3]

def policy_validator(credit_score: int, dti: float, ltv: float, employment_months: int, policy_chunks: List[str]) -> Dict[str, Any]:
    """
    Checks the applicant parameters against rules. Under the hood, this simulates
    an LLM comparing the loan details with retrieved policy chunks.
    """
    passed = True
    violations = []
    min_credit_met = True
    max_dti_threshold = 0.45

    # 1. Credit Score Policy Check
    if credit_score < 650:
        min_credit_met = False
        passed = False
        violations.append(f"Credit score {credit_score} is below the minimum required 650.")

    # 2. DTI Policy Check
    if dti > 0.50:
        passed = False
        violations.append(f"DTI ratio {dti:.2%} exceeds the absolute maximum threshold of 50%.")
    elif dti > 0.45:
        passed = False
        violations.append(f"DTI ratio {dti:.2%} exceeds the standard maximum threshold of 45%.")

    # 3. LTV Policy Check
    if ltv > 0.85:
        passed = False
        violations.append(f"LTV ratio {ltv:.2%} exceeds the absolute maximum threshold of 85%.")
    elif ltv > 0.80:
        # Exception check: score > 720 and DTI < 35%
        if credit_score <= 720 or dti >= 0.35:
            passed = False
            violations.append(f"LTV ratio {ltv:.2%} exceeds the standard 80% (exception conditions not met: score={credit_score}, DTI={dti:.2%}).")

    # 4. Employment Stability Check
    if employment_months < 12:
        # Less than 12 months is a policy violation unless reviewed (meaning it fails auto-pass)
        passed = False
        violations.append(f"Employment tenure of {employment_months} months is below the standard requirement of 12 months.")

    reasoning = (
        f"Policy check status: {'PASSED' if passed else 'FAILED'}. "
        f"Parameters checked: Credit Score = {credit_score} (Min Required: 650); "
        f"DTI = {dti:.2%} (Max Allowed: 45%); "
        f"LTV = {ltv:.2%} (Max Allowed: 80%); "
        f"Employment = {employment_months} months (Min Required: 12)."
    )
    if violations:
        reasoning += f" Violations: {'; '.join(violations)}."

    return {
        "passed": passed,
        "violations": violations,
        "min_credit_met": min_credit_met,
        "max_dti_threshold": max_dti_threshold,
        "reasoning": reasoning
    }
