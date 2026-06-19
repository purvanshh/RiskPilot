from typing import Literal


def calculate_credit_score(
    income: float,
    monthly_debt: float,
    employment_months: int,
    extracted_credit_score: int = None,
) -> int:
    """
    Computes a deterministic credit score based on income, debt, and employment history.

    Formula (matches PRD):
        dti = monthly_debt / (income / 12)
        base_score = 300 + (income / 1000) * 10 + employment_months * 2 - dti * 200
        If extracted credit score from credit report is provided, blends them:
            blended = 0.7 * base_score + 0.3 * extracted_credit_score

    Returns: int in range [300, 850]
    """
    # Guard against division by zero
    monthly_income = max(1.0, income / 12.0)
    dti = monthly_debt / monthly_income

    # Calculate base score using PRD formula
    base_score = 300 + (income / 1000.0) * 10.0 + employment_months * 2.0 - dti * 200.0

    if extracted_credit_score is not None:
        base_score = 0.7 * base_score + 0.3 * extracted_credit_score

    # Boundary constraints: 300 to 850
    return min(850, max(300, int(base_score)))


def dti_calculator(monthly_debt: float, annual_income: float) -> float:
    """
    Computes Debt-to-Income (DTI) ratio.
        DTI = monthly_debt / (annual_income / 12)
    Returns: float [0.0, ∞)
    """
    monthly_income = max(1.0, annual_income / 12.0)
    return monthly_debt / monthly_income


def risk_classifier(credit_score: int) -> Literal["low", "medium", "high", "very_high"]:
    """
    Maps credit score to a risk category per PRD thresholds:
        >= 720  → low
        >= 650  → medium
        >= 580  → high
        <  580  → very_high
    """
    if credit_score >= 720:
        return "low"
    elif credit_score >= 650:
        return "medium"
    elif credit_score >= 580:
        return "high"
    else:
        return "very_high"


def calculate_default_probability(credit_score: int) -> float:
    """
    Computes probability of default as a linear mapping from score to probability.
        score=300 → prob=1.0 (certain default)
        score=850 → prob=0.0 (no default)

    Formula: 1 - ((score - 300) / 550)
    Returns: float in [0.0, 1.0]
    """
    prob = 1.0 - ((credit_score - 300) / 550.0)
    return round(min(1.0, max(0.0, prob)), 4)


def calculate_confidence_score(
    income: float,
    monthly_debt: float,
    employment_months: int,
) -> float:
    """
    Computes a confidence score [0.0, 1.0] for the credit assessment based on
    the quality and completeness of input data:

    Scoring factors:
        - income > 0           → +0.35 (primary income verification)
        - monthly_debt >= 0    → +0.25 (debt data present)
        - employment_months >= 12 → +0.25 (stable employment)
        - employment_months >= 6  → +0.15 (partial employment)
        - income > 20_000      → +0.15 (income above minimum threshold)

    Returns: float in [0.0, 1.0] — values < 0.6 trigger human review guardrail.
    """
    score = 0.0

    # Factor 1: Income is valid and positive
    if income > 0:
        score += 0.35

    # Factor 2: Debt data is present (can be 0 legitimately)
    if monthly_debt >= 0:
        score += 0.25

    # Factor 3: Employment stability
    if employment_months >= 12:
        score += 0.25
    elif employment_months >= 6:
        score += 0.15

    # Factor 4: Income above minimum viable threshold ($20k/year)
    if income > 20_000:
        score += 0.15

    return round(min(1.0, max(0.0, score)), 4)
