from typing import Literal


def calculate_credit_score(
    income: float,
    monthly_debt: float,
    employment_months: int,
    extracted_credit_score: int = None,
) -> int:
    """
    Computes a deterministic credit score based on income, debt, and employment history.
    Formula:
    dti = monthly_debt / (income / 12)
    base_score = 300 + (income / 1000) * 10 + employment_months * 2 - dti * 200
    If extracted credit score from credit report is provided, blends them.
    """
    # Guard against division by zero
    monthly_income = max(1.0, income / 12.0)
    dti = monthly_debt / monthly_income

    # Calculate base score
    base_score = 300 + (income / 1000.0) * 10.0 + employment_months * 2.0 - dti * 200.0

    if extracted_credit_score is not None:
        base_score = 0.7 * base_score + 0.3 * extracted_credit_score

    # Boundary constraints: 300 to 850
    return min(850, max(300, int(base_score)))


def dti_calculator(monthly_debt: float, annual_income: float) -> float:
    """
    Computes Debt-to-Income (DTI) ratio.
    """
    monthly_income = max(1.0, annual_income / 12.0)
    return monthly_debt / monthly_income


def risk_classifier(credit_score: int) -> Literal["low", "medium", "high", "very_high"]:
    """
    Maps credit score to a risk category.
    """
    if credit_score >= 720:
        return "low"
    elif credit_score >= 650:
        return "medium"
    elif credit_score >= 600:
        return "high"
    else:
        return "very_high"
