# Credit Risk Policy

## 1. Credit Score Thresholds
- **Tier 1 (Low Risk)**: Credit Score >= 720. Eligible for all standard products.
- **Tier 2 (Medium Risk)**: Credit Score between 650 and 719. Requires standard review.
- **Tier 3 (High Risk)**: Credit Score between 600 and 649. Requires manual underwriter sign-off.
- **Tier 4 (Unacceptable Risk)**: Credit Score < 600. Direct denial unless compensating factors are present.

## 2. Minimum Credit Requirement
The absolute minimum credit score required for auto-approval is **650**. Any application with a credit score below 650 must be marked as `review_required` or `deny` depending on other risk vectors.
