# General Underwriting Guidelines

## 1. Override Authority

### 1.1 Loan Officer Override
A certified loan officer may override any system recommendation (`approve`, `deny`, or `review_required`) provided that:
- The override decision is documented with a written reason of at least 50 characters.
- The overriding officer holds a minimum authorisation level of **Level 2 Underwriter**.
- The override is recorded in the audit trail with the officer's ID, timestamp, and reason.

### 1.2 Prohibited Overrides
Loan officers **must not** override a `deny` recommendation to `approve` in the following circumstances:
- Credit score below **500** (unacceptable systemic risk).
- Confirmed fraud flag (KYC agent detected identity or income inconsistency).
- DTI ratio exceeding **65%** with no compensating factors.

### 1.3 Escalation Path
If a loan officer believes an override is justified in a prohibited case, the application must be escalated to a **Senior Underwriter** or **Risk Committee** for secondary review before any approval can be issued.

---

## 2. Exception Handling

### 2.1 Compensating Factors
An application that fails one or more policy thresholds may still be considered for approval if compensating factors are present and documented:
- **Strong liquid reserves**: Applicant holds at least **12 months** of mortgage payments in verified liquid assets.
- **Low LTV**: Loan-to-Value ratio is below **60%**, indicating significant equity cushion.
- **Long employment tenure**: Applicant has been continuously employed for **5+ years** at the same employer.
- **Co-borrower**: A creditworthy co-borrower with a credit score ≥ 720 is added to the application.

### 2.2 Temporary Income Disruption
Where an applicant has experienced a temporary reduction in income (e.g., parental leave, short-term medical leave), underwriters may use the **pre-disruption income** for affordability calculations, provided documentary evidence is supplied and the disruption period does not exceed **6 months**.

### 2.3 Self-Employed Applicants
Self-employed income must be verified using **2 years of tax returns** and averaged. Any single-year income spike must be discounted by 20% in affordability calculations.

---

## 3. Audit Trail Requirements

### 3.1 Mandatory Fields
Every loan application processed through the system must produce an audit record containing:
- `application_id`: Unique identifier for the loan application.
- `timestamp`: ISO-8601 UTC timestamp of each decision event.
- `agent`: The name of the agent node that generated the record (`kyc`, `credit`, `policy`, `arbitrator`, `human_review`).
- `event_type`: One of `validation_flag`, `recommendation`, `override`, `error`.
- `details`: Human-readable description of the event (minimum 20 characters).
- `officer_id`: Populated for any human decision; null for automated events.

### 3.2 Retention Policy
Audit logs must be retained for a minimum of **7 years** in accordance with applicable mortgage lending regulations. Logs must be stored in an immutable, append-only format (`logs/audit.jsonl`).

### 3.3 Tamper Evidence
Any modification to an existing audit record must be detected and flagged. The system uses append-only JSONL files as a lightweight tamper-evident mechanism. Production deployments must additionally use a cryptographic hash chain or a compliant audit database.

---

## 4. Regulatory Compliance

### 4.1 Fair Lending
All underwriting decisions must comply with applicable fair lending laws, including:
- **Equal Credit Opportunity Act (ECOA)**: Credit decisions must not be based on race, colour, religion, national origin, sex, marital status, age, or source of income.
- **Fair Housing Act (FHA)**: Discrimination in residential lending is prohibited.
- **Adverse Action Notices**: Any denial must be accompanied by a written adverse action notice listing the principal reason(s) for denial within 30 days.

### 4.2 Data Privacy
Applicant personal data (name, DOB, income, employment details) must be handled in accordance with applicable data protection legislation. Access to raw applicant records must be restricted to authorised personnel only.

### 4.3 Model Risk Management
The AI system is a **decision-support tool**, not an autonomous decision-maker. A qualified human loan officer must review and approve or override every recommendation before it is communicated to the applicant. No automated loan denial or approval may be issued without human sign-off.

### 4.4 Explainability Requirement
Every system recommendation must be accompanied by a human-readable summary and confidence score. The reasoning field of the `ArbitratorOutput` must clearly state the primary factors that drove the recommendation, enabling loan officers to explain the decision to applicants upon request.

---

## 5. System Confidence Thresholds

| Confidence Score | System Action |
|------------------|--------------|
| ≥ 0.80 | Recommendation surfaced; officer may approve without additional review. |
| 0.60 – 0.79 | Recommendation surfaced; officer must review agent reasoning before deciding. |
| < 0.60 | Application automatically routed to `review_required`; senior sign-off required. |
| 0.00 (system error) | Application flagged as failed; routed to human review with error log. |
