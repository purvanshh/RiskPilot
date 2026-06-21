# General Underwriting Guidelines

## 1. Manual Override Authority
- **Officer Tier Limits**: Loan officers may override automated denials only within their delegated authority. Tier 1 officers may approve overrides on loans up to **$150,000**. Tier 2 officers may approve overrides up to **$500,000**. Loans exceeding $500,000 require Tier 3 (senior underwriter) sign-off.
- **Override Documentation**: Every manual override must include a written justification referencing the specific compensating factor (e.g., high reserves, strong employment stability, low LTV). Overrides without documented justification are non-compliant.
- **Audit Trail**: All overrides are logged with officer ID, decision timestamp, and the original automated recommendation. Overrides are subject to quarterly compliance review.

## 2. Exception Handling
- **Eligible Exceptions**: Compensating factors that may justify an exception include reserves of 6+ months of housing payment, an LTV at or below 70%, or a credit score 30+ points above the minimum threshold. At least one such factor must be present.
- **Ineligible Exceptions**: Fraud indicators, hard-cap policy breaches (DTI > 50%, LTV > 85%, employment tenure < 6 months), and unverifiable income are **non-negotiable**. No exception is permitted in these cases regardless of compensating factors.
- **Single-Vector Exceptions**: At most one policy threshold may be exempted on a given application. Applications with two or more borderline violations must be denied or escalated to manual review — they may not be exception-approved.

## 3. Borderline Applications
- **Definition**: A borderline application has at least one metric within 10% of a policy threshold (e.g., DTI between 40% and 45%, LTV between 75% and 80%, credit score between 650 and 680).
- **Routing**: Borderline applications are not auto-approved. They must be routed to the Arbitrator agent and then to a human reviewer with the full set of retrieved policy chunks attached.
- **Confidence Threshold**: If the Credit Risk agent's confidence is below **0.60**, the application is treated as borderline regardless of metric values.

## 4. Conflict Resolution Between Agents
- **Unanimous Agreement**: When KYC, Credit Risk, and Policy agents agree on approve or deny, the recommendation flows through without arbitration.
- **Partial Disagreement**: If one agent dissents, the Arbitrator agent issues a `review_required` recommendation and forwards the application to a human reviewer with each agent's reasoning.
- **Hard Conflict**: A hard conflict (e.g., KYC `verified` + Policy `failed` + Credit `low risk`) always escalates to a senior underwriter. The Arbitrator must not auto-approve a hard-conflict case.

## 5. Documentation Requirements
- **Minimum Document Set**: Every application must include four documents — government-issued **ID**, **bank statement** (most recent 60 days), **pay slip** (most recent pay period), and **employment letter** (dated within 30 days).
- **Missing Documents**: Applications missing any required document are placed into the `retry` state by the KYC agent and are not eligible for credit or policy evaluation until complete.
- **Document Freshness**: Documents older than 90 days are treated as expired and must be re-submitted. Officers may not waive document freshness without Tier 2 authority.

## 6. Officer Responsibilities
- **Review Discipline**: Reviewing officers are responsible for reading every retrieved policy chunk and the full agent reasoning before issuing a decision. Decisions made without engaging the retrieved evidence are flagged in audit review.
- **Override Reason Field**: When overriding an automated recommendation, the officer must populate the `override_reason` field of the Human Decision record. Blank or boilerplate justifications ("approved per policy", "override applied") are not acceptable.
- **Conflict of Interest**: Officers may not review applications from applicants with whom they share a personal, familial, or financial relationship. Suspected conflicts must be escalated and reassigned.
- **Confidentiality**: All applicant data, retrieved policy chunks, and reasoning artifacts are confidential and may not be shared outside the underwriting team.
