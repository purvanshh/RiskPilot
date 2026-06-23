import argparse
import os
import sys

# Adjust path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph.graph import graph  # noqa: E402
from src.tools.data_loader import build_state_from_app, load_test_applications  # noqa: E402


def run_demo(use_pdf_paths: bool = True):
    print("==================================================")
    print("   RiskPilot Multi-Agent Underwriter Demo Run    ")
    print("==================================================")
    if use_pdf_paths:
        print("Mode: PDF parsing (Phase 4 pipeline active)")
    else:
        print("Mode: Embedded text (fast / offline)")
    print()

    test_apps_path = os.path.join(os.path.dirname(__file__), "../data/test_applications.json")
    if not os.path.exists(test_apps_path):
        print(f"Error: {test_apps_path} not found.")
        sys.exit(1)

    applications = load_test_applications(test_apps_path)
    print(f"Loaded {len(applications)} test applications.\n")

    success_count = 0
    for app in applications:
        app_id = app["application_id"]
        name = app["applicant_data"]["name"]
        expected = app["expected_recommendation"]
        desc = app.get("description", "")

        print("-" * 50)
        print(f"Running Application: {app_id} - {name}")
        print(f"Scenario: {desc}")
        print(f"Expected Recommendation: {expected.upper()}")
        print("-" * 50)

        # Build state – resolves PDF paths when use_pdf_paths=True
        initial_state = build_state_from_app(app, use_pdf_paths=use_pdf_paths)

        try:
            final_state = graph.invoke(initial_state)

            # Extract agent outputs
            kyc_out = final_state.get("kyc_output")
            credit_out = final_state.get("credit_output")
            policy_out = final_state.get("policy_output")
            arb_out = final_state.get("arbitrator_output")
            final_status = final_state.get("final_status")

            final_status_str = final_status.upper() if final_status else "N/A"
            print(f"Execution complete. Final status: {final_status_str}")

            if kyc_out:
                has_missing = kyc_out.get("missing_critical_docs")
                print(f"  [KYC] Verification status: {'FAILED' if has_missing else 'PASSED'}")
                if kyc_out.get("fraud_flag"):
                    print("  [KYC] ⚠️ FRAUD FLAG DETECTED!")
            if credit_out:
                category_str = credit_out.risk_category.upper()
                print(
                    f"  [Credit] Score: {credit_out.credit_score} | "
                    f"DTI: {credit_out.dti_ratio:.2%} | Category: {category_str}"
                )
            if policy_out:
                ltv_str = f"{policy_out.ltv_ratio:.2%}" if policy_out.ltv_ratio else "N/A"
                print(f"  [Policy] Passed: {policy_out.policy_passed} | LTV: {ltv_str}")
                if policy_out.violations:
                    print(f"  [Policy] Violations: {policy_out.violations}")
            if arb_out:
                print(f"  [Arbitrator] Recommendation: {arb_out.recommendation.upper()}")
                print(f"  [Arbitrator] Confidence Score: {arb_out.confidence_score:.2f}")
                print(f"  [Arbitrator] Summary: {arb_out.summary}")

                # Check if it matches expected
                actual = arb_out.recommendation
                if actual == expected or (
                    expected == "review_required" and final_status == "under_review"
                ):
                    print("  Result: ✅ MATCHED expected recommendation.")
                    success_count += 1
                else:
                    print(
                        f"  Result: ❌ MISMATCHED expected recommendation "
                        f"(Expected: {expected}, Actual: {actual})."
                    )
            else:
                # If arbitrator didn't run (e.g. bypassed due to missing docs or fraud)
                if expected == "review_required" and final_status == "under_review":
                    print(
                        "  Result: ✅ MATCHED expected recommendation "
                        "(Routed directly to Human Review)."
                    )
                    success_count += 1
                else:
                    print(
                        f"  Result: ❌ MISMATCHED expected recommendation "
                        f"(Expected: {expected}, Arbitrator did not run)."
                    )

        except Exception as e:
            print(f"Error running pipeline: {str(e)}")

        print("\n")

    print("==================================================")
    print(f"Demo Summary: {success_count}/{len(applications)} cases verified successfully.")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RiskPilot demo runner – runs all 6 test applications through the pipeline."
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use embedded text from JSON instead of resolving PDF file paths (faster, offline).",
    )
    args = parser.parse_args()
    run_demo(use_pdf_paths=not args.raw)
