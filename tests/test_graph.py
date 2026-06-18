from src.graph.graph import graph
from src.graph.state import ExtractedDocument, HumanDecision, LoanApplicationState


def test_full_graph_happy_path():
    """Tests the compiled LangGraph execution for a clean approval scenario."""
    # Assemble complete, valid application state
    initial_state = LoanApplicationState(
        application_id="APP-001",
        applicant_data={
            "name": "Alice Johnson",
            "income": 80000,
            "monthly_debt": 1200,
            "loan_amount": 200000,
            "property_value": 280000,
            "employment_months": 36,
        },
        documents=[
            ExtractedDocument(
                document_type="id_proof",
                extracted_text="Name: Alice Johnson",
                validation_status="valid",
                confidence=1.0,
                extracted_fields={"name": "Alice Johnson"},
            ),
            ExtractedDocument(
                document_type="pay_slip",
                extracted_text="Gross pay: $6,666/mo. Name: Alice Johnson",
                validation_status="valid",
                confidence=1.0,
                extracted_fields={"name": "Alice Johnson", "income_monthly": 6666},
            ),
            ExtractedDocument(
                document_type="bank_statement",
                extracted_text="Direct deposit: $6,666. Name: Alice Johnson",
                validation_status="valid",
                confidence=1.0,
                extracted_fields={"name": "Alice Johnson", "income_monthly": 6666},
            ),
        ],
    )

    # Run the graph
    final_state_dict = graph.invoke(initial_state)

    # Cast to object or retrieve keys
    assert final_state_dict["kyc_output"] is not None
    assert final_state_dict["credit_output"] is not None
    assert final_state_dict["policy_output"] is not None
    assert final_state_dict["arbitrator_output"] is not None

    # Since there is no human decision provided, it should halt at under_review
    assert final_state_dict["final_status"] == "under_review"


def test_full_graph_with_human_decision():
    """Tests that the graph processes human decisions and updates final status."""
    # Assemble complete, valid application state
    initial_state = LoanApplicationState(
        application_id="APP-001",
        applicant_data={
            "name": "Alice Johnson",
            "income": 80000,
            "monthly_debt": 1200,
            "loan_amount": 200000,
            "property_value": 280000,
            "employment_months": 36,
        },
        documents=[
            ExtractedDocument(
                document_type="id_proof",
                extracted_text="Name: Alice Johnson",
                validation_status="valid",
                confidence=1.0,
                extracted_fields={"name": "Alice Johnson"},
            ),
            ExtractedDocument(
                document_type="pay_slip",
                extracted_text="Gross pay: $6,666/mo. Name: Alice Johnson",
                validation_status="valid",
                confidence=1.0,
                extracted_fields={"name": "Alice Johnson", "income_monthly": 6666},
            ),
            ExtractedDocument(
                document_type="bank_statement",
                extracted_text="Direct deposit: $6,666. Name: Alice Johnson",
                validation_status="valid",
                confidence=1.0,
                extracted_fields={"name": "Alice Johnson", "income_monthly": 6666},
            ),
        ],
        human_decision=HumanDecision(
            officer_id="OFF-007", decision="approve", timestamp="2026-06-17T22:15:00"
        ),
    )

    # Run the graph
    final_state_dict = graph.invoke(initial_state)

    # Final status should transition to approved
    assert final_state_dict["final_status"] == "approved"
