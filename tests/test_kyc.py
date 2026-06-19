import pytest

from src.agents.kyc_agent import kyc_node
from src.graph.state import ExtractedDocument, LoanApplicationState


def test_kyc_success():
    """Tests KYC agent with a complete and matching set of documents."""
    state = LoanApplicationState(
        application_id="APP-001",
        applicant_data={"name": "Alice Johnson", "income": 80000},
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

    result = kyc_node(state)
    assert result["kyc_output"]["missing_critical_docs"] is False
    assert result["kyc_output"]["fraud_flag"] is False
    assert result["kyc_output"]["confidence"] == 1.0


def test_kyc_missing_documents():
    """Tests KYC agent flags missing documents when required files are not provided."""
    state = LoanApplicationState(
        application_id="APP-004",
        applicant_data={"name": "Diana Prince", "income": 90000},
        documents=[
            ExtractedDocument(
                document_type="id_proof",
                extracted_text="Name: Diana Prince",
                validation_status="valid",
                confidence=1.0,
                extracted_fields={"name": "Diana Prince"},
            )
        ],
    )

    result = kyc_node(state)
    assert result["kyc_output"]["missing_critical_docs"] is True
    assert "bank_statement" in result["kyc_output"]["missing_docs_list"]
    assert "pay_slip" in result["kyc_output"]["missing_docs_list"]


def test_kyc_fraud_name_mismatch():
    """Tests KYC agent flags a fraud scenario when document names do not match."""
    state = LoanApplicationState(
        application_id="APP-006",
        applicant_data={"name": "Alice Johnson", "income": 80000},
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
                extracted_text="Gross pay: $6,666/mo. Name: Bob Smith",
                validation_status="valid",
                confidence=1.0,
                extracted_fields={"name": "Bob Smith", "income_monthly": 6666},
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

    result = kyc_node(state)
    assert result["kyc_output"]["fraud_flag"] is True
    assert result["kyc_output"]["confidence"] < 0.7


def test_kyc_no_documents_raises_error():
    """Verify that an empty document list raises a ValueError."""
    state = LoanApplicationState(
        application_id="APP-ERR", applicant_data={"name": "Alice Johnson"}, documents=[]
    )
    with pytest.raises(ValueError, match="No documents provided"):
        kyc_node(state)


def test_kyc_low_document_confidence_triggers_review():
    """Verify that a document with confidence < 0.7 triggers needs_review status."""
    state = LoanApplicationState(
        application_id="APP-LOW-CONF",
        applicant_data={"name": "Alice Johnson"},
        documents=[
            ExtractedDocument(
                document_type="id_proof",
                extracted_text="Name: Alice Johnson",
                validation_status="valid",
                confidence=0.5,  # Low confidence
                extracted_fields={"name": "Alice Johnson"},
            ),
            ExtractedDocument(
                document_type="pay_slip",
                extracted_text="Pay slip",
                validation_status="valid",
                confidence=0.9,
                extracted_fields={"name": "Alice Johnson", "income_monthly": 6666},
            ),
            ExtractedDocument(
                document_type="bank_statement",
                extracted_text="Bank statement",
                validation_status="valid",
                confidence=0.9,
                extracted_fields={"name": "Alice Johnson", "income_monthly": 6666},
            ),
        ],
    )
    result = kyc_node(state)
    docs = result["documents"]
    # The first document validation_status should be updated to needs_review
    assert docs[0].validation_status == "needs_review"


def test_kyc_income_mismatch_fraud():
    """Verify that an income discrepancy > 20% triggers a fraud flag."""
    state = LoanApplicationState(
        application_id="APP-INC-MISMATCH",
        applicant_data={"name": "Alice Johnson"},
        documents=[
            ExtractedDocument(
                document_type="id_proof",
                extracted_text="Name: Alice Johnson",
                validation_status="valid",
                confidence=0.9,
                extracted_fields={"name": "Alice Johnson"},
            ),
            ExtractedDocument(
                document_type="pay_slip",
                extracted_text="Gross pay: $8,000",
                validation_status="valid",
                confidence=0.9,
                extracted_fields={"name": "Alice Johnson", "income_monthly": 8000},
            ),
            ExtractedDocument(
                document_type="bank_statement",
                extracted_text="Deposit: $5,000",  # >20% difference
                validation_status="valid",
                confidence=0.9,
                extracted_fields={"name": "Alice Johnson", "income_monthly": 5000},
            ),
        ],
    )
    result = kyc_node(state)
    assert result["kyc_output"]["fraud_flag"] is True
    assert result["kyc_output"]["confidence"] <= 0.5
