import logging
import os
from typing import Any, Dict, List

from src.graph.state import (
    ExtractedDocument,
    LoanApplicationState,
    graceful_fallback,
    timeout_resilience,
    validate_state,
)
from src.tools.document_tools import extract_fields, parse_document

logger = logging.getLogger(__name__)


@validate_state
@graceful_fallback("kyc")
@timeout_resilience(30.0)
def kyc_node(state: LoanApplicationState) -> Dict[str, Any]:
    """
    KYC / Document Agent Node
    Extracts and validates identity, income, and employment details from uploaded documents.
    """
    logger.info(f"Starting KYC processing for application {state.application_id}")

    error_log: List[str] = list(state.error_log)

    # Task 5.1: No fallback when documents are missing; instead, raise a validation error
    if not state.documents:
        raise ValueError(
            f"Validation Error: No documents provided for application {state.application_id}. "
            "Ingestion cannot proceed."
        )

    # Process and parse documents if they are file paths, otherwise use them directly
    extracted_docs: List[ExtractedDocument] = []
    for doc in state.documents:
        text = doc.extracted_text
        fields = doc.extracted_fields
        confidence = doc.confidence
        status = doc.validation_status

        # If text is an existing file path, parse it and extract fields
        if text and isinstance(text, str) and os.path.exists(text):
            try:
                parsed_text = parse_document(text)
                extracted_data = extract_fields(parsed_text, doc.document_type)
                fields = extracted_data.get("extracted_fields", {})
                confidence = extracted_data.get("confidence", 0.90)
                text = parsed_text
                status = "valid"
            except Exception as e:
                logger.error(f"Error parsing document file {text}: {str(e)}")
                error_log.append(f"KYC Agent parsing error for {doc.document_type}: {str(e)}")
                status = "invalid"
                confidence = 0.0
                fields = {}

        # Task 5.4: Calculate document-level confidence
        # Aggregate field confidences; if <0.7, set validation_status = "needs_review"
        if confidence < 0.7:
            status = "needs_review"

        extracted_docs.append(
            ExtractedDocument(
                document_type=doc.document_type,
                extracted_text=text,
                validation_status=status,
                confidence=confidence,
                extracted_fields=fields,
            )
        )

    # Task 5.3: Check presence of at least ID, pay slip, bank statement
    doc_types = {d.document_type for d in extracted_docs}
    required_types = {"id_proof", "bank_statement", "pay_slip"}
    missing_critical_docs = list(required_types - doc_types)

    # Check income mismatch between pay slip and bank statement (Task 5.2)
    pay_slip_income = None
    bank_statement_income = None
    for doc in extracted_docs:
        if doc.document_type == "pay_slip":
            pay_slip_income = (
                doc.extracted_fields.get("income_monthly")
                or doc.extracted_fields.get("income", 0) / 12
            )
        elif doc.document_type == "bank_statement":
            bank_statement_income = (
                doc.extracted_fields.get("income_monthly")
                or doc.extracted_fields.get("income", 0) / 12
            )

    fraud_flag = False
    kyc_confidence = 1.0

    # Compare income: flag if >20% difference (Task 5.2)
    if pay_slip_income and bank_statement_income:
        diff_ratio = abs(pay_slip_income - bank_statement_income) / max(
            pay_slip_income, bank_statement_income
        )
        if diff_ratio > 0.20:
            logger.warning(
                f"Income mismatch (>20%) detected between pay slip (${pay_slip_income:.2f}) "
                f"and bank statement (${bank_statement_income:.2f}). Difference: {diff_ratio:.2%}"
            )
            fraud_flag = True
            kyc_confidence = min(kyc_confidence, 0.5)

    # Check for name mismatch across documents to flag fraud (Task 5.2)
    id_name = None
    pay_slip_name = None
    for doc in extracted_docs:
        if doc.document_type == "id_proof":
            id_name = doc.extracted_fields.get("name")
        elif doc.document_type == "pay_slip":
            pay_slip_name = doc.extracted_fields.get("name")

    names = []
    for doc in extracted_docs:
        name = doc.extracted_fields.get("name")
        if name:
            names.append(name.strip().lower())

    if id_name and pay_slip_name:
        if id_name.strip().lower() != pay_slip_name.strip().lower():
            logger.warning(
                f"Name mismatch detected: ID name '{id_name}', " f"pay slip name '{pay_slip_name}'"
            )
            fraud_flag = True
            kyc_confidence = min(kyc_confidence, 0.4)
    elif len(set(names)) > 1:
        logger.warning("Name mismatch detected across documents.")
        fraud_flag = True
        kyc_confidence = min(kyc_confidence, 0.4)

    kyc_output = {
        "status": "completed",
        "missing_critical_docs": len(missing_critical_docs) > 0,
        "missing_docs_list": missing_critical_docs,
        "fraud_flag": fraud_flag,
        "confidence": kyc_confidence,
        "verified_fields": {
            "name": next(
                (
                    d.extracted_fields.get("name")
                    for d in extracted_docs
                    if d.extracted_fields.get("name")
                ),
                None,
            ),
            "income": next(
                (
                    d.extracted_fields.get("income_monthly") * 12
                    for d in extracted_docs
                    if d.extracted_fields.get("income_monthly")
                ),
                state.applicant_data.get("income"),
            ),
            "employer": next(
                (
                    d.extracted_fields.get("employer")
                    for d in extracted_docs
                    if d.extracted_fields.get("employer")
                ),
                None,
            ),
        },
    }

    return {
        "documents": extracted_docs,
        "kyc_output": kyc_output,
        "error_log": error_log,
    }
