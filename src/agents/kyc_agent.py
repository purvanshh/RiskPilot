import logging
from typing import Any, Dict, List

from src.graph.state import ExtractedDocument, LoanApplicationState, validate_state
from src.tools.document_tools import extract_fields, parse_document, validate_fields

logger = logging.getLogger(__name__)


@validate_state
def kyc_node(state: LoanApplicationState) -> Dict[str, Any]:
    """
    KYC / Document Agent Node
    Extracts and validates identity, income, and employment details from uploaded documents.
    """
    logger.info(f"Starting KYC processing for application {state.application_id}")

    extracted_docs: List[ExtractedDocument] = []
    error_log: List[str] = list(state.error_log)

    # In a real app, this would iterate over raw file attachments.
    # Here, we process documents already simulated in state or read them from a tools function.
    if not state.documents:
        # If no documents are attached in state, we simulate calling the tools
        logger.warning("No documents found in state. Running document parsing simulation.")
        # Simulating document parse and validation (boilerplate)
        try:
            # Assume we received files and we parse them
            mock_docs = [
                {
                    "type": "id_proof",
                    "text": "ID Proof: Alice Johnson, DOB: 12/10/1990",
                },
                {"type": "bank_statement", "text": "Bank statement deposits"},
                {"type": "pay_slip", "text": "Pay slip gross income"},
            ]
            for doc in mock_docs:
                parsed_text = parse_document(doc["text"])
                fields = extract_fields(parsed_text)
                is_valid = validate_fields(fields)

                extracted_docs.append(
                    ExtractedDocument(
                        document_type=doc["type"],
                        extracted_text=parsed_text,
                        validation_status="valid" if is_valid else "needs_review",
                        confidence=0.9,
                        extracted_fields=fields,
                    )
                )
        except Exception as e:
            error_log.append(f"KYC Agent error: {str(e)}")
    else:
        # Use documents already parsed/loaded in the state
        extracted_docs = state.documents

    # Guardrail Check: Reject if <3 required documents
    # Required documents: id_proof, bank_statement, pay_slip, employment_letter (optional)
    doc_types = {d.document_type for d in extracted_docs}
    required_types = {"id_proof", "bank_statement", "pay_slip"}
    missing_critical_docs = list(required_types - doc_types)

    # Check income mismatch between pay slip and bank statement (simple heuristic)
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

    if pay_slip_income and bank_statement_income:
        # If mismatch is > 10%
        if (
            abs(pay_slip_income - bank_statement_income)
            / max(pay_slip_income, bank_statement_income)
            > 0.10
        ):
            logger.warning("Income mismatch detected between pay slip and bank statement.")
            fraud_flag = True
            kyc_confidence = 0.5

    # Check for name mismatch across documents to flag fraud
    names = []
    for doc in extracted_docs:
        name = doc.extracted_fields.get("name")
        if name:
            names.append(name.strip().lower())

    if len(set(names)) > 1:
        logger.warning("Name mismatch detected across documents.")
        fraud_flag = True
        kyc_confidence = 0.4

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
