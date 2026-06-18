import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def validate_application_input(
    application_data: Dict[str, Any], documents: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """
    Validates input schema and documents before processing:
    - Minimum 3 documents required.
    - File types must be standard (PDF, JPG, PNG, TXT).
    - Checks required applicant fields.
    """
    errors = []

    # 1. Check required fields in application data
    required_fields = [
        "name",
        "income",
        "monthly_debt",
        "loan_amount",
        "property_value",
        "employment_months",
    ]
    for field in required_fields:
        if field not in application_data:
            errors.append(f"Missing required applicant data field: '{field}'")

    # 2. Check document count (At least 3 required)
    if len(documents) < 3:
        errors.append(
            f"Application has only {len(documents)} documents. "
            "Minimum of 3 required documents is mandated by KYC policy."
        )

    # 3. Check document formats (simulate validation)
    allowed_types = ["pdf", "jpg", "png", "jpeg", "txt"]
    for i, doc in enumerate(documents):
        doc_type = doc.get("document_type")
        if not doc_type:
            errors.append(f"Document at index {i} is missing its 'document_type' label.")

        # If actual file path/name is simulated
        filename = doc.get("filename", "")
        if filename:
            ext = filename.split(".")[-1].lower() if "." in filename else ""
            if ext not in allowed_types:
                errors.append(
                    f"Invalid file type for {filename}. Only PDF, JPG, PNG, and TXT are accepted."
                )

    is_valid = len(errors) == 0
    return is_valid, errors
