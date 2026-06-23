import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class ApplicantDataInputSchema(BaseModel):
    model_config = {"extra": "allow"}

    name: str = Field(min_length=1, description="Applicant name")
    income: float = Field(gt=0, description="Annual income must be positive")
    monthly_debt: float = Field(ge=0, description="Monthly debt obligations must be non-negative")
    loan_amount: float = Field(gt=0, description="Requested loan amount must be positive")
    property_value: float = Field(gt=0, description="Property value must be positive")
    employment_months: int = Field(
        ge=0, description="Employment duration must be non-negative integer"
    )


def validate_application_input(
    application_data: Dict[str, Any],
    documents: List[Any],
    application_id: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """
    Validates input schema and documents before processing:
    - Minimum 3 documents required.
    - File types must be standard (PDF, JPG, PNG).
    - Checks required applicant fields against Pydantic models.
    - Checks maximum document size limit of 10MB.
    - Logs violations to the audit log if application_id is provided.
    """
    errors = []

    # 1. Pydantic validation on application data
    try:
        ApplicantDataInputSchema.model_validate(application_data)
    except ValidationError as e:
        for err in e.errors():
            loc = " -> ".join(str(x) for x in err["loc"])
            msg = err["msg"]
            errors.append(f"Field '{loc}' validation failed: {msg}")

    # 2. Check document count (At least 3 required)
    if len(documents) < 3:
        errors.append(
            f"Application has only {len(documents)} documents. "
            "Minimum of 3 required documents is mandated by KYC policy."
        )

    # 3. Check document formats & size limits
    allowed_types = {"pdf", "jpg", "png", "jpeg"}
    size_limit = 10 * 1024 * 1024  # 10MB

    for i, doc in enumerate(documents):
        # Support both dictionary and Pydantic object
        if hasattr(doc, "to_dict"):
            doc_dict = doc.to_dict()
        elif isinstance(doc, dict):
            doc_dict = doc
        else:
            doc_dict = {}

        doc_type = doc_dict.get("document_type")
        if not doc_type:
            errors.append(f"Document at index {i} is missing its 'document_type' label.")

        # Check format/extension
        filename = doc_dict.get("filename", "")
        extracted_text = doc_dict.get("extracted_text", "")

        path_to_check = None
        if filename:
            path_to_check = filename
        elif extracted_text and isinstance(extracted_text, str) and os.path.exists(extracted_text):
            path_to_check = extracted_text

        if path_to_check:
            ext = path_to_check.split(".")[-1].lower() if "." in path_to_check else ""
            if ext not in allowed_types:
                errors.append(
                    f"Invalid file type for {os.path.basename(path_to_check)}. "
                    "Only PDF, JPG, and PNG are accepted."
                )

        # Check size limit: Max 10MB
        file_size = None
        for size_key in ["file_size", "size", "size_bytes", "file_size_bytes"]:
            if doc_dict.get(size_key) is not None:
                try:
                    file_size = int(doc_dict[size_key])
                    break
                except (ValueError, TypeError):
                    pass

        if file_size is None and path_to_check and os.path.exists(path_to_check):
            try:
                file_size = os.path.getsize(path_to_check)
            except Exception as e:
                logger.warning(f"Could not get file size for {path_to_check}: {e}")

        if file_size is not None and file_size > size_limit:
            name_str = (
                os.path.basename(path_to_check) if path_to_check else f"Document at index {i}"
            )
            errors.append(
                f"File {name_str} exceeds maximum size limit of 10MB "
                f"(actual: {file_size / (1024 * 1024):.2f}MB)."
            )

    # 4. Audit logging
    if application_id and errors:
        from src.guardrails.audit_logger import log_guardrail_flag

        for error in errors:
            log_guardrail_flag(application_id, "input", error)

    is_valid = len(errors) == 0
    return is_valid, errors
