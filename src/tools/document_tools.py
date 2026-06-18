import re
from typing import Any, Dict


def parse_document(document_content: str) -> str:
    """
    Simulates extracting text from raw documents (PDF, images, etc.).
    For this boilerplate, it returns the input string directly or simulated OCR output.
    """
    if not document_content:
        return ""
    # Strip basic whitespace / simulate cleanup
    return document_content.strip()


def extract_fields(text: str) -> Dict[str, Any]:
    """
    Simulates using an LLM to extract structured fields from document text.
    Uses regex or basic pattern matching as a fallback for synthetic data.
    """
    fields = {}

    # Simple regex parsing for synthetic format: "Field: Value"
    matches = re.findall(r"(\w+(?:\s+\w+)*):\s*([^.\n]+)", text)
    for key, value in matches:
        clean_key = key.lower().replace(" ", "_")
        clean_val = value.strip()

        # Try to parse numerical values
        if clean_val.replace(",", "").replace("$", "").isdigit():
            fields[clean_key] = int(clean_val.replace(",", "").replace("$", ""))
        else:
            fields[clean_key] = clean_val

    # Add defaults if not found
    if "income" in fields and "income_monthly" not in fields:
        fields["income_monthly"] = (
            int(fields["income"]) / 12 if isinstance(fields["income"], (int, float)) else 0
        )

    return fields


def validate_fields(fields: Dict[str, Any]) -> bool:
    """
    Checks field completeness and format validity.
    """
    # Needs at least a name or income-related data to be valid
    has_name = "name" in fields or "applicant_name" in fields
    has_income = "income" in fields or "income_monthly" in fields or "gross_pay" in fields

    return bool(has_name or has_income)
