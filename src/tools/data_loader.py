"""
data_loader.py – Phase 14 utility

Loads test_applications.json and resolves each applicant's document list to
the corresponding PDF file paths inside data/synthetic_docs/.

Usage
-----
from src.tools.data_loader import load_test_applications, build_state_from_app

apps = load_test_applications()          # list[dict] from JSON
state = build_state_from_app(apps[0])   # LoanApplicationState with file paths
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Resolve paths relative to the project root (two levels up from src/tools/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SYNTHETIC_DOCS_DIR = os.path.join(_PROJECT_ROOT, "data", "synthetic_docs")
_TEST_APPS_PATH = os.path.join(_PROJECT_ROOT, "data", "test_applications.json")

# Mapping from document_type to the filename suffix used in synthetic_docs/
_DOC_TYPE_TO_SUFFIX: Dict[str, str] = {
    "id_proof": "id",
    "bank_statement": "bank_statement",
    "pay_slip": "pay_slip",
    "employment_letter": "employment_letter",
}


def load_test_applications(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads and returns all test applications from test_applications.json.

    Parameters
    ----------
    path : str, optional
        Override the default path to test_applications.json.

    Returns
    -------
    list[dict]
        Parsed list of application records.
    """
    target = path or _TEST_APPS_PATH
    if not os.path.exists(target):
        raise FileNotFoundError(
            f"test_applications.json not found at '{target}'. "
            "Run the synthetic data generator first."
        )
    with open(target, "r", encoding="utf-8") as f:
        apps = json.load(f)
    logger.info("Loaded %d test applications from %s", len(apps), target)
    return apps


def resolve_pdf_path(application_id: str, document_type: str) -> Optional[str]:
    """
    Returns the absolute path to the synthetic PDF for a given (application_id,
    document_type) pair, or None if the file does not exist.

    Example
    -------
    resolve_pdf_path("APP-001", "bank_statement")
    # → "/…/data/synthetic_docs/APP-001-bank_statement.pdf"
    """
    suffix = _DOC_TYPE_TO_SUFFIX.get(document_type, document_type)
    filename = f"{application_id}-{suffix}.pdf"
    full_path = os.path.join(_SYNTHETIC_DOCS_DIR, filename)
    if os.path.exists(full_path):
        return full_path
    logger.debug(
        "Synthetic PDF not found for %s / %s (expected: %s)",
        application_id,
        document_type,
        full_path,
    )
    return None


def build_state_from_app(app: Dict[str, Any], use_pdf_paths: bool = True):
    """
    Constructs a LoanApplicationState from a test_applications.json record.

    When *use_pdf_paths* is True (default), each document's extracted_text is
    replaced with the absolute path to its synthetic PDF so that the Phase 4
    parsing pipeline (parse_document / extract_fields) is exercised.  If the
    PDF is absent the embedded raw text is kept as a fallback.

    Parameters
    ----------
    app : dict
        A single application record from test_applications.json.
    use_pdf_paths : bool
        If True, resolve PDF paths and use them as extracted_text.

    Returns
    -------
    LoanApplicationState
    """
    # Import here to avoid circular imports at module load time
    from src.graph.state import ExtractedDocument, LoanApplicationState

    application_id = app["application_id"]
    docs: List[ExtractedDocument] = []

    for doc in app.get("documents", []):
        doc_type = doc["document_type"]
        raw_text = doc["extracted_text"]

        # Attempt to resolve the real PDF path
        text_or_path = raw_text
        if use_pdf_paths:
            pdf_path = resolve_pdf_path(application_id, doc_type)
            if pdf_path:
                text_or_path = pdf_path
                logger.debug(
                    "Resolved PDF path for %s / %s → %s",
                    application_id,
                    doc_type,
                    pdf_path,
                )
            else:
                logger.warning(
                    "No PDF found for %s / %s – using embedded text as fallback.",
                    application_id,
                    doc_type,
                )

        docs.append(
            ExtractedDocument(
                document_type=doc_type,
                extracted_text=text_or_path,
                validation_status=doc["validation_status"],
                confidence=doc["confidence"],
                extracted_fields=doc["extracted_fields"],
            )
        )

    return LoanApplicationState(
        application_id=application_id,
        applicant_data=app["applicant_data"],
        documents=docs,
    )


def iter_states(use_pdf_paths: bool = True):
    """
    Generator that yields a LoanApplicationState for every entry in
    test_applications.json.

    Parameters
    ----------
    use_pdf_paths : bool
        Forwarded to build_state_from_app.

    Yields
    ------
    LoanApplicationState
    """
    for app in load_test_applications():
        yield build_state_from_app(app, use_pdf_paths=use_pdf_paths)
