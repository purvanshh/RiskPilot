import pytest

from src.graph.state import ExtractedDocument, LoanApplicationState, validate_state


def test_serialization():
    """Verify that serialization to and from dictionary works."""
    original = LoanApplicationState(
        application_id="APP-TEST",
        applicant_data={"income": 100000},
        documents=[
            ExtractedDocument(
                document_type="id_proof",
                extracted_text="ID sample",
                validation_status="valid",
                confidence=0.9,
                extracted_fields={"name": "Alice"},
            )
        ],
    )

    data = original.to_dict()
    assert isinstance(data, dict)
    assert data["application_id"] == "APP-TEST"
    assert data["state_version"] == "1.0.0"
    assert len(data["documents"]) == 1
    assert data["documents"][0]["document_type"] == "id_proof"

    deserialized = LoanApplicationState.from_dict(data)
    assert deserialized.application_id == "APP-TEST"
    assert len(deserialized.documents) == 1
    assert deserialized.documents[0].extracted_fields["name"] == "Alice"


def test_validate_state_decorator_success():
    """Verify that validate_state decorator passes valid nodes."""

    @validate_state
    def dummy_node(state: LoanApplicationState):
        return {"kyc_output": {"status": "success"}}

    state_dict = {"application_id": "APP-TEST", "applicant_data": {"income": 100000}}

    # Passing dict input
    res_dict = dummy_node(state_dict)
    assert res_dict == {"kyc_output": {"status": "success"}}

    # Passing LoanApplicationState input
    state_obj = LoanApplicationState.from_dict(state_dict)
    res_obj = dummy_node(state_obj)
    assert res_obj == {"kyc_output": {"status": "success"}}


def test_validate_state_decorator_invalid_input():
    """Verify that validate_state decorator catches invalid inputs."""

    @validate_state
    def dummy_node(state: LoanApplicationState):
        return {}

    # Invalid input (missing application_id)
    state_dict = {"applicant_data": {"income": 100000}}

    with pytest.raises(ValueError):
        dummy_node(state_dict)


def test_validate_state_decorator_invalid_output():
    """Verify that validate_state decorator catches invalid output updates."""

    @validate_state
    def dummy_node_invalid_out(state: LoanApplicationState):
        # Setting final_status to an invalid literal value
        return {"final_status": "invalid_status_value"}

    state_dict = {"application_id": "APP-TEST", "applicant_data": {"income": 100000}}

    with pytest.raises(ValueError):
        dummy_node_invalid_out(state_dict)


def test_validate_state_decorator_wrong_return_type():
    """Verify that validate_state decorator catches nodes returning non-dict types."""

    @validate_state
    def dummy_node_wrong_type(state: LoanApplicationState):
        return "not a dict"

    state_dict = {"application_id": "APP-TEST", "applicant_data": {"income": 100000}}

    with pytest.raises(TypeError):
        dummy_node_wrong_type(state_dict)
