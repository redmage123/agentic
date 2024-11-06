from services.client_service.backend.tca_client import (
    get_prediction,
)  # Updated import path


def test_get_prediction_success(requests_mock):
    """Test successful communication with the TCA."""
    mock_url = "http://tca-service:5001/predict"
    requests_mock.post(mock_url, json={"prediction": "mocked_response"})

    response = get_prediction({"inputData": "test_data"})
    assert response == {"prediction": "mocked_response"}


def test_get_prediction_failure(requests_mock):
    """Test failed communication with the TCA."""
    mock_url = "http://tca-service:5001/predict"
    requests_mock.post(mock_url, status_code=500)

    response = get_prediction({"inputData": "test_data"})
    assert "error" in response
