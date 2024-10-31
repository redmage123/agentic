import pytest
from app import app
from unittest.mock import patch
from agent_client import get_predictions_from_agents

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_status_endpoint(client):
    """Test the status endpoint to ensure the TCA is running."""
    response = client.get('/status')
    assert response.status_code == 200
    assert response.json == {"status": "TCA service is running"}

@patch('agent_client.get_predictions_from_agents')
def test_predict_endpoint(mock_get_predictions, client):
    """Test the predict endpoint with mocked agent responses."""
    # Mocking the get_predictions_from_agents function
    mock_get_predictions.return_value = {
        "hnn_agent": {"prediction": "cyclic_pattern"},
        "fnn_agent": {"prediction": "frequency_pattern"},
        "perturbation_agent": {"prediction": "trend_anomaly"}
    }

    response = client.post('/predict', json={"inputData": "test_stock_data"})
    assert response.status_code == 200
    assert response.json == {
        "hnn_agent": {"prediction": "cyclic_pattern"},
        "fnn_agent": {"prediction": "frequency_pattern"},
        "perturbation_agent": {"prediction": "trend_anomaly"}
    }

