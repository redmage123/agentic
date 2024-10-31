import pytest
from client_service.backend.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_status_endpoint(client):
    """Test the status endpoint to ensure the service is running."""
    response = client.get('/api/status')
    assert response.status_code == 200
    assert response.json == {"status": "Backend service is running"}

def test_predict_endpoint(client, monkeypatch):
    """Test the predict endpoint with mocked TCA response."""

    # Mocking get_prediction to avoid actual TCA call
    def mock_get_prediction(data):
        return {"prediction": "mocked_prediction"}
    
    monkeypatch.setattr('client_service.backend.tca_client.get_prediction', mock_get_prediction)

    # Send test data to the predict endpoint
    response = client.post('/api/predict', json={"inputData": "test_stock_data"})
    assert response.status_code == 200
    assert response.json == {"prediction": "mocked_prediction"}

