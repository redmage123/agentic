import pytest
import grpc
from unittest.mock import Mock, patch
from services.client_service.backend.app import app, get_prediction_from_tca
from services.protos.hamiltonian_agent_pb2 import PredictionResponse

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_grpc_stub():
    with patch('services.client_service.backend.app.HamiltonianPredictionServiceStub') as mock_stub:
        # Create a mock stub instance
        stub_instance = Mock()
        mock_stub.return_value = stub_instance
        
        # Create a mock response
        mock_response = PredictionResponse(
            prediction="test_prediction",
            details="test_details"
        )
        stub_instance.Predict.return_value = mock_response
        yield stub_instance

def test_status_endpoint_success(client):
    """Test status endpoint when TCA is available"""
    with patch('grpc.insecure_channel') as mock_channel:
        mock_channel.return_value.channel_ready.return_value = True
        response = client.get('/api/status')
        assert response.status_code == 200
        assert response.json == {
            "status": "Backend service is running",
            "tca_service": "connected"
        }


def test_predict_endpoint_success(client, mock_grpc_stub):
    """Test successful prediction request"""
    test_data = {"inputData": "test_stock_data"}
    response = client.post('/api/predict', json=test_data)
    
    assert response.status_code == 200
    assert response.json == {
        "prediction": "test_prediction",
        "details": "test_details"
    }
    # Verify gRPC stub was called correctly
    mock_grpc_stub.Predict.assert_called_once()

def test_status_endpoint_tca_down(client):
    """Test status endpoint when TCA is unavailable"""
    with patch('grpc.insecure_channel') as mock_channel:
        # Setup the mock to raise an error when checking channel readiness
        mock_channel_instance = mock_channel.return_value
        mock_channel_instance.channel_ready.side_effect = grpc.RpcError("Connection failed")
        
        # Make the request
        response = client.get('/api/status')
        
        # Verify response
        assert response.status_code == 200
        assert response.json["status"] == "Backend service is running"
        assert response.json["tca_service"] == "disconnected"

def test_predict_endpoint_missing_data(client):
    """Test prediction request with missing data"""
    response = client.post('/api/predict', json={})
    assert response.status_code == 400
    assert "error" in response.json

def test_predict_endpoint_grpc_error(client, mock_grpc_stub):
    """Test prediction request when TCA service fails"""
    mock_grpc_stub.Predict.side_effect = grpc.RpcError("Service unavailable")
    response = client.post('/api/predict', json={"inputData": "test_data"})
    assert "error" in response.json
    assert "gRPC error" in response.json["error"]
