import pytest
import grpc
from concurrent import futures
from unittest.mock import Mock, patch

from utils.grpc_clients import get_hnn_prediction
from services.tca_service.service import HNNPredictionService
from services.protos.hamiltonian_agent_pb2 import PredictionRequest, PredictionResponse
from services.protos.hamiltonian_agent_pb2_grpc import (
    HamiltonianPredictionServiceStub,
    add_HamiltonianPredictionServiceServicer_to_server
)

@pytest.fixture
def service():
    return HNNPredictionService()

@pytest.fixture
def test_server():
    """Create a test gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    servicer = HNNPredictionService()
    add_HamiltonianPredictionServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port('[::]:0')  # Random port
    server.start()
    yield f'localhost:{port}'
    server.stop(0)

def test_predict_service_direct(service):
    """Test the prediction service directly"""
    request = PredictionRequest(input_data="test_data")
    context = Mock()  # Mock gRPC context
    
    response = service.Predict(request, context)
    assert isinstance(response, PredictionResponse)
    assert response.prediction != ""
    assert response.details != ""

def test_predict_service_grpc(test_server):
    """Test the prediction service over gRPC"""
    with grpc.insecure_channel(test_server) as channel:
        stub = HamiltonianPredictionServiceStub(channel)
        request = PredictionRequest(input_data="test_data")
        response = stub.Predict(request)
        assert isinstance(response, PredictionResponse)
        assert response.prediction != ""
        assert response.details != ""

def test_server_error_handling(service):
    """Test error handling in the service"""
    request = PredictionRequest(input_data="")  # Invalid input
    context = Mock()
    context.set_code = Mock()
    context.set_details = Mock()
    
    response = service.Predict(request, context)
    assert context.set_code.called
    assert context.set_details.called
