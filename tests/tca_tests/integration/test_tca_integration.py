import grpc
import pytest
from concurrent import futures
from unittest import mock
from typing import Dict, Any

from hnn_agent_pb2 import PredictionRequest, PredictionResponse
import hnn_agent_pb2_grpc
from agent_client import get_hnn_prediction

# Mock HNN Agent server for integration testing
class MockHNNPredictionService(hnn_agent_pb2_grpc.HNNPredictionServiceServicer):
    def Predict(self, request: PredictionRequest, context) -> PredictionResponse:
        """
        Mocked Predict method to simulate HNN Agent's response.
        """
        # Here we simply echo back the input data with a mock prediction
        return PredictionResponse(
            prediction="mocked_cyclic_behavior_pattern",
            details="This is a mocked integration test response from HNN Agent"
        )

@pytest.fixture(scope="module")
def grpc_server():
    """
    Fixture to set up and tear down a mock gRPC server for integration testing.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    hnn_agent_pb2_grpc.add_HNNPredictionServiceServicer_to_server(MockHNNPredictionService(), server)
    server.add_insecure_port("[::]:5002")
    server.start()
    yield server
    server.stop(None)

def test_tca_hnn_integration(grpc_server):
    """
    Integration test to verify TCA can communicate with the HNN Agent via gRPC.
    """
    # Test data to send to the TCA's gRPC client function
    test_data: Dict[str, Any] = {"input_data": "test_stock_data"}
    
    # Call the TCA's client function for the HNN Agent
    result = get_hnn_prediction(test_data)
    
    # Validate the mock response
    assert result == {
        "prediction": "mocked_cyclic_behavior_pattern",
        "details": "This is a mocked integration test response from HNN Agent"
    }

