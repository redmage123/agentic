# tests/tca_tests/integration/test_tca_integration.py
from concurrent import futures
from typing import Any, Dict

import grpc
import pytest

from utils.grpc_clients import get_hnn_prediction
from services.protos import (  # Import everything we need from protos
    hamiltonian_agent_pb2,
    hamiltonian_agent_pb2_grpc,
    PredictionRequest,
    PredictionResponse
)


# Mock HNN Agent server for integration testing
class MockHNNPredictionService(hamiltonian_agent_pb2_grpc.HamiltonianPredictionServiceServicer):
    def Predict(self, request: PredictionRequest, context) -> PredictionResponse:
        """
        Mocked Predict method to simulate HNN Agent's response.
        """
        # Here we simply echo back the input data with a mock prediction
        return PredictionResponse(
            prediction="mocked_cyclic_behavior_pattern",
            details="This is a mocked integration test response from HNN Agent",
        )


@pytest.fixture(scope="module")
def grpc_server():
    """
    Fixture to set up and tear down a mock gRPC server for integration testing.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    hamiltonian_agent_pb2_grpc.add_HamiltonianPredictionServiceServicer_to_server(
        MockHNNPredictionService(), server
    )
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
        "details": "This is a mocked integration test response from HNN Agent",
    }

