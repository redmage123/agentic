import grpc
from unittest import mock
from typing import Dict, Any
from hnn_agent_pb2 import PredictionRequest, PredictionResponse
import hnn_agent_pb2_grpc
from agent_client import get_hnn_prediction

def test_get_hnn_prediction_success():
    """
    Test successful prediction response from the HNN Agent via gRPC.
    """
    # Mock the gRPC channel and stub
    with mock.patch("grpc.insecure_channel") as mock_channel:
        # Create a mock stub
        mock_stub = mock.MagicMock()
        mock_channel.return_value = mock_stub

        # Prepare the mock response
        mock_response = PredictionResponse(
            prediction="cyclic_behavior_pattern",
            details="This is a mocked response from the HNN model"
        )
        
        # Mock the Predict method on the stub
        mock_stub.Predict.return_value = mock_response

        # Call the function with test data
        data = {"input_data": "test_stock_data"}
        result = get_hnn_prediction(data)

        # Verify the response
        assert result == {
            "prediction": "cyclic_behavior_pattern",
            "details": "This is a mocked response from the HNN model"
        }
        mock_stub.Predict.assert_called_once_with(PredictionRequest(input_data="test_stock_data"))

def test_get_hnn_prediction_failure():
    """
    Test error handling when the HNN Agent is unreachable via gRPC.
    """
    with mock.patch("grpc.insecure_channel") as mock_channel:
        mock_stub = mock.MagicMock()
        mock_channel.return_value = mock_stub

        # Simulate a gRPC error
        mock_stub.Predict.side_effect = grpc.RpcError("Mocked gRPC error")

        # Call the function with test data
        data = {"input_data": "test_stock_data"}
        result = get_hnn_prediction(data)

        # Verify that the error is captured in the response
        assert "error" in result
        assert "Mocked gRPC error" in result["error"]

