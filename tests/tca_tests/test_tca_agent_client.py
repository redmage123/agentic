from unittest import mock

import grpc
from services.tca_service.agent_client import get_hnn_prediction
from services.protos.hamiltonian_agent_pb2 import PredictionRequest, PredictionResponse
from utils.grpc_clients import get_hnn_prediction

def test_get_hnn_prediction_success():
    """
    Test successful prediction response from the HNN Agent via gRPC.
    """
    with mock.patch("grpc.insecure_channel") as mock_channel:
        # Create a mock stub with our expected response
        stub = mock.MagicMock()
        
        # Create a response that acts like a proper gRPC response
        response = mock.MagicMock()
        mock_prediction = "cyclic_behavior_pattern"
        mock_details = "This is a mocked response from the HNN model"
        
        # Configure the mock response properties
        response.configure_mock(**{
            'prediction': mock_prediction,
            'details': mock_details
        })
        
        # Configure the stub to return our response
        stub.configure_mock(**{
            'Predict.return_value': response
        })
        
        # Configure the channel to return our stub
        mock_channel.return_value.__enter__.return_value = stub
        
        # Test the function
        result = get_hnn_prediction({"input_data": "test_stock_data"})
        
        # Verify the results
        assert result == {
            "prediction": mock_prediction,
            "details": mock_details
        }
        
        # Verify the stub was called correctly
        stub.Predict.assert_called_once()

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
