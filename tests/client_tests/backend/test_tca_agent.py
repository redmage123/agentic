import pytest
import grpc
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from services.protos.hamiltonian_agent_pb2 import PredictionRequest, PredictionResponse
from utils.grpc_clients import get_hnn_prediction

def create_prediction_response(prediction: str, details: str) -> PredictionResponse:
    """Creates a real PredictionResponse instance instead of a mock"""
    response = PredictionResponse()
    response.prediction = prediction
    response.details = details
    return response

def setup_grpc_mock(mock_channel, response: PredictionResponse = None, error: Exception = None):
    """Sets up the gRPC mock chain to match the real implementation"""
    if error:
        mock_channel.return_value.__enter__.side_effect = error
        return

    # Create a mock Predict method that returns our response
    predict_method = MagicMock(return_value=response)
    
    # Create a mock stub that has our Predict method
    mock_stub = MagicMock()
    mock_stub.Predict = predict_method
    
    # Create a mock channel that returns our stub
    mock_instance = MagicMock()
    mock_instance.HamiltonianPredictionServiceStub = MagicMock(return_value=mock_stub)
    mock_channel.return_value.__enter__.return_value = mock_instance

class TestHNNPrediction:
    """Tests for the HNN prediction client"""

    def test_successful_prediction(self):
        """Test successful prediction with valid input"""
        # Create a real response object
        response = PredictionResponse()
        response.prediction = "cyclic_behavior_pattern"
        response.details = "This is a mocked response from the HNN model"
    
        # Create the predict callable that will return our response
        predict_method = MagicMock(return_value=response)
    
        # Create unary_unary that returns our predict method
        unary_unary = MagicMock(return_value=predict_method)
    
        # Create channel that provides unary_unary
        channel_instance = MagicMock()
        channel_instance.unary_unary = unary_unary
    
        with patch('grpc.insecure_channel') as mock_channel:
            # Setup channel context manager
            mock_channel.return_value.__enter__.return_value = channel_instance
        
            # Make the call
            result = get_hnn_prediction({"input_data": "test_stock_data"})
        
            # Verify everything
            assert predict_method.called  # Verify the call was made
            assert result == {
                "prediction": "cyclic_behavior_pattern",
                "details": "This is a mocked response from the HNN model"
            }
    def test_grpc_error_handling(self):
        """Test handling of gRPC communication errors"""
        with patch('grpc.insecure_channel') as mock_channel:
            setup_grpc_mock(mock_channel, error=grpc.RpcError("Connection failed"))
            
            result = get_hnn_prediction({"input_data": "test_stock_data"})
            
            assert "error" in result
            assert "Connection failed" in str(result["error"])

    def test_missing_input_data(self):
        """Test handling of missing input data"""
        with patch('grpc.insecure_channel') as mock_channel:
            setup_grpc_mock(mock_channel, response=create_prediction_response(
                prediction="error",
                details="Missing input data"
            ))
            
            result = get_hnn_prediction({})  # Missing input_data key
            
            assert result["prediction"] == "error"
            assert "Missing input data" in result["details"]

    @pytest.mark.parametrize("prediction_data", [
        ("upward_trend", "Strong buying pressure"),
        ("downward_trend", "Market correction expected"),
        ("sideways", "Consolidation phase")
    ])
    def test_different_prediction_types(self, prediction_data):
        """Test handling of different types of predictions"""
        prediction, details = prediction_data
        response = create_prediction_response(prediction=prediction, details=details)
        
        with patch('grpc.insecure_channel') as mock_channel:
            setup_grpc_mock(mock_channel, response=response)
            
            result = get_hnn_prediction({"input_data": "test_stock_data"})
            
            assert result["prediction"] == prediction
            assert result["details"] == details

    def test_empty_input_handling(self):
        """Test handling of empty input string"""
        response = create_prediction_response(
            prediction="error",
            details="Empty input data"
        )
        
        with patch('grpc.insecure_channel') as mock_channel:
            setup_grpc_mock(mock_channel, response=response)
            
            result = get_hnn_prediction({"input_data": ""})
            
            assert result["prediction"] == "error"
            assert "Empty input data" in result["details"]
