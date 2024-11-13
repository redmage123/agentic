import pytest
import grpc
import unittest
from unittest import mock
from unittest.mock import patch, MagicMock  # Add this import


from services.protos.hamiltonian_agent_pb2 import PredictionRequest, PredictionResponse
from utils.grpc_clients import get_hnn_prediction
class test_tca_client_agent(unittest.TestCase):

    def test_get_hnn_prediction_success(self):
        """
        Test successful prediction response from the HNN Agent via gRPC.
        """

        with mock.patch("services.protos.hamiltonian_agent_pb2_grpc.HamiltonianPredictionServiceStub") as mock_stub_class:

            # Create a mock stub with our expected response
            mock_stub_instance = mock_stub_class.return_value
            mock_stub_instance.Predict.return_value = mock.MagicMock(
                prediction="cyclic_behavior_pattern",
                details="This is a mocked response from the HNN model"
            )
        
            # Create a response that acts like a proper gRPC response
            response = mock.MagicMock()
            response.prediction = "cyclic_behavior_pattern"
            response.details = "This is a mocked response from the HNN model"



            # Test the function
            result = get_hnn_prediction({"input_data": "test_stock_data"})
        
            # Configure the mock response properties
            self.assertTrue ("prediction" in result)
            self.assertTrue ("details" in result)
            self.assertEqual(result["prediction"], "cyclic_behavior_pattern")
            self.assertEqual(result["details"], "This is a mocked response from the HNN model")

            # Verify the stub was called correctly
            mock_stub_instance.Predict.assert_called_once_with(PredictionRequest(input_data="test_stock_data"))

    def test_get_hnn_prediction_failure(self):
        """
        Test error handling when the HNN Agent is unreachable via gRPC.
        """
        with mock.patch("services.protos.hamiltonian_agent_pb2_grpc.HamiltonianPredictionServiceStub") as mock_stub_class:

            # Simulate a gRPC error
            mock_stub_instance = mock_stub_class.return_value
            mock_stub_instance.Predict.side_effect = grpc.RpcError("Mocked gRPC error")

            # Call the function with test data
            data = {"input_data": "test_stock_data"}
            result = get_hnn_prediction(data)

            # Verify that the error is captured in the response
            self.assertIn("error", result)
            self.assertIn("Mocked gRPC error", result["error"])

