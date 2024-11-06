from concurrent import futures
from typing import Any, Dict

import grpc
from models.hnn_model import process_hnn_prediction
from ..logging_service import logger

from ..protos import hamiltonian_agent_pb2
from ..protos import hamiltonian_agent_pb2_grpc
from utils.grpc_clients import get_hnn_prediction


class HNNPredictionService(hamiltonian_agent_pb2_grpc.HamiltonianPredictionServiceServicer):
    def Predict(self, request: hamiltonian_agent_pb2.PredictionRequest, context) -> hamiltonian_agent_pb2.PredictionResponse:

        """
        Handles prediction requests from the TCA.
        
        Parameters:
        - request (PredictionRequest): gRPC request containing input data.
        - context: gRPC context object for managing RPC status.
        
        Returns:
        - PredictionResponse: gRPC response with prediction data.
        """
        input_data = request.input_data
        logger.info("Received prediction request in HNN Agent: %s", input_data)
        
        # Call the model processing function (stubbed in this skeleton)
        prediction: Dict[str, Any] = process_hnn_prediction({"input_data": input_data})
        
        # Construct response using the gRPC response type
        response = hamiltonian_agent_pb2.PredictionResponse(
            prediction=prediction["prediction"],
            details=prediction["details"]
        )
        
        logger.info("Generated prediction in HNN Agent: %s", prediction)
        return response

def serve() -> None:
    """
    Starts the gRPC server to listen for TCA requests.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    hamiltonian_agent_pb2_grpc.add_HNNPredictionServiceServicer_to_server(HNNPredictionService(), server)
    server.add_insecure_port("[::]:5002")
    logger.info("Starting HNN Agent gRPC service on port 5002")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()


