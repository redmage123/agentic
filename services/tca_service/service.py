from concurrent import futures
from typing import Any, Dict

import grpc

# from hnn_model import process_hnn_prediction
from services.logging_service import logger
from services.protos.hamiltonian_agent_pb2 import PredictionRequest, PredictionResponse
from services.protos.hamiltonian_agent_pb2_grpc import HamiltonianPredictionServiceServicer, add_HamiltonianPredictionServiceServicer_to_server
# from utils.grpc_clients import get_hnn_prediction
from models.hnn_model import process_hnn_prediction  # Add this import


class HNNPredictionService(HamiltonianPredictionServiceServicer):
    def Predict(self, request: PredictionRequest, context) -> PredictionResponse:
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

        # Add input validation
        if not input_data or input_data.isspace():
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Input data cannot be empty")
            return PredictionResponse(
                prediction="error",
                details="Input data cannot be empty"
            )

        try:
            # Call the model processing function (stubbed in this skeleton)
            prediction: Dict[str, Any] = process_hnn_prediction({"input_data": input_data})

            # Construct response using the gRPC response type
            response = PredictionResponse(
                prediction=prediction["prediction"], 
                details=prediction["details"]
            )

            logger.info("Generated prediction in HNN Agent: %s", prediction)
            return response
            
        except Exception as e:
            logger.error(f"Error processing prediction: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error processing prediction: {str(e)}")
            return PredictionResponse(
                prediction="error",
                details=f"Internal error: {str(e)}"
            )




def serve() -> None:
    """
    Starts the gRPC server to listen for TCA requests.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_HNNPredictionServiceServicer_to_server(
        HNNPredictionService(), server
    )
    server.add_insecure_port("[::]:5002")
    logger.info("Starting HNN Agent gRPC service on port 5002")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()

