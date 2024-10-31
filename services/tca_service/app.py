from concurrent import futures
import grpc
from typing import Dict, Any

import hnn_agent_pb2
import hnn_agent_pb2_grpc
from hnn_model import process_hnn_prediction
from logging_service import logger

class HNNPredictionService(hnn_agent_pb2_grpc.HNNPredictionServiceServicer):
    def Predict(self, request: hnn_agent_pb2.PredictionRequest, context) -> hnn_agent_pb2.PredictionResponse:
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
        response = hnn_agent_pb2.PredictionResponse(
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
    hnn_agent_pb2_grpc.add_HNNPredictionServiceServicer_to_server(HNNPredictionService(), server)
    server.add_insecure_port("[::]:5002")
    logger.info("Starting HNN Agent gRPC service on port 5002")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()

