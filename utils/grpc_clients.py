# utils/grpc_clients.py
from typing import Dict, Any
import grpc

from services.protos import hamiltonian_agent_pb2
from services.protos import hamiltonian_agent_pb2_grpc

def get_hnn_prediction(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Client function to make gRPC calls to the HNN service
    """
    try:
        with grpc.insecure_channel('localhost:5002') as channel:
            # Create gRPC stub
            stub = hamiltonian_agent_pb2_grpc.HamiltonianPredictionServiceStub(channel)
            
            # Create request
            request = hamiltonian_agent_pb2.PredictionRequest(
                input_data=str(data.get('input_data', ''))
            )
            
            # Make gRPC call
            response = stub.Predict(request)
            
            # Return formatted response
            return {
                "prediction": response.prediction,
                "details": response.details
            }
    except Exception as e:
        return {"error": str(e)}
