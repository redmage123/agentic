#!C:/Users/bbrel/agentic/.venv/bin/python
from flask import Flask, jsonify, request
import grpc

from services.protos.hamiltonian_agent_pb2 import PredictionRequest
from services.protos.hamiltonian_agent_pb2_grpc import HamiltonianPredictionServiceStub

app = Flask(__name__)

# Configure gRPC client
TCA_SERVER_ADDRESS = 'localhost:5002'  # Update with your TCA service address

def get_prediction_from_tca(data):
    """
    Send prediction request to TCA service via gRPC.
    """
    try:
        # Create a gRPC channel
        with grpc.insecure_channel(TCA_SERVER_ADDRESS) as channel:
            # Create a stub (client)
            stub = HamiltonianPredictionServiceStub(channel)
            
            # Create request object
            request = PredictionRequest(input_data=str(data['inputData']))
            
            # Make the gRPC call
            response = stub.Predict(request)
            
            return {
                "prediction": response.prediction,
                "details": response.details
            }
    except grpc.RpcError as e:
        return {"error": f"gRPC error: {str(e)}"}
    except Exception as e:
        return {"error": f"Error communicating with TCA: {str(e)}"}

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Endpoint to request a stock prediction.
    Expects JSON data with prediction parameters.
    """
    data = request.json
    if not data or 'inputData' not in data:
        return jsonify({"error": "Missing input data"}), 400
    
    response = get_prediction_from_tca(data)
    return jsonify(response)

@app.route("/api/status", methods=["GET"])
def status():
    """
    Health check endpoint for the backend.
    """
    try:
        with grpc.insecure_channel('localhost:5002') as channel:
            try:
                channel.channel_ready()
                tca_status = "connected"
            except grpc.RpcError:
                tca_status = "disconnected"
    except Exception:
        tca_status = "disconnected"
        
    return jsonify({
        "status": "Backend service is running",
        "tca_service": tca_status
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
