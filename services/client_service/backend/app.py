#!C:/Users/bbrel/agentic/.venv/bin/python
from flask import Flask, jsonify, request
from tca_client import get_prediction

app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint to request a stock prediction.
    Expects JSON data with prediction parameters.
    """
    data = request.json
    response = get_prediction(data)
    return jsonify(response)

@app.route('/api/status', methods=['GET'])
def status():
    """
    Health check endpoint for the backend.
    """
    return jsonify({"status": "Backend service is running"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

