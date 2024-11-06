#!/bin/bash

# Exit on error
set -e

# Activate the existing virtual environment
echo "Activating existing virtual environment in .venv..."
source .venv/Scripts/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Generate protobuf files for gRPC services
echo "Generating protobuf files..."
PROTO_DIR="services/protos"                 # Directory for .proto files
OUTPUT_DIR="services/hamiltonian_agent"      # Output directory for the Hamiltonian agent

mkdir -p $OUTPUT_DIR                         # Ensure the output directory exists
python -m grpc_tools.protoc -I=$PROTO_DIR --python_out=$OUTPUT_DIR --grpc_python_out=$OUTPUT_DIR $PROTO_DIR/hamiltonian_agent.proto

# Add generated protobuf files to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/$OUTPUT_DIR

# Run TCA unit and integration tests
echo "Running TCA unit and integration tests..."
pytest tests/tca_tests --disable-warnings

# Run client backend tests (Flask)
echo "Running Client Service backend (Flask) tests..."
pytest tests/client_tests/backend --disable-warnings

# Deactivate the virtual environment
deactivate

# Run frontend tests for React in the client directory
echo "Running Client Service frontend (React) tests..."
cd tests/client_tests/frontend
npm install
npm test -- --watchAll=false

echo "All setup, protobuf generation, and tests completed successfully."


