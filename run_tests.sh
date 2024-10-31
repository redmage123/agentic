#!/bin/bash

# Exit on error
set -e

# Activate the existing virtual environment for backend tests
echo "Activating existing virtual environment in .venv..."
source .venv/Scripts/activate

# Install backend dependencies
echo "Installing backend dependencies..."
pip install -r requirements.txt

# Run TCA unit and integration tests
echo "Running TCA unit and integration tests..."
pytest tests/tca_tests --disable-warnings

# Run client backend tests (Flask)
echo "Running Client Service backend (Flask) tests..."
pytest tests/client_tests/backend --disable-warnings

# Deactivate the virtual environment
deactivate

# Navigate to the React frontend tests directory
echo "Running Client Service frontend (React) tests..."
cd tests/client_tests/frontend
npm install
npm test -- --watchAll=false

echo "All tests completed successfully."

