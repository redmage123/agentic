#!/usr/bin/bash

# Exit on error
set -e

echo "Setting up the Client Service..."

# Check for Python and Pip installation
if ! command -v python3 &> /dev/null
then
    echo "Python3 is not installed. Please install Python3 and rerun the script."
    exit
fi

if ! command -v pip3 &> /dev/null
then
    echo "Pip3 is not installed. Please install Pip3 and rerun the script."
    exit
fi

# Check for Node and npm installation
if ! command -v node &> /dev/null
then
    echo "Node.js is not installed. Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

# Install Python dependencies for the Flask backend
echo "Installing Python dependencies for the Flask backend..."
pip3 install -r ./requirements.txt

# Install JavaScript dependencies for the React frontend
echo "Installing JavaScript dependencies for the React frontend..."
cd services/client_service/frontend
npm install
cd ../../..

# Check for Docker installation
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
fi

# Build Docker images for both backend and frontend
echo "Building Docker images..."
docker build -t client-backend ./client_service/backend
docker build -t client-frontend ./client_service/frontend

echo "Setup complete. You can now run the application."

