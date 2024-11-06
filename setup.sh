#!/usr/bin/bash

# Exit on error
set -e

echo "Setting up the Client Service..."

SERVICE_PATH="./services"
CLIENT_SERVICE_PATH="$SERVICE_PATH/client_service/"
TCA_SERVICE_PATH="$SERVICE_PATH/tca_service/"

# Detect OS platform
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="ubuntu"
elif [[ "$OSTYPE" == "msys" ]]; then
    OS="windows"
else
    echo "Unsupported OS detected. This setup script only supports Ubuntu and Windows."
    exit 1
fi

# Function to check and create a directory if it does not exist
ensure_directory() {
    local dir_path=$1
    if [ ! -d "$dir_path" ]; then
        echo "Directory $dir_path not found. Attempting to create it..."
        mkdir -p "$dir_path"
        if [ ! -d "$dir_path" ]; then
            echo "Failed to create directory $dir_path. Please check permissions and try again."
            exit 1
        fi
    fi
}

# Function to install dependencies on Ubuntu
install_ubuntu_dependencies() {
    echo "Detected OS: Ubuntu"
    sudo apt update

    # Install Python3 and pip if not installed
    if ! command -v python3 &> /dev/null; then
        echo "Installing Python3..."
        sudo apt install -y python3 python3-pip
    fi

    # Install Node.js if not installed
    if ! command -v node &> /dev/null; then
        echo "Installing Node.js..."
        sudo apt install -y nodejs npm
    fi

    # Install Protocol Buffers compiler if not installed
    if ! command -v protoc &> /dev/null; then
        echo "Installing Protocol Buffers compiler (protoc)..."
        sudo apt install -y protobuf-compiler
    fi
}

# Function to install dependencies on Windows
install_windows_dependencies() {
    echo "Detected OS: Windows"

    # Install Chocolatey if not installed
    if ! command -v choco &> /dev/null; then
        echo "Installing Chocolatey..."
        powershell -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command \
        "Set-ExecutionPolicy Bypass -Scope Process; \
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; \
        iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"
    fi

    # Install Python3 and pip if not installed
    if ! command -v python3 &> /dev/null; then
        echo "Installing Python3..."
        choco install -y python
    fi

    # Install Node.js if not installed
    if ! command -v node &> /dev/null; then
        echo "Installing Node.js..."
        choco install -y nodejs
    fi

    # Install Protocol Buffers compiler if not installed
    if ! command -v protoc &> /dev/null; then
        echo "Installing Protocol Buffers compiler (protoc)..."
        PROTOC_ZIP="protoc-21.12-win64.zip"
        curl -LO "https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP"
        ensure_directory "$HOME/protoc"
        unzip -o $PROTOC_ZIP -d "$HOME/protoc" bin/protoc.exe
        rm $PROTOC_ZIP
        export PATH="$HOME/protoc/bin:$PATH"
    fi
}

# Install dependencies based on detected OS
if [[ "$OS" == "ubuntu" ]]; then
    install_ubuntu_dependencies
elif [[ "$OS" == "windows" ]]; then
    install_windows_dependencies
fi

# Install Python dependencies for the Flask backend
echo "Installing Python dependencies for the Flask backend..."
pip3 install -r ./requirements.txt

# Ensure the directory for JavaScript dependencies exists and install them
ensure_directory "services/client_service/frontend"
echo "Installing JavaScript dependencies for the React frontend..."
cd services/client_service/frontend
npm install
cd ../../..

# Install Python gRPC and protobuf tools
echo "Installing gRPC and protobuf tools for Python..."
pip3 install grpcio grpcio-tools

# Define paths for protobuf generation for each agent
PROTO_DIR="services/protos"
declare -A AGENTS=(
    ["hamiltonian"]="hamiltonian_agent.proto"
    ["fourier"]="fourier_agent.proto"
    ["perturbation"]="perturbation_agent.proto"
    ["generative"]="generative_agent.proto"
)
declare -A OUTPUT_DIRS=(
    ["hamiltonian"]="services/hamiltonian_agent"
    ["fourier"]="services/fourier_agent"
    ["perturbation"]="services/perturbation_agent"
    ["generative"]="services/generative_agent"
)

# Ensure the proto and output directories exist
ensure_directory "$PROTO_DIR"
for agent in "${!OUTPUT_DIRS[@]}"; do
    ensure_directory "${OUTPUT_DIRS[$agent]}"
done

# Create default .proto files if they are missing or empty, and generate protobuf files
for agent in "${!AGENTS[@]}"; do
    PROTO_PATH="$PROTO_DIR/${AGENTS[$agent]}"
    OUTPUT_PATH="${OUTPUT_DIRS[$agent]}"

    # Check if the .proto file is empty or missing and create it if necessary
    if [ ! -s "$PROTO_PATH" ]; then
        echo "Creating default content for $PROTO_PATH..."
        cat <<EOF > "$PROTO_PATH"
syntax = "proto3";

package ${agent}_service;

// Service for ${agent^} Neural Network predictions
service ${agent^}PredictionService {
  rpc Predict (PredictionRequest) returns (PredictionResponse);
}

// Message defining the input structure for predictions
message PredictionRequest {
  string input_data = 1; // Input data for ${agent^} Neural Network
}

// Message defining the structure of the prediction response
message PredictionResponse {
  string result = 1;       // Prediction result
  string details = 2;      // Additional details
}
EOF
        echo "Default .proto content for $agent created."
    fi

    # Generate protobuf files for gRPC services
    echo "Generating protobuf files for ${agent^} agent..."
    python -m grpc_tools.protoc -I=$PROTO_DIR --python_out=$OUTPUT_PATH --grpc_python_out=$OUTPUT_PATH $PROTO_PATH
done

echo "Setup complete. You can now run the application."

