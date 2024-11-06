#!/bin/bash

# Function to check if a process started successfully
check_process() {
    if [ $? -eq 0 ]; then
        echo "$1 started successfully."
    else
        echo "Error starting $1."
        exit 1
    fi
}

echo "Starting up the Client Service backend..."
cd services/client_service/backend || exit
python app.py &
check_process "Client Service backend"
sleep 2  # Allow time for the service to initialize

echo "Starting the TCA Agent..."
cd ../../tca_agent || exit
python app.py &
check_process "TCA Agent"
sleep 2  # Allow time for the service to initialize

echo "All services have started."

