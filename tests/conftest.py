import os
import sys

import pytest

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture
def app():
    from services.client_service.backend.app import app

    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def grpc_channel():
    import grpc

    return grpc.insecure_channel("localhost:50051")


@pytest.fixture
def mock_tca_response():
    return {"prediction": "mocked_prediction", "details": "mocked details"}
