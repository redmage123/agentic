# C:\Users\bbrel\agentic\setup.py
import os
import subprocess

from setuptools import find_packages, setup

# Define proto files directory
PROTO_DIR = os.path.join("services", "protos")
# Define output directory for generated files
OUT_DIR = os.path.join("services", "protos")


def generate_grpc_code():
    """Generate gRPC Python code from proto files"""
    # Get list of all .proto files
    proto_files = [f for f in os.listdir(PROTO_DIR) if f.endswith(".proto")]

    print("Generating gRPC code for:", proto_files)

    for proto_file in proto_files:
        proto_path = os.path.join(PROTO_DIR, proto_file)
        # Generate both *_pb2.py and *_pb2_grpc.py files
        command = [
            "python",
            "-m",
            "grpc_tools.protoc",
            f"--proto_path={PROTO_DIR}",
            f"--python_out={OUT_DIR}",
            f"--grpc_python_out={OUT_DIR}",
            proto_path,
        ]
        print(f"Generating code for {proto_file}...")
        subprocess.run(command, check=True)


# Generate gRPC code before setup
generate_grpc_code()

setup(
    name="agentic",
    version="0.1.0",
    author="Your Name",
    description="Stock prediction microservices with ML agents",
    packages=find_packages(include=['services','models','tests']),
    python_requires=">=3.9",
    install_requires=[
        "grpcio>=1.54.0",
        "grpcio-tools>=1.54.0",
        "protobuf>=4.22.3",
        "flask>=2.0.0",
        "pytest>=7.3.1",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        # Add any other dependencies your services need
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "pytest-cov>=4.0.0",
            "black>=22.3.0",
            "flake8>=4.0.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-client=services.client_service.app:main",
            "run-tca=services.tca_service.app:main",
            "run-fourier=services.fourier_agent.service:serve",
            "run-hamiltonian=services.hamiltonian_agent.service:serve",
            "run-generative=services.generative_agent.service:serve",
            "run-perturbation=services.perturbation_agent.service:serve",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
    ],
)
