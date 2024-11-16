# setup.py
import os
import subprocess
from setuptools import find_packages, setup

# Define proto files directory
PROTO_DIR = os.path.join("microservices", "protos")
# Define output directory for generated files
OUT_DIR = os.path.join("microservices", "protos")

def generate_grpc_code():
    """Generate gRPC Python code from proto files"""
    # Create output directory if it doesn't exist
    os.makedirs(OUT_DIR, exist_ok=True)
    
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
    description="Multi-Agent AI Architecture Demonstration System",
    packages=find_packages(include=['microservices', 'models', 'tests', 'conf']),
    install_requires=[
        # gRPC dependencies
        "grpcio>=1.54.0",
        "grpcio-tools>=1.54.0",
        "protobuf>=4.22.3",
        
        # Web framework
        "flask>=2.0.0",
        
        # Testing
        "pytest>=7.3.1",
        "pytest-asyncio>=0.21.1",  # For async test support
        "pytest-timeout>=2.1.0",   # For test timeouts
        
        # Data processing
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        
        # Configuration management
        "hydra-core>=1.3.2",
        "hydra-colorlog>=1.2.0",    # For colored logging output
        "hydra-optuna-sweeper>=1.2.0",  # For parameter sweeping
        "omegaconf>=2.3.0",
        
        # Logging and monitoring
        "structlog>=23.1.0",      # For structured logging
        "prometheus-client>=0.17.1",  # For metrics
        "psutil>=5.9.0",          # For system metrics
        
        # Async support
        "aiohttp>=3.8.5",
        "async-timeout>=3.0,<4.0",
        
        # Utilities
        "pydantic>=2.0.0",       # For data validation
        "python-dotenv>=1.0.0",  # For environment variables
    ],
    extras_require={
        "dev": [
            # Testing and coverage
            "pytest>=7.3.1",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "pytest-asyncio>=0.21.1",
            "pytest-timeout>=2.1.0",
            "pytest-env>=0.8.1",      # For environment variables in tests
            "pytest-xdist>=3.3.1",    # For parallel testing
            
            # Code quality
            "black>=22.3.0",
            "flake8>=4.0.1",
            "mypy>=1.0.0",
            "isort>=5.12.0",
            
            # Documentation
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            
            # Development tools
            "ipython>=8.12.0",
            "jupyterlab>=4.0.2",
            
            # Debugging
            "debugpy>=1.6.7",
        ],
        "monitoring": [
            "prometheus-client>=0.17.1",
            "grafana-client>=2.2.0",
            "psutil>=5.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Main services
            "run-logging=microservices.logging_service.main:main",
            "run-registry=microservices.registry_service.main:main",
            "run-tca=microservices.tca_service.main:main",
            
            # AI Agents
            "run-fourier=microservices.fourier_agent.main:main",
            "run-hamiltonian=microservices.hamiltonian_agent.main:main",
            "run-generative=microservices.generative_agent.main:main",
            "run-perturbation=microservices.perturbation_agent.main:main",
            
            # Development tools
            "run-dev-server=microservices.client_service.app:main",
        ],
    },
    package_data={
        "conf": ["**/*.yaml"],  # Include all YAML files in conf directory
        "microservices": ["protos/*.proto"],  # Include proto files
    },
        classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    # Add pytest configuration
    options={
        'pytest': {
            'addopts': '--verbose --cov=microservices --cov-report=term-missing --asyncio-mode=auto',
            'testpaths': ['tests'],
            'python_files': 'test_*.py',
            'python_functions': 'test_*',
            'env': {
                'ENVIRONMENT': 'testing',
                'LOG_LEVEL': 'DEBUG',
                'HYDRA_FULL_ERROR': '1'
            }
        }
    }
)

