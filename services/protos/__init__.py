# services/protos/__init__.py
from .hamiltonian_agent_pb2 import PredictionRequest, PredictionResponse
from . import hamiltonian_agent_pb2
from . import hamiltonian_agent_pb2_grpc 

__all__ = [
   'hamiltonian_agent_pb2',
   'hamiltonian_agent_pb2_grpc',
   'PredictionRequest',
   'PredictionResponse'
]
