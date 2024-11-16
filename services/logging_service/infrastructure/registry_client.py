# services/logging_service/infrastructure/registry_client.py
import grpc
from ..domain.interfaces import RegistryClient
from services.protos import registry_service_pb2 as registry_pb2
from services.protos import registry_service_pb2_grpc as registry_pb2_grpc

class GrpcRegistryClient:
    """Concrete implementation of registry client using gRPC"""
    def __init__(self, registry_url: str):
        self.registry_url = registry_url
        self.channel = None
        self.stub = None

    async def _ensure_connected(self):
        """Ensure gRPC connection is established"""
        if not self.channel:
            self.channel = grpc.aio.insecure_channel(self.registry_url)
            self.stub = registry_pb2_grpc.RegistryServiceStub(self.channel)

    async def register_service(self, service_url: str) -> bool:
        """Register with the service registry"""
        try:
            await self._ensure_connected()
            request = registry_pb2.RegisterRequest(
                service_name="logging_service",
                service_url=service_url,
                service_type="logging",
                health_check_endpoint="/health"
            )
            
            response = await self.stub.Register(request)
            return response.success
            
        except Exception:
            return False

    async def deregister_service(self) -> bool:
        """Deregister from the service registry"""
        try:
            await self._ensure_connected()
            request = registry_pb2.DeregisterRequest(
                service_name="logging_service"
            )
            
            response = await self.stub.Deregister(request)
            return response.success
            
        except Exception:
            return False
