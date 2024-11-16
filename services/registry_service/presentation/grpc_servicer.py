# services/registry_service/presentation/grpc_servicer.py
import grpc
from datetime import datetime
from typing import Dict, Any

from ..domain.interfaces import ServiceRegistry, ServiceInfo, HealthStatus
from ..application.service import RegistryService
from services.protos import registry_service_pb2 as pb2
from services.protos import registry_service_pb2_grpc as pb2_grpc

class RegistryServicer(pb2_grpc.RegistryServiceServicer):
    def __init__(self, registry_service: RegistryService):
        self.registry_service = registry_service

    async def Register(self, request: pb2.RegisterRequest, context) -> pb2.RegisterResponse:
        try:
            service_info = ServiceInfo(
                name=request.service_name,
                host=request.host,
                port=request.port,
                metadata=dict(request.metadata),
                last_heartbeat=datetime.now(),
                status="STARTING",
                version=request.version
            )
            
            service_id = await self.registry_service.registry.register(service_info)
            return pb2.RegisterResponse(
                success=True,
                service_id=service_id,
                message="Service registered successfully"
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.RegisterResponse(
                success=False,
                message=f"Registration failed: {str(e)}"
            )
