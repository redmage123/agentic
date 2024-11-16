# microservices/registry_service/main.py
import asyncio
import hydra
from omegaconf import DictConfig
import grpc
from concurrent import futures

from .presentation.grpc_servicer import RegistryServicer
from .application.service import RegistryService
from microservices.protos import registry_service_pb2_grpc as pb2_grpc

@hydra.main(config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set service name for config resolution
    cfg["service_name"] = "registry"
    
    # Initialize service
    registry_service = RegistryService(cfg)
    servicer = RegistryServicer(registry_service)
    
    # Create gRPC server
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_RegistryServiceServicer_to_server(servicer, server)
    
    # Add secure credentials if needed
    server.add_insecure_port(f'[::]:{cfg.service.port}')
    
    async def serve():
        await registry_service.start()
        await server.start()
        await server.wait_for_termination()
        await registry_service.stop()
    
    asyncio.run(serve())

if __name__ == "__main__":
    main()
