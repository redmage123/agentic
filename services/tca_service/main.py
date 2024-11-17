# services/tca_service/main.py

import os
import asyncio
from pathlib import Path
import hydra
from omegaconf import DictConfig
import grpc
from concurrent import futures

from services.tca_service.presentation.grpc_servicer import TCAServicer
from services.tca_service.application.service import TCAService
from services.tca_service.infrastructure.routing import WeightedRoundRobinStrategy
from services.tca_service.infrastructure.agent_manager import AgentManager
from services.tca_service.infrastructure.response_aggregator import WeightedAverageAggregator
from services.tca_service.infrastructure.metrics_collector import PrometheusMetricsCollector
from services.tca_service.infrastructure.registry_client import RegistryClient
from services.tca_service.infrastructure.health import HealthMonitor
from services.protos import tca_service_pb2_grpc

# Get project root directory for config path
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = str(PROJECT_ROOT / "conf")

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the Traffic Control Agent service.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set service name for config resolution
    cfg["service_name"] = "tca"
    
    try:
        # Initialize components
        metrics_collector = PrometheusMetricsCollector(
            port=cfg.service.monitoring.metrics_port
        )
        
        registry_client = RegistryClient(
            host=cfg.service.registry.host,
            port=cfg.service.registry.port
        )
        
        routing_strategy = WeightedRoundRobinStrategy()
        agent_manager = AgentManager(registry_client)
        response_aggregator = WeightedAverageAggregator()
        
        # Initialize main service
        service = TCAService(
            config=cfg.service,
            routing_strategy=routing_strategy,
            agent_manager=agent_manager,
            response_aggregator=response_aggregator,
            metrics_collector=metrics_collector,
            registry_client=registry_client
        )
        
        # Initialize health monitor
        health_monitor = HealthMonitor(
            service=service,
            port=cfg.service.monitoring.health_check_port
        )
        
        # Initialize gRPC server
        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=cfg.service.grpc.max_workers),
            options=[
                ('grpc.max_send_message_length', cfg.service.grpc.max_message_size),
                ('grpc.max_receive_message_length', cfg.service.grpc.max_message_size),
            ]
        )
        
        # Add servicer to server
        servicer = TCAServicer(service)
        tca_service_pb2_grpc.add_TrafficControlServiceServicer_to_server(
            servicer, server
        )
        
        # Add insecure port
        server_address = f"{cfg.service.host}:{cfg.service.port}"
        server.add_insecure_port(server_address)
        
        async def serve():
            """Start all service components"""
            try:
                # Start service components
                await service.start()
                await health_monitor.start()
                
                # Start gRPC server
                await server.start()
                
                print(f"TCA service starting on {server_address}")
                
                # Wait for termination
                await server.wait_for_termination()
                
            except Exception as e:
                print(f"Error starting TCA service: {str(e)}")
                raise
            finally:
                # Cleanup
                await service.stop()
                await health_monitor.stop()
                await server.stop(grace=5)
        
        # Run the service
        asyncio.run(serve())
        
    except Exception as e:
        print(f"Failed to start TCA service: {str(e)}")
        raise
    finally:
        # Any additional cleanup if needed
        pass

if __name__ == "__main__":
    main()
