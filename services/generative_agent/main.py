import asyncio
from pathlib import Path
import hydra
from omegaconf import DictConfig
import grpc
from concurrent import futures

from services.utils.environment import load_environment
from services.generative_agent.presentation.grpc_servicer import GenerativeServicer
from services.generative_agent.application.service import GenerativeService
from services.generative_agent.infrastructure.llm_client import LLMClient
from services.generative_agent.infrastructure.prompt_manager import PromptManager
from services.generative_agent.infrastructure.cache_manager import CacheManager
from services.generative_agent.infrastructure.registry_client import RegistryClient
from services.generative_agent.infrastructure.health import HealthMonitor
from services.protos import generative_service_pb2_grpc

# Get project root directory for config path
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = str(PROJECT_ROOT / "conf")

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the Generative Agent service.
    
    Args:
        cfg: Hydra configuration object
    """
    try:
        # Load environment variables (including ANTHROPIC_API_KEY)
        load_environment()
        
        # Set service name for config resolution
        cfg["service_name"] = "generative"
        
        # Initialize components
        prompt_manager = PromptManager(
            base_path=PROJECT_ROOT / "services" / "generative_agent" / "prompts"
        )
        
        cache_manager = CacheManager(
            enabled=cfg.service.cache.enabled,
            ttl=cfg.service.cache.ttl,
            max_size=cfg.service.cache.max_size
        )
        
        llm_client = LLMClient(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model=cfg.service.llm.model,
            temperature=cfg.service.llm.temperature,
            max_tokens=cfg.service.llm.max_tokens
        )
        
        registry_client = RegistryClient(
            host=cfg.service.registry.host,
            port=cfg.service.registry.port
        )
        
        # Initialize main service
        service = GenerativeService(
            config=cfg.service,
            prompt_manager=prompt_manager,
            llm_client=llm_client,
            cache_manager=cache_manager,
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
        servicer = GenerativeServicer(service)
        generative_service_pb2_grpc.add_GenerativeServiceServicer_to_server(
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
                
                # Register with service registry
                await registry_client.register_service()
                
                # Start gRPC server
                await server.start()
                
                print(f"Generative Agent service starting on {server_address}")
                
                # Wait for termination
                await server.wait_for_termination()
                
            except Exception as e:
                print(f"Error starting Generative Agent service: {str(e)}")
                raise
            finally:
                # Cleanup
                await service.stop()
                await health_monitor.stop()
                await registry_client.deregister_service()
                await server.stop(grace=5)
        
        # Run the service
        asyncio.run(serve())
        
    except Exception as e:
        print(f"Failed to start Generative Agent service: {str(e)}")
        raise
    finally:
        # Any additional cleanup if needed
        pass

if __name__ == "__main__":
    main()
