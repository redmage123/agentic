# services/logging_service/application/service.py
import hydra
from omegaconf import DictConfig
from pathlib import Path
from ..domain.interfaces import ServiceContext

class LoggingService:
    """Main logging service implementation using Hydra configuration"""
    
    def __init__(self, context: ServiceContext):
        self.context = context
        self.setup_hydra_paths()
    
    def setup_hydra_paths(self):
        """Setup paths considering Hydra's working directory"""
        # Get the original working directory from Hydra
        orig_cwd = hydra.utils.get_original_cwd()
        
        # Update log directory path to be absolute
        if not self.context.config.logging.root_dir.is_absolute():
            self.context.config.logging.root_dir = Path(orig_cwd) / self.context.config.logging.root_dir
            
        # Create log directory if it doesn't exist
        self.context.config.logging.root_dir.mkdir(parents=True, exist_ok=True)
    
    async def run(self):
        """Run the service with Hydra configuration"""
        try:
            # Start components
            await self.context.registry.register_service()
            await self.context.registry.start_heartbeat()
            
            # Start gRPC server
            server = await self.setup_grpc_server()
            await server.start()
            
            # Wait for shutdown
            await server.wait_for_termination()
            
        finally:
            # Cleanup
            await self.context.registry.stop_heartbeat()
            await self.context.registry.deregister_service()
