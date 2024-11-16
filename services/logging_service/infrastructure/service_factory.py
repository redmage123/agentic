# services/logging_service/infrastructure/service_factory.py
from ..domain.config import ServiceConfig
from ..domain.interfaces import ServiceContext, ServiceFactory
from .file_handler import FileLogWriter
from .log_repository import FileLogRepository
from .log_processor import StandardLogProcessor
from .registry_client import GrpcRegistryClient

class StandardServiceFactory(ServiceFactory):
    """Factory for creating service components with Hydra configuration"""
    
    @staticmethod
    def create_context(config: ServiceConfig) -> ServiceContext:
        """Create fully configured service context"""
        
        # Initialize components with configuration
        writer = FileLogWriter(config.logging)
        repository = FileLogRepository(config)
        processor = StandardLogProcessor(config.processing)
        registry = GrpcRegistryClient(config.registry)
        
        return ServiceContext(
            config=config,
            writer=writer,
            repository=repository,
            processor=processor,
            registry=registry
        )
