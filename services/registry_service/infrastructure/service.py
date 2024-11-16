# services/registry_service/application/service.py
import asyncio
from datetime import datetime
from typing import Optional

from ..domain.interfaces import ServiceRegistry, ServiceInfo, HealthStatus
from ..infrastructure.etcd_registry import EtcdRegistry

class RegistryService:
    """Main registry service implementation"""
    
    def __init__(self, config: dict):
        self.config = config
        self.registry: Optional[EtcdRegistry] = None
        self.health_check_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the registry service"""
        self.registry = EtcdRegistry(self.config)
        self.health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop(self):
        """Stop the registry service"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
    
    async def _health_check_loop(self):
        """Periodic health check of registered services"""
        while True:
            try:
                services = await self.registry.list_services()
                for service in services:
                    try:
                        # Implement health check logic here
                        # This could be a gRPC call to the service's health check endpoint
                        pass
                    except Exception as e:
                        await self.registry.update_health(
                            service.name,
                            HealthStatus(
                                is_healthy=False,
                                last_check=datetime.now(),
                                message=str(e)
                            )
                        )
            except Exception as e:
                # Log error
                pass
            
            await asyncio.sleep(self.config['health_check']['interval'])
