# microservices/registry_service/domain/interfaces.py
from abc import ABC, abstractmethod
from typing import Protocol, List, Dict, Optional, AsyncIterator
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ServiceInfo:
    """Information about a registered service"""
    name: str
    host: str
    port: int
    metadata: Dict[str, str]
    last_heartbeat: datetime
    status: str
    version: str

@dataclass
class HealthStatus:
    """Health status of a service"""
    is_healthy: bool
    last_check: datetime
    message: str

class ServiceRegistry(Protocol):
    """Protocol for service registry operations"""
    
    async def register(self, service_info: ServiceInfo) -> str:
        """Register a service and return its ID"""
        pass
    
    async def deregister(self, service_id: str) -> bool:
        """Deregister a service"""
        pass
    
    async def get_service(self, service_name: str) -> Optional[ServiceInfo]:
        """Get information about a specific service"""
        pass
    
    async def list_services(self) -> List[ServiceInfo]:
        """List all registered services"""
        pass
    
    async def watch_services(self) -> AsyncIterator[ServiceInfo]:
        """Watch for service changes"""
        pass
    
    async def update_health(self, service_id: str, status: HealthStatus) -> bool:
        """Update service health status"""
        pass

class KeyValueStore(Protocol):
    """Protocol for configuration storage"""
    
    async def set(self, key: str, value: str) -> bool:
        """Set a configuration value"""
        pass
    
    async def get(self, key: str) -> Optional[str]:
        """Get a configuration value"""
        pass
    
    async def delete(self, key: str) -> bool:
        """Delete a configuration value"""
        pass
    
    async def watch_prefix(self, prefix: str) -> AsyncIterator[tuple[str, str]]:
        """Watch for changes to keys with prefix"""
        pass
