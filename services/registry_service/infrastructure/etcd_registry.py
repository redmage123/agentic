# microservices/registry_service/infrastructure/etcd_registry.py
import etcd3
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, AsyncIterator
from tenacity import retry, stop_after_attempt, wait_exponential

from ..domain.interfaces import ServiceRegistry, ServiceInfo, HealthStatus, KeyValueStore

class EtcdRegistry(ServiceRegistry, KeyValueStore):
    """Etcd-based implementation of service registry"""
    
    def __init__(self, config: dict):
        self.config = config
        self.client = etcd3.client(
            host=config['etcd']['hosts'][0],
            port=config['etcd']['port']
        )
        self.service_prefix = "/services/"
        self.config_prefix = "/config/"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def register(self, service_info: ServiceInfo) -> str:
        """Register a service with etcd"""
        service_id = f"{service_info.name}-{service_info.host}-{service_info.port}"
        key = f"{self.service_prefix}{service_id}"
        value = json.dumps({
            "name": service_info.name,
            "host": service_info.host,
            "port": service_info.port,
            "metadata": service_info.metadata,
            "last_heartbeat": datetime.now().isoformat(),
            "status": service_info.status,
            "version": service_info.version
        })
        
        lease = self.client.lease(ttl=self.config['lease']['ttl'])
        self.client.put(key, value, lease=lease)
        return service_id
    
    async def deregister(self, service_id: str) -> bool:
        """Deregister a service from etcd"""
        key = f"{self.service_prefix}{service_id}"
        self.client.delete(key)
        return True
    
    async def get_service(self, service_name: str) -> Optional[ServiceInfo]:
        """Get service information from etcd"""
        for value, metadata in self.client.get_prefix(f"{self.service_prefix}{service_name}"):
            if value:
                data = json.loads(value)
                return ServiceInfo(**data)
        return None
    
    async def list_services(self) -> List[ServiceInfo]:
        """List all registered services"""
        services = []
        for value, metadata in self.client.get_prefix(self.service_prefix):
            if value:
                data = json.loads(value)
                services.append(ServiceInfo(**data))
        return services
    
    async def watch_services(self) -> AsyncIterator[ServiceInfo]:
        """Watch for service changes"""
        events_iterator, cancel = self.client.watch_prefix(self.service_prefix)
        try:
            for event in events_iterator:
                if event.value:
                    data = json.loads(event.value)
                    yield ServiceInfo(**data)
        finally:
            cancel()
    
    async def update_health(self, service_id: str, status: HealthStatus) -> bool:
        """Update service health status"""
        key = f"{self.service_prefix}{service_id}/health"
        value = json.dumps({
            "is_healthy": status.is_healthy,
            "last_check": status.last_check.isoformat(),
            "message": status.message
        })
        self.client.put(key, value)
        return True
    
    # KeyValueStore implementation
    async def set(self, key: str, value: str) -> bool:
        """Set a configuration value"""
        full_key = f"{self.config_prefix}{key}"
        self.client.put(full_key, value)
        return True
    
    async def get(self, key: str) -> Optional[str]:
        """Get a configuration value"""
        full_key = f"{self.config_prefix}{key}"
        value = self.client.get(full_key)
        return value[0].decode('utf-8') if value[0] else None
    
    async def watch_prefix(self, prefix: str) -> AsyncIterator[tuple[str, str]]:
        """Watch for configuration changes"""
        full_prefix = f"{self.config_prefix}{prefix}"
        events_iterator, cancel = self.client.watch_prefix(full_prefix)
        try:
            for event in events_iterator:
                if event.value:
                    key = event.key.decode('utf-8').removeprefix(self.config_prefix)
                    yield key, event.value.decode('utf-8')
        finally:
            cancel()
