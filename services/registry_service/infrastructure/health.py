# services/registry_service/infrastructure/health.py
from dataclasses import dataclass
from datetime import datetime
import psutil
import asyncio

@dataclass
class ServiceHealth:
    cpu_percent: float
    memory_percent: float
    connected_services: int
    last_check: datetime
    etcd_healthy: bool

class HealthMonitor:
    def __init__(self, registry_service):
        self.registry_service = registry_service
        self.health_task = None

    async def start(self):
        self.health_task = asyncio.create_task(self._health_check_loop())

    async def stop(self):
        if self.health_task:
            self.health_task.cancel()
            try:
                await self.health_task
            except asyncio.CancelledError:
                pass

    async def _health_check_loop(self):
        while True:
            try:
                health = await self._check_health()
                # Update metrics or status
                await asyncio.sleep(30)
            except Exception as e:
                # Log error
                await asyncio.sleep(5)

    async def _check_health(self) -> ServiceHealth:
        return ServiceHealth(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            connected_services=len(await self.registry_service.registry.list_services()),
            last_check=datetime.now(),
            etcd_healthy=await self._check_etcd()
        )

    async def _check_etcd(self) -> bool:
        try:
            await self.registry_service.registry.get("health_check")
            return True
        except Exception:
            return False
