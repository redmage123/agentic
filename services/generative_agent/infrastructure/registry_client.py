# services/generative_agent/infrastructure/registry_client.py
import grpc
import asyncio
from typing import Optional

from services.protos import registry_service_pb2 as pb2
from services.protos import registry_service_pb2_grpc as pb2_grpc
from ..domain.exceptions import RegistryError

class RegistryClient:
    def __init__(
       self,
       host: str = "localhost",
       port: int = 50050,
       service_name: str = "generative_agent",
       retry_interval: int = 30
   ):
       self.address = f"{host}:{port}"
       self.service_name = service_name
       self.retry_interval = retry_interval
       self.channel = None
       self.stub = None
       self._heartbeat_task = None
       self._registered = False

   async def connect(self):
       if not self.channel:
           self.channel = grpc.aio.insecure_channel(self.address)
           self.stub = pb2_grpc.RegistryServiceStub(self.channel)

   async def start(self):
       await self.connect()
       await self.register_service()
       self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

   async def stop(self):
       if self._heartbeat_task:
           self._heartbeat_task.cancel()
           await self.deregister_service()
       if self.channel:
           await self.channel.close()

   async def register_service(self) -> bool:
       try:
           await self.connect()
           response = await self.stub.Register(pb2.RegisterRequest(
               service_name=self.service_name,
               host="localhost",
               port=50056,  # Generative Agent port
               metadata={"type": "ai_agent", "model": "claude-3"}
           ))
           self._registered = response.success
           return response.success
       except Exception as e:
           raise RegistryError(f"Registration failed: {str(e)}")

   async def deregister_service(self) -> bool:
       if not self._registered:
           return True
       try:
           response = await self.stub.Deregister(pb2.DeregisterRequest(
               service_name=self.service_name
           ))
           self._registered = not response.success
           return response.success
       except Exception as e:
           raise RegistryError(f"Deregistration failed: {str(e)}")

   async def _heartbeat_loop(self):
       while True:
           try:
               await asyncio.sleep(self.retry_interval)
               await self._send_heartbeat()
           except asyncio.CancelledError:
               break
           except Exception:
               continue

   async def _send_heartbeat(self):
       try:
           await self.stub.ReportHealth(pb2.HealthReport(
               service_id=self.service_name,
               is_healthy=True,
               message="Generative Agent is healthy"
           ))
       except Exception as e:
           self._registered = False
           raise RegistryError(f"Heartbeat failed: {str(e)}")
