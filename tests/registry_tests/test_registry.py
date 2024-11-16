# tests/test_registry_service/test_registry.py
import pytest
import asyncio
import etcd3
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from services.registry_service.domain.interfaces import ServiceInfo, HealthStatus
from services.registry_service.infrastructure.etcd_registry import EtcdRegistry
from services.registry_service.application.service import RegistryService

@pytest.fixture
def etcd_client():
    """Provide a mock etcd client"""
    with patch('etcd3.client') as mock_client:
        yield mock_client

@pytest.fixture
def test_config():
    """Provide test configuration"""
    return {
        'service': {
            'name': 'registry_service',
            'host': 'localhost',
            'port': 50051,
            'etcd': {
                'hosts': ['localhost'],
                'port': 2379,
                'timeout': 1,
                'retry': {
                    'max_attempts': 1,
                    'initial_delay': 0,
                    'max_delay': 1
                }
            },
            'health_check': {
                'interval': 1,
                'timeout': 1
            },
            'lease': {
                'ttl': 5
            }
        }
    }

class TestRegistryService:
    """Test registry service functionality"""

    @pytest.mark.asyncio
    async def test_service_registration(self, etcd_client, test_config):
        """Test service registration"""
        registry = EtcdRegistry(test_config['service'])
        
        service_info = ServiceInfo(
            name="test_service",
            host="localhost",
            port=8080,
            metadata={"version": "1.0.0"},
            last_heartbeat=datetime.now(),
            status="STARTING",
            version="1.0.0"
        )
        
        service_id = await registry.register(service_info)
        assert service_id == "test_service-localhost-8080"
        etcd_client.return_value.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_deregistration(self, etcd_client, test_config):
        """Test service deregistration"""
        registry = EtcdRegistry(test_config['service'])
        
        success = await registry.deregister("test-service-id")
        assert success
        etcd_client.return_value.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_service(self, etcd_client, test_config):
        """Test getting service information"""
        registry = EtcdRegistry(test_config['service'])
        
        # Mock etcd response
        mock_value = {
            "name": "test_service",
            "host": "localhost",
            "port": 8080,
            "metadata": {"version": "1.0.0"},
            "last_heartbeat": datetime.now().isoformat(),
            "status": "RUNNING",
            "version": "1.0.0"
        }
        etcd_client.return_value.get_prefix.return_value = [(
            bytes(str(mock_value), 'utf-8'),
            None
        )]
        
        service = await registry.get_service("test_service")
        assert service is not None
        assert service.name == "test_service"
        assert service.status == "RUNNING"

class TestHealthChecks:
    """Test health check functionality"""

    @pytest.mark.asyncio
    async def test_health_update(self, etcd_client, test_config):
        """Test health status updates"""
        registry = EtcdRegistry(test_config['service'])
        
        status = HealthStatus(
            is_healthy=True,
            last_check=datetime.now(),
            message="Service is healthy"
        )
        
        success = await registry.update_health("test-service-id", status)
        assert success
        etcd_client.return_value.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_loop(self, etcd_client, test_config):
        """Test health check loop"""
        service = RegistryService(test_config)
        
        # Start health check loop
        await service.start()
        
        # Let it run for a bit
        await asyncio.sleep(2)
        
        # Stop service
        await service.stop()
        
        # Verify health checks were performed
        etcd_client.return_value.get_prefix.assert_called()

class TestErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio
    async def test_etcd_connection_failure(self, etcd_client, test_config):
        """Test handling of etcd connection failures"""
        etcd_client.return_value.put.side_effect = etcd3.exceptions.ConnectionFailedError()
        
        registry = EtcdRegistry(test_config['service'])
        service_info = ServiceInfo(
            name="test_service",
            host="localhost",
            port=8080,
            metadata={},
            last_heartbeat=datetime.now(),
            status="STARTING",
            version="1.0.0"
        )
        
        with pytest.raises(etcd3.exceptions.ConnectionFailedError):
            await registry.register(service_info)

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, etcd_client, test_config):
        """Test retry mechanism for transient failures"""
        # Make the first call fail but second succeed
        etcd_client.return_value.put.side_effect = [
            etcd3.exceptions.ConnectionFailedError(),
            None
        ]
        
        registry = EtcdRegistry(test_config['service'])
        service_info = ServiceInfo(
            name="test_service",
            host="localhost",
            port=8080,
            metadata={},
            last_heartbeat=datetime.now(),
            status="STARTING",
            version="1.0.0"
        )
        
        service_id = await registry.register(service_info)
        assert service_id == "test_service-localhost-8080"
        assert etcd_client.return_value.put.call_count == 2

class TestConfigManagement:
    """Test configuration management"""

    @pytest.mark.asyncio
    async def test_set_config(self, etcd_client, test_config):
        """Test setting configuration values"""
        registry = EtcdRegistry(test_config['service'])
        
        success = await registry.set("test_key", "test_value")
        assert success
        etcd_client.return_value.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_config(self, etcd_client, test_config):
        """Test getting configuration values"""
        registry = EtcdRegistry(test_config['service'])
        
        etcd_client.return_value.get.return_value = (b"test_value", None)
        
        value = await registry.get("test_key")
        assert value == "test_value"
        etcd_client.return_value.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_watch_config(self, etcd_client, test_config):
        """Test configuration change notifications"""
        registry = EtcdRegistry(test_config['service'])
        
        # Mock watch events
        mock_event = Mock()
        mock_event.key = b"/config/test_key"
        mock_event.value = b"new_value"
        
        etcd_client.return_value.watch_prefix.return_value = (
            [mock_event],
            lambda: None
        )
        
        changes = []
        async for key, value in registry.watch_prefix("test_key"):
            changes.append((key, value))
            break
        
        assert len(changes) == 1
        assert changes[0] == ("test_key", "new_value")
