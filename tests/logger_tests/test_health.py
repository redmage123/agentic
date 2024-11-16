# tests/test_logging_service/test_health.py

"""
Health and operational status tests for the logging microservice.
These tests verify that the service is running correctly, responding to healthchecks,
and managing resources appropriately.
"""

import pytest
import asyncio
import aiohttp
import psutil
from pathlib import Path
import os

@pytest.fixture
async def running_service(test_config, mock_registry_client, temp_log_dir):
    """Provides a running instance of the logging service"""
    test_config.service.health_check.enabled = True
    test_config.service.health_check.port = 8080
    test_config.service.log_handling.root_dir = str(temp_log_dir)
    
    service = LoggingService(
        config=test_config,
        writer=FileLogWriter(test_config.service.log_handling),
        registry_client=mock_registry_client
    )
    
    await service.start()
    yield service
    await service.stop()

class TestServiceHealth:
    """Tests for service health and operational status"""

    @pytest.mark.asyncio
    async def test_service_is_running(self, running_service):
        """Verify service is up and responding"""
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8080/health') as response:
                assert response.status == 200
                data = await response.json()
                assert data['status'] == 'healthy'
                assert data['service_name'] == 'logging_service'

    @pytest.mark.asyncio
    async def test_service_ports(self, running_service):
        """Verify service ports are open and listening"""
        connections = psutil.net_connections()
        listening_ports = [conn.laddr.port for conn in connections if conn.status == 'LISTEN']
        
        assert running_service.config.service.port in listening_ports  # Main service port
        assert 8080 in listening_ports  # Health check port

    @pytest.mark.asyncio
    async def test_resource_usage(self, running_service):
        """Verify service resource usage is within acceptable limits"""
        process = psutil.Process()
        
        # Memory check
        mem_info = process.memory_info()
        assert mem_info.rss < 500 * 1024 * 1024  # Less than 500MB RAM
        
        # CPU check
        cpu_percent = process.cpu_percent(interval=1.0)
        assert cpu_percent < 50  # Less than 50% CPU

class TestHealthEndpoints:
    """Tests for health check endpoints"""

    @pytest.mark.asyncio
    async def test_health_check(self, running_service):
        """Test main health check endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8080/health') as response:
                data = await response.json()
                assert response.status == 200
                assert 'status' in data
                assert 'version' in data
                assert 'timestamp' in data

    @pytest.mark.asyncio
    async def test_readiness_probe(self, running_service):
        """Test readiness probe endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8080/ready') as response:
                data = await response.json()
                assert response.status == 200
                assert data['ready'] is True

    @pytest.mark.asyncio
    async def test_liveness_probe(self, running_service):
        """Test liveness probe endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8080/live') as response:
                data = await response.json()
                assert response.status == 200
                assert data['alive'] is True

class TestDependencyHealth:
    """Tests for dependency health checks"""

    @pytest.mark.asyncio
    async def test_registry_connection(self, running_service):
        """Verify registry service connection"""
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8080/health/dependencies') as response:
                data = await response.json()
                assert data['dependencies']['registry']['status'] == 'connected'

    @pytest.mark.asyncio
    async def test_filesystem_access(self, running_service, temp_log_dir):
        """Verify filesystem access for logging"""
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8080/health/dependencies') as response:
                data = await response.json()
                assert data['dependencies']['file_system']['status'] == 'accessible'

class TestErrorRecovery:
    """Tests for service error recovery"""

    @pytest.mark.asyncio
    async def test_degraded_state_recovery(self, running_service):
        """Verify service can recover from degraded state"""
        # Simulate errors
        running_service._failed_operations = 3
        
        async with aiohttp.ClientSession() as session:
            # Check degraded state
            async with session.get('http://localhost:8080/health') as response:
                data = await response.json()
                assert data['status'] == 'degraded'
        
        # Reset errors
        running_service._failed_operations = 0
        await asyncio.sleep(1)
        
        async with aiohttp.ClientSession() as session:
            # Verify recovery
            async with session.get('http://localhost:8080/health') as response:
                data = await response.json()
                assert data['status'] == 'healthy'

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, running_service):
        """Verify service shuts down gracefully"""
        shutdown_task = asyncio.create_task(running_service.stop())
        
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8080/health') as response:
                data = await response.json()
                assert data['status'] == 'shutting_down'
        
        await shutdown_task
