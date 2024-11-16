# tests/test_logging_service/test_unit.py
"""
Unit tests for the logging service code.
These tests verify the core logging functionality, configuration handling,
and individual component behavior.

Key features of these unit tests:

1. **Test Categories**:
   - LogWriter: Basic functionality tests
   - ConfigValidation: Configuration validation tests
   - MetadataHandling: Metadata processing tests
   - ErrorHandling: Error condition tests
   - Concurrency: Concurrent operation tests
   - Formatting: Log format tests

2. **Fixtures**:
   - test_config: Provides test configuration
   - temp_log_dir: Provides temporary log directory

3. **Test Coverage**:
   - Basic logging functionality
   - Log rotation
   - Configuration validation
   - Error handling
   - Concurrent operations
   - Format options
   - Metadata handling
"""

import pytest
from datetime import datetime
from pathlib import Path
import json
import os
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock
from omegaconf import OmegaConf

from microservices.logging_service.domain.models import LogEntry, LogLevel
from microservices.logging_service.infrastructure.file_handler import FileLogWriter
from microservices.logging_service.domain.interfaces import LogWriter, RegistryClient
from microservices.logging_service.infrastructure.service_factory import LoggingServiceFactory

@pytest.fixture
def test_config():
    """Provide test configuration"""
    config = {
        "service": {
            "name": "logging_service",
            "host": "localhost",
            "port": 50052,
            "log_handling": {
                "root_dir": "/tmp/test_logs",
                "format": "DETAILED",
                "default_level": "DEBUG",
                "retention_days": 1,
                "handlers": {
                    "file": {
                        "enabled": True,
                        "path": "/tmp/test_logs/service.log",
                        "rotation": {
                            "max_bytes": 1024,
                            "backup_count": 3
                        }
                    },
                    "console": {
                        "enabled": True
                    }
                }
            }
        }
    }
    return OmegaConf.create(config)

@pytest.fixture
def temp_log_dir(tmp_path):
    """Provide temporary directory for logs"""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir

class TestLogWriter:
    """Tests for the core logging functionality"""

    @pytest.mark.asyncio
    async def test_basic_logging(self, temp_log_dir, test_config):
        """Test basic log writing functionality"""
        test_config.service.log_handling.root_dir = str(temp_log_dir)
        writer = FileLogWriter(test_config.service.log_handling)
        
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            correlation_id="test-123",
            metadata={"test": "value"}
        )
        
        success = await writer.write_log(entry)
        assert success
        
        log_file = Path(temp_log_dir) / "service.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content
        assert "test-123" in content
        assert "value" in content

    @pytest.mark.asyncio
    async def test_log_levels(self, temp_log_dir, test_config):
        """Test logging at different severity levels"""
        test_config.service.log_handling.root_dir = str(temp_log_dir)
        writer = FileLogWriter(test_config.service.log_handling)
        
        for level in LogLevel:
            entry = LogEntry(
                timestamp=datetime.now(),
                level=level,
                message=f"Test {level.name}",
                correlation_id="test-123",
                metadata={}
            )
            await writer.write_log(entry)
            
            content = (Path(temp_log_dir) / "service.log").read_text()
            assert level.name in content
            assert f"Test {level.name}" in content

    @pytest.mark.asyncio
    async def test_log_rotation(self, temp_log_dir, test_config):
        """Test log file rotation"""
        test_config.service.log_handling.root_dir = str(temp_log_dir)
        test_config.service.log_handling.handlers.file.rotation.max_bytes = 100
        writer = FileLogWriter(test_config.service.log_handling)
        
        # Write enough to trigger rotation
        for i in range(10):
            entry = LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="X" * 20,  # Message long enough to trigger rotation
                correlation_id=f"test-{i}",
                metadata={}
            )
            await writer.write_log(entry)
        
        log_files = list(Path(temp_log_dir).glob("service.log*"))
        assert len(log_files) > 1

class TestConfigValidation:
    """Tests for configuration validation"""

    def test_required_fields(self, test_config):
        """Test validation of required configuration fields"""
        # Remove required field
        del test_config.service.log_handling.root_dir
        
        with pytest.raises(ValueError, match="root_dir is required"):
            FileLogWriter(test_config.service.log_handling)

    def test_invalid_log_level(self, test_config):
        """Test validation of log level configuration"""
        test_config.service.log_handling.default_level = "INVALID_LEVEL"
        
        with pytest.raises(ValueError, match="Invalid log level"):
            FileLogWriter(test_config.service.log_handling)

    def test_invalid_rotation_config(self, test_config):
        """Test validation of rotation configuration"""
        test_config.service.log_handling.handlers.file.rotation.max_bytes = -1
        
        with pytest.raises(ValueError, match="Invalid max_bytes"):
            FileLogWriter(test_config.service.log_handling)

class TestMetadataHandling:
    """Tests for metadata handling in logs"""

    @pytest.mark.asyncio
    async def test_metadata_formatting(self, temp_log_dir, test_config):
        """Test metadata formatting in logs"""
        test_config.service.log_handling.root_dir = str(temp_log_dir)
        writer = FileLogWriter(test_config.service.log_handling)
        
        metadata = {
            "request_id": "req-123",
            "user_id": "user-456",
            "service": "test-service"
        }
        
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test with metadata",
            correlation_id="test-123",
            metadata=metadata
        )
        
        await writer.write_log(entry)
        
        content = (Path(temp_log_dir) / "service.log").read_text()
        for key, value in metadata.items():
            assert value in content

class TestErrorHandling:
    """Tests for error handling"""

    @pytest.mark.asyncio
    async def test_write_permission_error(self, temp_log_dir, test_config):
        """Test handling of write permission errors"""
        test_config.service.log_handling.root_dir = str(temp_log_dir)
        writer = FileLogWriter(test_config.service.log_handling)
        
        # Remove write permissions
        os.chmod(temp_log_dir, 0o444)
        
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            correlation_id="test-123",
            metadata={}
        )
        
        with pytest.raises(PermissionError):
            await writer.write_log(entry)

    @pytest.mark.asyncio
    async def test_disk_full_handling(self, temp_log_dir, test_config, monkeypatch):
        """Test handling of disk full condition"""
        test_config.service.log_handling.root_dir = str(temp_log_dir)
        writer = FileLogWriter(test_config.service.log_handling)
        
        def mock_write(*args, **kwargs):
            raise OSError("No space left on device")
        
        monkeypatch.setattr("builtins.open", mock_write)
        
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            correlation_id="test-123",
            metadata={}
        )
        
        with pytest.raises(OSError, match="No space left on device"):
            await writer.write_log(entry)

class TestConcurrency:
    """Tests for concurrent operations"""

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, temp_log_dir, test_config):
        """Test concurrent log writing"""
        test_config.service.log_handling.root_dir = str(temp_log_dir)
        writer = FileLogWriter(test_config.service.log_handling)
        
        async def write_log(index: int):
            entry = LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message=f"Concurrent message {index}",
                correlation_id=f"test-{index}",
                metadata={}
            )
            return await writer.write_log(entry)
        
        # Create multiple concurrent writes
        tasks = [write_log(i) for i in range(100)]
        results = await asyncio.gather(*tasks)
        
        # Verify all writes succeeded
        assert all(results)
        
        # Verify log content
        content = (Path(temp_log_dir) / "service.log").read_text()
        for i in range(100):
            assert f"Concurrent message {i}" in content

class TestFormatting:
    """Tests for log formatting"""

    @pytest.mark.asyncio
    async def test_format_options(self, temp_log_dir, test_config):
        """Test different log format options"""
        test_config.service.log_handling.root_dir = str(temp_log_dir)
        formats = ["SIMPLE", "DETAILED", "JSON"]
        
        for format_type in formats:
            test_config.service.log_handling.format = format_type
            writer = FileLogWriter(test_config.service.log_handling)
            
            entry = LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="Test format",
                correlation_id="test-123",
                metadata={"format": format_type}
            )
            
            await writer.write_log(entry)
            
            content = (Path(temp_log_dir) / "service.log").read_text()
            if format_type == "JSON":
                # Verify JSON format
                try:
                    log_data = json.loads(content.splitlines()[-1])
                    assert log_data["message"] == "Test format"
                    assert log_data["correlation_id"] == "test-123"
                except json.JSONDecodeError:
                    pytest.fail("Invalid JSON format")
