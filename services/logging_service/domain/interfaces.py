# services/logging_service/domain/interfaces.py
from abc import ABC, abstractmethod
from typing import Protocol, AsyncIterator, Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from .models import LogEntry, LogStats, LogLevel
from .config import ServiceConfig, LoggingConfig, HandlerConfig

class ConfigurationProvider(Protocol):
    """Interface for configuration management using Hydra"""
    
    @property
    def config(self) -> ServiceConfig:
        """Get the current service configuration"""
        raise NotImplementedError
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration at runtime
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            bool: True if update was successful
        """
        raise NotImplementedError
    
    def get_handler_config(self, handler_name: str) -> HandlerConfig:
        """
        Get configuration for a specific handler
        
        Args:
            handler_name: Name of the handler
            
        Returns:
            HandlerConfig: Handler configuration
        """
        raise NotImplementedError

class LogWriter(Protocol):
    """Interface for log writing implementations"""
    
    def __init__(self, config: LoggingConfig):
        """
        Initialize with Hydra configuration
        
        Args:
            config: Logging configuration from Hydra
        """
        raise NotImplementedError
    
    async def write_log(self, entry: LogEntry) -> bool:
        """
        Write a log entry to the underlying storage
        
        Args:
            entry: The log entry to write
            
        Returns:
            bool: True if write was successful
        """
        raise NotImplementedError

    async def rotate_if_needed(self) -> bool:
        """
        Perform log rotation based on Hydra configuration
        
        Returns:
            bool: True if rotation was successful or not needed
        """
        raise NotImplementedError
    
    async def flush(self) -> bool:
        """
        Force flush any buffered logs
        
        Returns:
            bool: True if flush was successful
        """
        raise NotImplementedError
    
    def get_current_log_file(self) -> Path:
        """
        Get the current active log file path
        
        Returns:
            Path: Path to the current log file
        """
        raise NotImplementedError

class LogRepository(Protocol):
    """Interface for log storage operations"""
    
    def __init__(self, config: ServiceConfig):
        """
        Initialize with Hydra configuration
        
        Args:
            config: Service configuration from Hydra
        """
        raise NotImplementedError
    
    async def save(self, entry: LogEntry) -> bool:
        """
        Save a log entry to the repository
        
        Args:
            entry: The log entry to save
            
        Returns:
            bool: True if save was successful
        """
        raise NotImplementedError
    
    async def get_stats(self) -> LogStats:
        """
        Get current logging statistics
        
        Returns:
            LogStats: Current logging statistics
        """
        raise NotImplementedError
    
    async def get_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[LogLevel] = None,
        service_name: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = None  # Uses Hydra config default if None
    ) -> AsyncIterator[LogEntry]:
        """
        Query logs with optional filters
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            level: Optional log level filter
            service_name: Optional service name filter
            correlation_id: Optional correlation ID filter
            limit: Maximum number of entries to return
            
        Returns:
            AsyncIterator[LogEntry]: Iterator over matching log entries
        """
        raise NotImplementedError

class LogProcessor(Protocol):
    """Interface for log processing and enrichment"""
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize with Hydra configuration
        
        Args:
            config: Processing configuration from Hydra
        """
        raise NotImplementedError
    
    async def process_log(self, entry: LogEntry) -> LogEntry:
        """
        Process and enrich a log entry
        
        Args:
            entry: Original log entry
            
        Returns:
            LogEntry: Processed and enriched log entry
        """
        raise NotImplementedError
    
    async def batch_process(self, entries: List[LogEntry]) -> List[LogEntry]:
        """
        Process multiple log entries in batch based on Hydra configuration
        
        Args:
            entries: List of log entries to process
            
        Returns:
            List[LogEntry]: Processed log entries
        """
        raise NotImplementedError

class RegistryClient(Protocol):
    """Interface for service registry interactions"""
    
    def __init__(self, config: RegistryConfig):
        """
        Initialize with Hydra configuration
        
        Args:
            config: Registry configuration from Hydra
        """
        raise NotImplementedError
    
    async def register_service(self) -> bool:
        """
        Register service using configuration from Hydra
        
        Returns:
            bool: True if registration was successful
        """
        raise NotImplementedError
    
    async def deregister_service(self) -> bool:
        """
        Deregister service
        
        Returns:
            bool: True if deregistration was successful
        """
        raise NotImplementedError
    
    async def start_heartbeat(self) -> None:
        """Start heartbeat based on Hydra configuration interval"""
        raise NotImplementedError
    
    async def stop_heartbeat(self) -> None:
        """Stop heartbeat"""
        raise NotImplementedError

@dataclass
class ServiceContext:
    """Context for dependency injection of configured components"""
    config: ServiceConfig
    writer: LogWriter
    repository: LogRepository
    processor: LogProcessor
    registry: RegistryClient

class ServiceFactory(Protocol):
    """Factory for creating configured service components"""
    
    @staticmethod
    def create_context(config_path: str) -> ServiceContext:
        """
        Create service context with all configured components
        
        Args:
            config_path: Path to Hydra configuration
            
        Returns:
            ServiceContext: Configured service context
        """
        raise NotImplementedError
