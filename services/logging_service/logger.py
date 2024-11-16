"""
Advanced Structured Logging System for Multi-Agent AI Architecture

This module implements a modern, structured logging system specifically designed for 
distributed multi-agent AI systems. It addresses several key requirements:

1. Structured Logging: 
   - Supports multiple output formats including JSON for machine parsing
   - Maintains context across asynchronous operations
   - Enables correlation of logs across different agents

2. Performance and Reliability:
   - Uses rotating file handlers to prevent disk space issues
   - Implements efficient context management
   - Provides thread-safe operations

3. Extensibility:
   - Supports custom handlers for specialized logging needs
   - Allows for dynamic log level adjustment
   - Enables addition of custom context fields

4. Debugging and Monitoring:
   - Maintains request context across async boundaries
   - Provides detailed formatting for development
   - Supports automated log analysis

The system is designed to be both powerful and easy to use, following the principle
of "batteries included but swappable."

Author: [Your Name]
Date: November 2024
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial, wraps
from logging import Logger, getLogger, StreamHandler, Formatter
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Protocol, Dict, Any, List
import json
import logging
from datetime import datetime
import contextvars
import threading

# Thread-local storage for request tracking
request_id_var = contextvars.ContextVar('request_id', default=None)
_thread_local = threading.local()

class LogLevel(Enum):
    """
    Enumeration of logging levels with corresponding standard logging values.
    
    Using an enum instead of direct integers provides:
    - Type safety
    - Self-documentation
    - IDE autocompletion
    - Prevention of invalid level assignments
    """
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class LogFormat(Enum):
    """
    Defines available log output formats.
    
    SIMPLE: Basic logging for development
    DETAILED: Extended information for debugging
    JSON: Machine-readable format for log aggregation systems
    """
    SIMPLE = auto()
    DETAILED = auto()
    JSON = auto()

@dataclass(frozen=True)
class LogConfig:
    """
    Immutable configuration for logger instances.
    
    This class uses a frozen dataclass to ensure configuration immutability,
    preventing runtime modifications that could lead to inconsistent logging
    behavior. Default values are carefully chosen for production use while
    remaining flexible for development.
    
    Attributes:
        name: Logger identifier, used to trace log sources
        level: Minimum level for log capture
        log_format: Output format selection
        file_path: Optional log file location
        max_bytes: Size threshold for log rotation
        backup_count: Number of rotated logs to maintain
        additional_fields: Extra fields to include in all log records
    """
    name: str
    level: LogLevel
    log_format: LogFormat
    file_path: Optional[Path] = None
    max_bytes: int = 5 * 1024 * 1024  # 5MB default
    backup_count: int = 5
    additional_fields: Dict[str, Any] = field(default_factory=dict)

class LogHandler(Protocol):
    """
    Protocol defining the interface for custom log handlers.
    
    This protocol enables extension of logging functionality without modifying
    the core logger. Implementations might include:
    - Network transport handlers
    - Database logging
    - Metrics collection
    - Alert systems
    
    The protocol approach allows for runtime handler swapping and testing
    with mock implementations.
    """
    def handle(self, record: Dict[str, Any]) -> None:
        """Process a single log record"""
        ...

@dataclass
class LogRecord:
    """
    Represents a structured log entry with contextual information.
    
    This class serves as both a data container and a validator for log
    records. Using a dataclass provides:
    - Automatic initialization
    - Runtime type checking
    - Easy serialization
    
    The request_id is automatically populated from context variables,
    enabling request tracking across async boundaries.
    """
    timestamp: datetime
    level: LogLevel
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = field(default_factory=lambda: request_id_var.get(None))

class StructuredLogger:
    """
    Core logging implementation with structured output and context management.
    
    This class provides a modern logging interface that addresses common
    challenges in distributed systems:
    - Correlation of related log entries
    - Consistent structured output
    - Context preservation
    - Extensible handling
    
    It's designed to be used as a singleton per application component but
    can also be instantiated multiple times with different configurations
    if needed.
    """
    
    def __init__(self, config: LogConfig):
        """
        Initialize logger with provided configuration.
        
        Args:
            config: Immutable configuration instance
        
        The initialization process:
        1. Stores configuration
        2. Sets up basic logger
        3. Configures handlers
        4. Initializes handler registry
        """
        self.config = config
        self.logger = self._setup_logger()
        self.handlers: List[LogHandler] = []
        self._lock = threading.Lock()
    
    def _setup_logger(self) -> Logger:
        """
        Create and configure the underlying logger instance.
        
        This method handles:
        1. Base logger creation
        2. Handler setup based on configuration
        3. Formatter assignment
        
        Returns:
            Configured logging.Logger instance
        
        The setup process ensures thread-safety and proper resource cleanup
        by removing existing handlers before adding new ones.
        """
        logger = getLogger(self.config.name)
        logger.setLevel(self.config.level.value)
        
        with self._lock:
            logger.handlers.clear()
            
            match self.config.log_format:
                case LogFormat.JSON:
                    formatter = self._create_json_formatter()
                case LogFormat.DETAILED:
                    formatter = self._create_detailed_formatter()
                case _:
                    formatter = self._create_simple_formatter()
            
            console_handler = StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            if self.config.file_path:
                file_handler = RotatingFileHandler(
                    self.config.file_path,
                    maxBytes=self.config.max_bytes,
                    backupCount=self.config.backup_count
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def _create_json_formatter(self) -> Formatter:
        """
        Create a JSON formatter for machine-readable logs.
        
        This formatter structures logs for easy parsing by log aggregation
        tools. It includes:
        - Standard fields (timestamp, level, message)
        - Request context
        - Additional configured fields
        - Custom context
        
        Returns:
            Formatter that produces JSON-formatted log entries
        """
        return Formatter(
            lambda record: json.dumps({
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'message': record.getMessage(),
                'logger': record.name,
                'request_id': request_id_var.get(None),
                **self.config.additional_fields,
                **getattr(record, 'context', {})
            })
        )
    
    def _create_detailed_formatter(self) -> Formatter:
        """
        Create a detailed formatter for debugging and development.
        
        This formatter provides maximum information for human readers,
        including timing, context, and request correlation.
        
        Returns:
            Formatter that produces detailed human-readable logs
        """
        return Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(request_id)s - %(message)s'
        )
    
    def _create_simple_formatter(self) -> Formatter:
        """
        Create a simple formatter for basic logging needs.
        
        This formatter provides essential information in a concise format,
        suitable for development and simple applications.
        
        Returns:
            Formatter that produces simplified log entries
        """
        return Formatter('%(name)s - %(levelname)s - %(message)s')
    
    def add_handler(self, handler: LogHandler) -> None:
        """
        Add a custom handler to the logger.
        
        This method allows runtime extension of logging capabilities.
        Handlers are executed in addition to standard logging.
        
        Args:
            handler: Custom handler implementing LogHandler protocol
        """
        with self._lock:
            self.handlers.append(handler)
    
    def _log(self, level: LogLevel, message: str, context: Dict[str, Any] = None) -> None:
        """
        Internal logging method that handles both standard and custom handlers.
        
        This method:
        1. Creates a structured log record
        2. Logs through standard handlers
        3. Processes through custom handlers
        4. Handles context propagation
        
        Args:
            level: Log severity level
            message: Log message
            context: Optional contextual information
        """
        record = LogRecord(
            timestamp=datetime.now(),
            level=level,
            message=message,
            context=context or {},
        )
        
        extra = {
            'request_id': record.request_id,
            'context': record.context
        }
        
        with self._lock:
            self.logger.log(level.value, message, extra=extra)
            
            for handler in self.handlers:
                handler.handle(record.__dict__)
    
    # Convenience methods using partial application for different log levels
    debug = partial(_log, level=LogLevel.DEBUG)
    info = partial(_log, level=LogLevel.INFO)
    warning = partial(_log, level=LogLevel.WARNING)
    error = partial(_log, level=LogLevel.ERROR)
    critical = partial(_log, level=LogLevel.CRITICAL)

class RequestContextManager:
    """
    Context manager for request tracking across async boundaries.
    
    This class provides a way to maintain request context throughout
    the processing lifetime of a request, even across async operations
    and thread boundaries.
    """
    
    def __init__(self, request_id: str):
        """
        Initialize context manager with request ID.
        
        Args:
            request_id: Unique identifier for the request
        """
        self.request_id = request_id
        self.token = None
    
    def __enter__(self):
        """Set request context on entry"""
        self.token = request_id_var.set(self.request_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clear request context on exit"""
        request_id_var.reset(self.token)

def with_request_context(func):
    """
    Decorator to automatically handle request context.
    
    This decorator simplifies request context management by automatically
    setting up and tearing down context based on the request_id parameter.
    
    Args:
        func: Async function to wrap
    
    Returns:
        Wrapped function that handles request context
    """
    @wraps(func)
    async def wrapper(*args, request_id: str = None, **kwargs):
        if request_id:
            with RequestContextManager(request_id):
                return await func(*args, **kwargs)
        return await func(*args, **kwargs)
    return wrapper

# Create default logger instance with production-ready configuration
default_config = LogConfig(
    name="TCA",
    level=LogLevel.INFO,
    log_format=LogFormat.DETAILED,
    file_path=Path("tca_service.log")
)

# Initialize the global logger instance
logger = StructuredLogger(default_config)
