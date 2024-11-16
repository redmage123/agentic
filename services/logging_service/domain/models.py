# services/logging_service/domain/models.py
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

class LogLevel(Enum):
    """Value object representing log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass(frozen=True)
class LogEntry:
    """Value object representing a log entry"""
    service_name: str
    level: LogLevel
    message: str
    timestamp: datetime
    correlation_id: str
    metadata: Dict[str, str]

@dataclass
class LogStats:
    """Entity tracking logging statistics"""
    messages_processed: int = 0
    bytes_written: int = 0
    current_log_file: str = ""
