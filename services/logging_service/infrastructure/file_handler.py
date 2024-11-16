# services/logging_service/infrastructure/file_handler.py
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from ..domain.interfaces import LogWriter
from ..domain.models import LogEntry

class FileLogWriter:
    """Concrete implementation of log writer using rotating files"""
    def __init__(self, log_dir: Path, max_bytes: int, backup_count: int):
        self.log_dir = log_dir
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._setup_handler()

    def _setup_handler(self):
        """Initialize the rotating file handler"""
        self.log_dir.mkdir(exist_ok=True)
        log_file = self.log_dir / "service.log"
        
        self.handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(correlation_id)s] - %(service_name)s - %(message)s'
        )
        self.handler.setFormatter(formatter)

    async def write_log(self, entry: LogEntry) -> bool:
        """Write a log entry to file"""
        try:
            record = logging.LogRecord(
                name=entry.service_name,
                level=entry.level.value,
                pathname="",
                lineno=0,
                msg=entry.message,
                args=(),
                exc_info=None
            )
            
            # Add extra fields
            record.correlation_id = entry.correlation_id
            record.service_name = entry.service_name
            
            self.handler.emit(record)
            return True
            
        except Exception:
            return False

    async def rotate_if_needed(self) -> bool:
        """Check and perform rotation if needed"""
        try:
            self.handler.doRollover()
            return True
        except Exception:
            return False
