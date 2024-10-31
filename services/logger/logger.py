import logging
from logging import Logger
from logging.handlers import RotatingFileHandler

def setup_logger(name: str = "TCA", log_file: str = "tca_service.log", level: int = logging.INFO) -> Logger:
    """
    Sets up a centralized logger with both console and file output.

    Parameters:
    - name (str): Logger name.
    - log_file (str): File to log messages to.
    - level (int): Logging level (default: logging.INFO).

    Returns:
    - Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # File handler with rotation
    file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

# Initialize a logger instance that can be imported elsewhere
logger: Logger = setup_logger()

