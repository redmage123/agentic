# microservices/logging_service/main.py
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import asyncio
from pathlib import Path
from typing import Any

from .domain.config import LoggingServiceConfig
from .application.service import LoggingService
from .infrastructure.service_factory import LoggingServiceFactory

def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__).resolve()  # Get the full path of main.py
    # Go up three levels: main.py -> logging_service -> microservices -> project_root
    return current_file.parent.parent.parent


def get_config_path() -> Path:
    """Get the configuration directory path."""
    return get_project_root() / "conf"

# Get the root project directory

# Use these functions to set the paths
PROJECT_ROOT = get_project_root()
CONFIG_PATH = get_config_path()


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> Any:
    """
    Main entry point for the logging service.
    
    The configuration hierarchy is:
    1. Base config (config.yaml)
    2. Schema definitions (schema/*.yaml)
    3. Component defaults (components/logging/default.yaml)
    4. Environment overrides (env/*.yaml)
    5. Command-line overrides
    """
    # Set service name for config resolution
    cfg["service_name"] = "logging"
    
    # Convert to structured config
    config = OmegaConf.to_object(cfg)
    
    # Create factory and service context
    factory = LoggingServiceFactory.from_config(config)
    context = factory.create_context()
    
    # Initialize service
    service = LoggingService(context)
    
    # Run the service
    asyncio.run(service.run())

if __name__ == "__main__":
    main()
