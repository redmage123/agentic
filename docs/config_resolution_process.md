"""
Complete Hydra Configuration Resolution Process for Logging Service

Here's how the configuration is built, step by step:
"""

# Step 1: Initial Load from conf/config.yaml
initial_config = """
defaults:
  - schema/common
  - schema/${service_name}    # Will resolve to schema/logging
  - components/${service_name} # Will resolve to components/logging/default
  - env: development          # Will load env/development.yaml
  - _self_

hydra:
  run:
    dir: ${oc.env:LOG_DIR,${original_cwd}/logs}/${now:%Y-%m-%d}/${now:%H-%M-%S}
  job:
    chdir: true

project:
  name: "agentic"
  version: "1.0.0"
"""

# Step 2: Load Schema (schema/common.yaml)
common_schema = """
service_base:
  name: ???                # Required field
  version: ${project.version}
  host: localhost
  port: ???               # Required field
  metrics_enabled: true
  health_check:
    enabled: true
    interval: 30
    timeout: 5
"""

# Step 3: Load Service Schema (schema/logging.yaml)
logging_schema = """
logging_service:
  _target_: microservices.logging_service.domain.config.LoggingServiceConfig
  _inherits_: service_base
  name: logging_service
  port: 50051
  
  log_handling:
    root_dir: ${hydra.run.dir}
    rotation:
      max_bytes: 10485760
      backup_count: 5
    retention_days: 30
    default_level: INFO
    format: DETAILED
"""

# Step 4: Load Component Default (components/logging/default.yaml)
component_default = """
defaults:
  - /schema/logging

service:
  name: "logging_service"
  host: localhost
  port: 50051
  metrics_enabled: true
  
  log_handling:
    format: DETAILED
    handlers:
      file:
        enabled: true
        path: ${logging_service.log_handling.root_dir}/service.log
        rotation:
          max_bytes: ${logging_service.log_handling.rotation.max_bytes}
          backup_count: ${logging_service.log_handling.rotation.backup_count}
      console:
        enabled: true
"""

# Step 5: Load Environment Config (env/development.yaml)
environment_config = """
defaults:
  - components/logging/development

logging_service:
  host: localhost
  port: 50051
  log_handling:
    root_dir: ${oc.env:LOG_DIR,${hydra.runtime.cwd}/logs/dev}
    default_level: DEBUG
    format: DETAILED
    console_output: true
    retention_days: 7
  
  registry:
    url: localhost:50050
    heartbeat_interval: 15
    retry_count: 5

  monitoring:
    metrics_enabled: true
    metrics_port: 9090
    debug_endpoints: true
"""

# Final Resolution Process
resolution_order = """
1. Base Configuration (config.yaml):
   - Sets up basic structure
   - Defines project info
   - Sets Hydra behavior

2. Schema Loading:
   a. Common Schema (schema/common.yaml):
      - Defines base service structure
      - Sets required fields
      - Establishes defaults
   
   b. Logging Schema (schema/logging.yaml):
      - Inherits from service_base
      - Defines logging-specific structure
      - Sets logging defaults

3. Component Configuration:
   a. Default (components/logging/default.yaml):
      - Provides concrete values for required fields
      - Sets up basic logging configuration
      - Configures handlers
   
   b. Environment-specific (components/logging/development.yaml):
      - Overrides relevant default values
      - Sets environment-specific parameters

4. Environment Configuration (env/development.yaml):
   - Final overrides for development environment
   - Sets environment-specific paths
   - Configures debugging and monitoring

5. Variable Interpolation:
   - Resolves ${} references
   - Applies environment variable overrides
   - Handles defaults for missing values

6. Validation:
   - Checks all required fields are present
   - Validates types and values
   - Ensures configuration completeness
"""

# Example of final resolved configuration
final_config = """
logging_service:
  name: "logging_service"
  version: "1.0.0"
  host: localhost
  port: 50051
  metrics_enabled: true
  
  log_handling:
    root_dir: "/home/bbrel/agentic/logs/dev"
    default_level: DEBUG
    format: DETAILED
    console_output: true
    retention_days: 7
    handlers:
      file:
        enabled: true
        path: "/home/bbrel/agentic/logs/dev/service.log"
        rotation:
          max_bytes: 10485760
          backup_count: 5
      console:
        enabled: true
  
  registry:
    url: localhost:50050
    heartbeat_interval: 15
    retry_count: 5
  
  monitoring:
    metrics_enabled: true
    metrics_port: 9090
    debug_endpoints: true
  
  health_check:
    enabled: true
    interval: 30
    timeout: 5
"""

# You can see this resolution process by running:
"""
python -m microservices.logging_service.main --info

# Or see the final config with:
python -m microservices.logging_service.main --cfg hydra
"""
