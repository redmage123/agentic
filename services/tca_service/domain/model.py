# services/tca_service/domain/models.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

class RequestType(Enum):
    UNKNOWN = "UNKNOWN"
    TIME_SERIES = "TIME_SERIES"
    CLASSIFICATION = "CLASSIFICATION"
    GENERATION = "GENERATION"

class AgentStatus(Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DEGRADED = "DEGRADED"

class RoutingStrategy(Enum):
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_LOADED = "least_loaded"
    RESPONSE_TIME = "response_time"
    CAPABILITY_BASED = "capability_based"

@dataclass
class AgentInfo:
    """Information about a registered AI agent"""
    agent_id: str
    agent_type: str
    host: str
    port: int
    supported_types: List[RequestType]
    capabilities: Dict[str, str]
    status: AgentStatus = AgentStatus.ACTIVE
    load: float = 0.0
    last_health_check: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictionRequest:
    """Incoming prediction request"""
    request_id: str
    input_data: str
    request_type: RequestType
    metadata: Dict[str, str]
    preferred_agents: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AgentPrediction:
    """Prediction from a single agent"""
    agent_id: str
    prediction: str
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AggregatedPrediction:
    """Final aggregated prediction"""
    request_id: str
    predictions: List[AgentPrediction]
    aggregated_result: str
    confidence_score: float
    metadata: Dict[str, str]
    processing_time: float
