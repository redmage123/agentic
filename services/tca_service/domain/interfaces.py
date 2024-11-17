# services/tca_service/domain/interfaces.py
from abc import ABC, abstractmethod
from typing import Protocol, List, Dict, Optional, AsyncIterator
from .models import (
    AgentInfo, PredictionRequest, AgentPrediction, 
    AggregatedPrediction, RequestType, AgentStatus
)

class RoutingStrategy(Protocol):
    """Protocol for implementing routing strategies"""
    
    async def select_agents(
        self,
        request: PredictionRequest,
        available_agents: List[AgentInfo],
        count: int = 1
    ) -> List[AgentInfo]:
        """Select appropriate agents for a request"""
        pass

class AgentManager(Protocol):
    """Protocol for managing AI agents"""
    
    async def register_agent(self, agent_info: AgentInfo) -> bool:
        """Register a new agent"""
        pass
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """Deregister an agent"""
        pass
    
    async def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent information"""
        pass
    
    async def list_agents(
        self,
        status: Optional[AgentStatus] = None,
        request_type: Optional[RequestType] = None
    ) -> List[AgentInfo]:
        """List available agents"""
        pass
    
    async def update_agent_status(
        self,
        agent_id: str,
        status: AgentStatus,
        metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update agent status and metrics"""
        pass

class ResponseAggregator(Protocol):
    """Protocol for aggregating agent responses"""
    
    async def aggregate(
        self,
        request: PredictionRequest,
        predictions: List[AgentPrediction]
    ) -> AggregatedPrediction:
        """Aggregate multiple agent predictions"""
        pass

class AgentClient(Protocol):
    """Protocol for communicating with AI agents"""
    
    async def predict(
        self,
        agent_info: AgentInfo,
        request: PredictionRequest,
        timeout: Optional[float] = None
    ) -> AgentPrediction:
        """Send prediction request to an agent"""
        pass
    
    async def check_health(self, agent_info: AgentInfo) -> bool:
        """Check agent health status"""
        pass

class CircuitBreaker(Protocol):
    """Protocol for circuit breaker implementation"""
    
    async def mark_success(self, agent_id: str) -> None:
        """Mark a successful request"""
        pass
    
    async def mark_failure(self, agent_id: str) -> None:
        """Mark a failed request"""
        pass
    
    async def can_execute(self, agent_id: str) -> bool:
        """Check if requests can be sent to agent"""
        pass

    async def get_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get circuit breaker metrics for an agent"""
        pass

class MetricsCollector(Protocol):
    """Protocol for collecting system metrics"""
    
    async def record_request(
        self,
        request_id: str,
        request_type: RequestType,
        agents: List[str]
    ) -> None:
        """Record an incoming request"""
        pass
    
    async def record_response(
        self,
        request_id: str,
        processing_time: float,
        success: bool
    ) -> None:
        """Record a completed request"""
        pass
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        pass
