# services/generative_agent/domain/interfaces.py
from typing import Protocol, AsyncIterator, List, Dict, Optional, Any
from datetime import datetime
from .models import (
    AnalysisRequest, AnalysisResponse, PromptChain, 
    PromptMetadata, AnalysisType
)

class LLMProvider(Protocol):
    """Interface for LLM interactions"""
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: Optional[float] = None
    ) -> str:
        """Generate response from LLM"""
        pass
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass
    
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        pass

class PromptManager(Protocol):
    """Interface for prompt management"""
    
    async def load_prompt(self, name: str) -> str:
        """Load a prompt by name"""
        pass
    
    async def get_prompt_metadata(self, name: str) -> PromptMetadata:
        """Get metadata for a prompt"""
        pass
    
    async def create_chain(
        self,
        analysis_type: AnalysisType,
        context: Dict[str, Any]
    ) -> PromptChain:
        """Create a prompt chain for analysis"""
        pass

class CacheManager(Protocol):
    """Interface for cache management"""
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        pass
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set item in cache"""
        pass
    
    async def invalidate(self, key: str) -> bool:
        """Invalidate cache entry"""
        pass

class AnalysisProvider(Protocol):
    """Interface for analysis providers"""
    
    async def analyze(
        self,
        request: AnalysisRequest,
        context: Dict[str, Any]
    ) -> AnalysisResponse:
        """Perform analysis"""
        pass
    
    def can_handle(
        self,
        analysis_type: AnalysisType,
        context: Dict[str, Any]
    ) -> bool:
        """Check if provider can handle analysis type"""
        pass

class MetricsCollector(Protocol):
    """Interface for metrics collection"""
    
    async def record_request(
        self,
        request_id: str,
        analysis_type: AnalysisType,
        timestamp: datetime
    ) -> None:
        """Record analysis request"""
        pass
    
    async def record_response(
        self,
        request_id: str,
        processing_time: float,
        token_count: int,
        success: bool
    ) -> None:
        """Record analysis response"""
        pass
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        pass

class HealthCheck(Protocol):
    """Interface for health checking"""
    
    async def check_health(self) -> Dict[str, Any]:
        """Check service health"""
        pass
    
    async def get_status(self) -> str:
        """Get service status"""
        pass
