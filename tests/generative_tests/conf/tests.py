import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from omegaconf import OmegaConf

from services.generative_agent.infrastructure.llm_client import LLMClient
from services.generative_agent.domain.models import AnalysisRequest, AnalysisResponse

@pytest.fixture
def test_config():
    """Provide test configuration"""
    return OmegaConf.create({
        "service": {
            "name": "generative_agent",
            "llm": {
                "provider": "anthropic",
                "model": "claude-3-opus-20240229",
                "temperature": 0.7,
                "max_tokens": 2000,
                "timeout": 30,
            },
            "prompts": {
                "base_path": Path(__file__).parent.parent.parent / "services" / "generative_agent" / "prompts"
            }
        }
    })

@pytest.fixture
def mock_llm_client():
    """Provide a mock LLM client"""
    client = Mock(spec=LLMClient)
    client.generate = AsyncMock()
    return client

@pytest.fixture
def sample_prompts():
    """Provide sample prompts for testing"""
    return {
        "base": "You are an expert financial analyst...",
        "risk_evaluation": "Focus on comprehensive risk evaluation...",
        "pattern_recognition": "Focus on identifying and analyzing market patterns..."
    }

@pytest.fixture
def sample_request():
    """Provide a sample analysis request"""
    return AnalysisRequest(
        query="What is the risk profile of AAPL given current market conditions?",
        analysis_type="risk_evaluation",
        context={
            "ticker": "AAPL",
            "timeframe": "current"
        }
    )

@pytest.fixture
def sample_response():
    """Provide a sample analysis response"""
    return AnalysisResponse(
        analysis="Detailed risk analysis of AAPL...",
        confidence=0.85,
        reasoning=["Market conditions...", "Technical factors..."],
        recommendations=["Consider hedging...", "Monitor support levels..."]
    )
