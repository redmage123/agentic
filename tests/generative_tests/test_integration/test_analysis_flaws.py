# tests/test_generative_agent/test_integration/test_analysis_flows.py

import pytest
from services.generative_agent.application.service import GenerativeService
from services.generative_agent.domain.models import AnalysisRequest

class TestAnalysisFlows:
    """Test end-to-end analysis flows"""

    @pytest.mark.asyncio
    async def test_risk_analysis_flow(self, test_config, sample_request):
        """Test complete risk analysis flow"""
        service = GenerativeService(test_config)
        response = await service.analyze(sample_request)
        
        assert response is not None
        assert response.confidence >= 0.0
        assert len(response.reasoning) > 0
        assert len(response.recommendations) > 0

    @pytest.mark.asyncio
    async def test_pattern_recognition_flow(self, test_config):
        """Test pattern recognition analysis flow"""
        request = AnalysisRequest(
            query="Identify trading patterns in AAPL over the last month",
            analysis_type="pattern_recognition",
            context={"ticker": "AAPL", "timeframe": "1M"}
        )
        
        service = GenerativeService(test_config)
        response = await service.analyze(request)
        
        assert response is not None
        assert "pattern" in response.analysis.lower()
        assert response.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_multi_prompt_analysis(self, test_config):
        """Test analysis using multiple prompts"""
        request = AnalysisRequest(
            query="Provide comprehensive analysis of AAPL",
            analysis_type="full_analysis",
            context={"ticker": "AAPL"}
        )
        
        service = GenerativeService(test_config)
        response = await service.analyze(request)
        
        assert response is not None
        assert response.analysis is not None
        assert "risk" in response.analysis.lower()
        assert "pattern" in response.analysis.lower()
