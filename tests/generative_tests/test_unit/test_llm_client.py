# tests/test_generative_agent/test_unit/test_llm_client.py

import pytest
from unittest.mock import patch

from services.generative_agent.infrastructure.llm_client import LLMClient
from services.generative_agent.domain.exceptions import LLMError

class TestLLMClient:
    """Test LLM client functionality"""

    @pytest.mark.asyncio
    async def test_successful_generation(self, test_config, mock_llm_client):
        """Test successful response generation"""
        mock_llm_client.generate.return_value = "Analysis response"
        
        client = LLMClient(test_config)
        response = await client.generate("Test prompt")
        
        assert response is not None
        assert isinstance(response, str)
        assert "Analysis" in response

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, test_config, mock_llm_client):
        """Test handling of rate limits"""
        mock_llm_client.generate.side_effect = [
            LLMError("Rate limit exceeded"),
            "Retry successful"
        ]
        
        client = LLMClient(test_config)
        response = await client.generate("Test prompt")
        
        assert response == "Retry successful"
        assert mock_llm_client.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_handling(self, test_config, mock_llm_client):
        """Test handling of timeouts"""
        mock_llm_client.generate.side_effect = TimeoutError("Request timed out")
        
        client = LLMClient(test_config)
        with pytest.raises(LLMError):
            await client.generate("Test prompt")
