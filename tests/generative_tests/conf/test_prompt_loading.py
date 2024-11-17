# tests/test_generative_agent/test_prompts/test_prompt_loading.py

import pytest
from pathlib import Path

from services.generative_agent.infrastructure.prompt_manager import PromptManager

class TestPromptLoading:
    """Test prompt loading and management"""

    def test_prompt_directory_exists(self, test_config):
        """Test that prompt directory exists"""
        prompt_path = test_config.service.prompts.base_path
        assert prompt_path.exists()
        assert prompt_path.is_dir()

    def test_load_single_prompt(self, test_config):
        """Test loading a single prompt file"""
        prompt_manager = PromptManager(test_config)
        prompt = prompt_manager.load_prompt("financial_analysis/base.prompt")
        
        assert prompt is not None
        assert "financial analyst" in prompt.lower()
        assert "core responsibilities" in prompt.lower()

    def test_load_prompt_chain(self, test_config):
        """Test loading a chain of prompts"""
        prompt_manager = PromptManager(test_config)
        chain = prompt_manager.load_prompt_chain([
            "financial_analysis/base.prompt",
            "financial_analysis/risk_evaluation.prompt"
        ])
        
        assert len(chain) == 2
        assert all(p is not None for p in chain)
        assert "financial analyst" in chain[0].lower()
        assert "risk" in chain[1].lower()

    def test_invalid_prompt_path(self, test_config):
        """Test handling of invalid prompt paths"""
        prompt_manager = PromptManager(test_config)
        with pytest.raises(FileNotFoundError):
            prompt_manager.load_prompt("nonexistent.prompt")
