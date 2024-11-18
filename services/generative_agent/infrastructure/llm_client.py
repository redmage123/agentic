# services/generative_agent/infrastructure/llm_client.py
import os
import anthropic
import asyncio
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

from ..domain.interfaces import LLMProvider
from ..domain.exceptions import LLMError
from .memo_manager import managed_memoize


class AnthropicClient(LLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-opus-20240229",
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        self.model = model
        self.timeout = timeout
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        self.max_retries = max_retries

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: Optional[float] = None,
    ) -> str:
        try:
            timeout = timeout or self.timeout
            async with asyncio.timeout(timeout):
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text

        except anthropic.RateLimitError as e:
            raise LLMError(f"Rate limit exceeded: {str(e)}")
        except anthropic.APIError as e:
            raise LLMError(f"API error: {str(e)}")
        except asyncio.TimeoutError:
            raise LLMError(f"Request timed out after {timeout} seconds")
        except Exception as e:
            raise LLMError(f"Unexpected error: {str(e)}")

    @managed_memoize(cache_name="token_counting", ttl=300)
    async def count_tokens(self, text: str) -> int:
        try:
            return anthropic.count_tokens(text)
        except Exception as e:
            raise LLMError(f"Token counting error: {str(e)}")

    def is_available(self) -> bool:
        try:
            return bool(self.client.api_key)
        except Exception:
            return False
