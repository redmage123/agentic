BB
What's next


Let's implement the main service class (service.py). This is where our core business logic lives and orchestrates all other components:

python

Copy
# services/generative_agent/application/service.py
from typing import Dict, Any, Optional
from datetime import datetime

from ..domain.models import AnalysisRequest, AnalysisResponse, AnalysisType
from ..domain.exceptions import GenerativeServiceError, LLMError
from ..infrastructure.llm_client import AnthropicClient
from ..infrastructure.prompt_manager import FileSystemPromptManager
from ..infrastructure.health import HealthMonitor
from ..infrastructure.registry_client import RegistryClient

class GenerativeService:
    def __init__(
        self,
        config: Dict[str, Any],
        llm_client: Optional[AnthropicClient] = None,
        prompt_manager: Optional[FileSystemPromptManager] = None,
        registry_client: Optional[RegistryClient] = None
    ):
        self.config = config
        self.llm_client = llm_client or AnthropicClient()
        self.prompt_manager = prompt_manager
        self.registry_client = registry_client
        self.health_monitor = HealthMonitor(self)

    async def start(self):
        """Start the service"""
        await self.health_monitor.start()
        if self.registry_client:
            await self.registry_client.start()

    async def stop(self):
        """Stop the service"""
        await self.health_monitor.stop()
        if self.registry_client:
            await self.registry_client.stop()

    async def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Process analysis request"""
        start_time = datetime.now()
        
        try:
            # Build prompt chain
            prompt_chain = await self.prompt_manager.create_chain(
                request.analysis_type,
                request.context
            )

            # Generate response
            responses = []
            for prompt in prompt_chain.prompts:
                response = await self.llm_client.generate(
                    prompt,
                    temperature=self.config.get('temperature', 0.7)
                )
                responses.append(response)

            # Process and format response
            result = await self._process_responses(
                request,
                responses,
                prompt_chain.metadata
            )

            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.health_monitor.update_metrics(True, processing_time)

            return result

        except LLMError as e:
            self.health_monitor.update_metrics(False, 0)
            raise GenerativeServiceError(f"LLM error: {str(e)}")
        except Exception as e:
            self.health_monitor.update_metrics(False, 0)
            raise GenerativeServiceError(f"Analysis error: {str(e)}")

    async def _process_responses(
        self,
        request: AnalysisRequest,
        responses: list[str],
        metadata: Dict[str, Any]
    ) -> AnalysisResponse:
        """Process and combine LLM responses"""
        # Process responses based on analysis type
        if request.analysis_type == AnalysisType.FINANCIAL:
            return await self._process_financial_analysis(responses, metadata)
        elif request.analysis_type == AnalysisType.PATTERN:
            return await self._process_pattern_analysis(responses, metadata)
        else:
            return await self._process_general_analysis(responses, metadata)

    async def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        return await self.health_monitor.get_health_status()

    # Add specific analysis processing methods based on type
    async def _process_financial_analysis(
        self,
        responses: list[str],
        metadata: Dict[str, Any]
    ) -> AnalysisResponse:
        # Implement financial analysis processing
        pass

    async def _process_pattern_analysis(
        self,
        responses: list[str],
        metadata: Dict[str, Any]
    ) -> AnalysisResponse:
        # Implement pattern analysis processing
        pass

    async def _process_general_analysis(
        self,
        responses: list[str],
        metadata: Dict[str, Any]
    ) -> AnalysisResponse:
        # Implement general analysis processing
        pass
