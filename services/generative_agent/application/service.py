# services/generative_agent/application/service.py

from typing import List, Dict, Any, Optional
from datetime import datetime

from ..domain.models import (
    AgentInfo, PredictionRequest, AgentPrediction,
    AggregatedPrediction, RequestType, AgentStatus
)
from ..domain.interfaces import (
    RoutingStrategy, AgentManager, ResponseAggregator,
    MetricsCollector, CircuitBreaker
)
from .analyzers.financial_analyzer import FinancialAnalyzer
from .analyzers.pattern_analyzer import PatternAnalyzer
from .analyzers.regime_analyzer import MarketRegimeAnalyzer
from .analyzers.risk_analyzer import RiskAnalyzer
from .analyzers.volatility_analyzer import VolatilityAnalyzer
from .analyzers.liquidity_analyzer import LiquidityAnalyzer

class GenerativeTCAService:
    """
    Integrated Generative Traffic Control Agent that combines:
    - Agent coordination (from TCA)
    - Market analysis (from Generative Agent)
    - Response aggregation and enrichment
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        routing_strategy: RoutingStrategy,
        agent_manager: AgentManager,
        response_aggregator: ResponseAggregator,
        metrics_collector: MetricsCollector,
        circuit_breaker: CircuitBreaker
    ):
        self.config = config
        self.routing_strategy = routing_strategy
        self.agent_manager = agent_manager
        self.response_aggregator = response_aggregator
        self.metrics_collector = metrics_collector
        self.circuit_breaker = circuit_breaker
        
        # Initialize analyzers
        self.financial_analyzer = FinancialAnalyzer(
            hamiltonian_client=None,  # Will be set after agent registration
            fourier_client=None,
            perturbation_client=None
        )
        self.pattern_analyzer = PatternAnalyzer()
        self.regime_analyzer = MarketRegimeAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()

    async def process_request(
        self,
        request: PredictionRequest
    ) -> AggregatedPrediction:
        """Process prediction request with integrated analysis"""
        try:
            # Record request metrics
            await self.metrics_collector.record_request(
                request.request_id,
                request.request_type,
                request.preferred_agents
            )
            
            start_time = datetime.now()
            
            # Select appropriate agents
            available_agents = await self.agent_manager.list_agents(
                status=AgentStatus.ACTIVE,
                request_type=request.request_type
            )
            
            selected_agents = await self.routing_strategy.select_agents(
                request, available_agents
            )
            
            # Get predictions from physics-based agents
            predictions = await self._get_agent_predictions(
                request, selected_agents
            )
            
            # Analyze predictions using various analyzers
            analysis_results = await self._analyze_predictions(
                request, predictions
            )
            
            # Aggregate and enrich results
            enriched_predictions = self._enrich_predictions(
                predictions, analysis_results
            )
            
            aggregated_result = await self.response_aggregator.aggregate(
                request, enriched_predictions
            )
            
            # Record response metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            await self.metrics_collector.record_response(
                request.request_id,
                processing_time,
                True
            )
            
            return aggregated_result

        except Exception as e:
            # Record failure metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            await self.metrics_collector.record_response(
                request.request_id,
                processing_time,
                False
            )
            raise

    async def _get_agent_predictions(
        self,
        request: PredictionRequest,
        agents: List[AgentInfo]
    ) -> List[AgentPrediction]:
        """Get predictions from physics-based agents"""
        predictions = []
        
        for agent in agents:
            if await self.circuit_breaker.can_execute(agent.agent_id):
                try:
                    prediction = await self._get_prediction(request, agent)
                    predictions.append(prediction)
                    await self.circuit_breaker.mark_success(agent.agent_id)
                except Exception as e:
                    await self.circuit_breaker.mark_failure(agent.agent_id)
                    
        return predictions

    async def _analyze_predictions(
        self,
        request: PredictionRequest,
        predictions: List[AgentPrediction]
    ) -> Dict[str, Any]:
        """Analyze predictions using various analyzers"""
        market_data = self._extract_market_data(request, predictions)
        
        # Run analyses concurrently
        financial_analysis = self.financial_analyzer.analyze(market_data)
        pattern_analysis = self.pattern_analyzer.analyze(market_data)
        regime_analysis = self.regime_analyzer.analyze(market_data)
        risk_analysis = self.risk_analyzer.analyze(market_data)
        volatility_analysis = self.volatility_analyzer.analyze(market_data)
        liquidity_analysis = self.liquidity_analyzer.analyze(market_data)
        
        results = await asyncio.gather(
            financial_analysis,
            pattern_analysis,
            regime_analysis,
            risk_analysis,
            volatility_analysis,
            liquidity_analysis,
            return_exceptions=True
        )
        
        return {
            'financial': results[0] if not isinstance(results[0], Exception) else None,
            'pattern': results[1] if not isinstance(results[1], Exception) else None,
            'regime': results[2] if not isinstance(results[2], Exception) else None,
            'risk': results[3] if not isinstance(results[3], Exception) else None,
            'volatility': results[4] if not isinstance(results[4], Exception) else None,
            'liquidity': results[5] if not isinstance(results[5], Exception) else None
        }

    def _enrich_predictions(
        self,
        predictions: List[AgentPrediction],
        analysis_results: Dict[str, Any]
    ) -> List[AgentPrediction]:
        """Enrich agent predictions with analysis results"""
        enriched_predictions = []
        
        for prediction in predictions:
            # Add analysis metadata
            enriched = AgentPrediction(
                agent_id=prediction.agent_id,
                prediction=prediction.prediction,
                confidence=prediction.confidence,
                processing_time=prediction.processing_time,
                metadata={
                    **prediction.metadata,
                    'analysis_results': {
                        'financial': analysis_results['financial'].summary if analysis_results['financial'] else None,
                        'pattern': analysis_results['pattern'].summary if analysis_results['pattern'] else None,
                        'regime': analysis_results['regime'].summary if analysis_results['regime'] else None,
                        'risk': analysis_results['risk'].summary if analysis_results['risk'] else None,
                        'volatility': analysis_results['volatility'].summary if analysis_results['volatility'] else None,
                        'liquidity': analysis_results['liquidity'].summary if analysis_results['liquidity'] else None
                    }
                }
            )
            enriched_predictions.append(enriched)
            
        return enriched_predictions

    def _extract_market_data(
        self,
        request: PredictionRequest,
        predictions: List[AgentPrediction]
    ) -> Dict[str, Any]:
        """Extract market data from request and predictions"""
        # Implementation depends on your data format
        pass

    async def start(self):
        """Start the service"""
        # Initialize connections to physics-based agents
        agents = await self.agent_manager.list_agents()
        for agent in agents:
            if agent.agent_type == "hamiltonian":
                self.financial_analyzer.hamiltonian_client = agent
            elif agent.agent_type == "fourier":
                self.financial_analyzer.fourier_client = agent
            elif agent.agent_type == "perturbation":
                self.financial_analyzer.perturbation_client = agent

    async def stop(self):
        """Stop the service"""
        # Cleanup
        pass
