# services/generative_agent/application/service.py

from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from ..domain.models import (
    AnalysisRequest, AnalysisResponse, PredictionRequest, 
    AgentPrediction, AggregatedPrediction, RequestType
)
from ..domain.exceptions import AnalysisError
from .analyzers.financial_analyzer import FinancialAnalyzer
from .analyzers.pattern_analyzer import PatternAnalyzer
from .analyzers.regime_analyzer import MarketRegimeAnalyzer
from .analyzers.risk_analyzer import RiskAnalyzer
from .analyzers.volatility_analyzer import VolatilityAnalyzer
from .analyzers.liquidity_analyzer import LiquidityAnalyzer

class GenerativeService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize analyzers
        self.financial_analyzer = FinancialAnalyzer(config)
        self.pattern_analyzer = PatternAnalyzer(config)
        self.regime_analyzer = MarketRegimeAnalyzer(config)
        self.risk_analyzer = RiskAnalyzer(config)
        self.volatility_analyzer = VolatilityAnalyzer(config)
        self.liquidity_analyzer = LiquidityAnalyzer(config)

    async def analyze_market(
        self,
        request: AnalysisRequest
    ) -> AnalysisResponse:
        """Process market analysis request"""
        try:
            start_time = datetime.now()
            
            # Extract market data
            market_data = self._prepare_market_data(request)
            
            # Run all analyses concurrently
            analysis_results = await self._run_analyses(market_data)
            
            # Synthesize results
            synthesized_analysis = await self._synthesize_analyses(
                analysis_results,
                market_data
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResponse(
                request_id=request.request_id,
                analysis_type=request.analysis_type,
                summary=synthesized_analysis.summary,
                detailed_analysis=synthesized_analysis.detailed_analysis,
                recommendations=synthesized_analysis.recommendations,
                confidence_score=self._calculate_confidence(analysis_results),
                metadata={
                    'processing_time': processing_time,
                    'analysis_results': analysis_results,
                    'market_context': market_data.get('context'),
                },
                processing_time=processing_time
            )

        except Exception as e:
            raise AnalysisError(f"Market analysis failed: {str(e)}")

    async def _run_analyses(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run all analyses concurrently"""
        results = await asyncio.gather(
            self.financial_analyzer.analyze(market_data),
            self.pattern_analyzer.analyze(market_data),
            self.regime_analyzer.analyze(market_data),
            self.risk_analyzer.analyze(market_data),
            self.volatility_analyzer.analyze(market_data),
            self.liquidity_analyzer.analyze(market_data),
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

    async def _synthesize_analyses(
        self,
        analysis_results: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize analysis results"""
        # Collect all recommendations
        recommendations = []
        confidence_scores = []
        
        for analysis_type, result in analysis_results.items():
            if result:
                recommendations.extend(result.recommendations)
                confidence_scores.append(result.confidence_score)
        
        # Prioritize and deduplicate recommendations
        unique_recommendations = list(set(recommendations))
        
        # Generate summary
        summary = self._generate_summary(analysis_results, market_data)
        
        # Generate detailed analysis
        detailed_analysis = self._generate_detailed_analysis(
            analysis_results,
            market_data
        )
        
        return {
            'summary': summary,
            'detailed_analysis': detailed_analysis,
            'recommendations': unique_recommendations
        }

    def _prepare_market_data(
        self,
        request: AnalysisRequest
    ) -> Dict[str, Any]:
        """Prepare market data for analysis"""
        return {
            'prices': request.market_data.get('prices', []),
            'volumes': request.market_data.get('volumes', []),
            'timestamps': request.market_data.get('timestamps', []),
            'market_caps': request.market_data.get('market_caps', []),
            'indicators': request.market_data.get('indicators', {}),
            'context': request.market_data.get('context', {})
        }

    def _calculate_confidence(
        self,
        analysis_results: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence score"""
        scores = []
        weights = {
            'financial': 0.25,
            'pattern': 0.15,
            'regime': 0.2,
            'risk': 0.2,
            'volatility': 0.1,
            'liquidity': 0.1
        }
        
        for analysis_type, result in analysis_results.items():
            if result:
                scores.append(
                    result.confidence_score * weights[analysis_type]
                )
                
        return sum(scores) / sum(weights.values()) if scores else 0.0

    def _generate_summary(
        self,
        analysis_results: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> str:
        """Generate analysis summary"""
        summaries = []
        for analysis_type, result in analysis_results.items():
            if result:
                summaries.append(result.summary)
        return "\n\n".join(summaries)

    def _generate_detailed_analysis(
        self,
        analysis_results: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> str:
        """Generate detailed analysis"""
        analyses = []
        for analysis_type, result in analysis_results.items():
            if result:
                analyses.append(f"## {analysis_type.title()} Analysis")
                analyses.append(result.detailed_analysis)
        return "\n\n".join(analyses)
