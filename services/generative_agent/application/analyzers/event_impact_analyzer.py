# services/generative_agent/application/analyzers/event_impact_analyzer.py

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import asyncio

from ...domain.models import AnalysisResponse, AnalysisType
from ...domain.interfaces import LLMProvider
from ...domain.exceptions import AnalysisError

class EventType(Enum):
    ECONOMIC = "economic"          # Fed meetings, GDP releases, etc.
    GEOPOLITICAL = "geopolitical"  # Political events, conflicts, etc.
    CORPORATE = "corporate"        # Earnings, M&A, management changes
    MARKET = "market"             # Market structure events, circuit breakers
    REGULATORY = "regulatory"     # Policy changes, regulations
    SECTOR = "sector"            # Sector-specific events
    
class EventSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"

class ImpactTimeframe(Enum):
    IMMEDIATE = "immediate"      # Intraday impact
    SHORT_TERM = "short_term"    # 1-5 days
    MEDIUM_TERM = "medium_term"  # 1-4 weeks
    LONG_TERM = "long_term"      # 1+ months

@dataclass
class EventImpact:
    """Represents the analyzed impact of an event"""
    event_id: str
    event_type: EventType
    description: str
    severity: EventSeverity
    timeframe: ImpactTimeframe
    affected_symbols: List[str]
    price_impact: float  # Estimated price impact percentage
    volume_impact: float  # Estimated volume impact percentage
    volatility_impact: float  # Estimated volatility impact percentage
    confidence_score: float
    similar_events: List[str]  # IDs of similar historical events
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class MarketReaction:
    """Represents market reaction to an event"""
    price_change: float
    volume_change: float
    volatility_change: float
    sector_correlation: float
    market_breadth: float
    sentiment_shift: float
    liquidity_impact: float

class EventImpactAnalyzer:
    """
    Analyzes market impact of various events using historical data, 
    real-time market reactions, and LLM-based analysis.
    """
    
    def __init__(
        self,
        llm_client: LLMProvider,
        historical_lookback: int = 365,  # days
        min_confidence_threshold: float = 0.7,
        volatility_window: int = 20,     # days
        correlation_threshold: float = 0.6
    ):
        self.llm_client = llm_client
        self.historical_lookback = historical_lookback
        self.min_confidence_threshold = min_confidence_threshold
        self.volatility_window = volatility_window
        self.correlation_threshold = correlation_threshold

    async def analyze(
        self,
        event_data: Dict[str, Any],
        market_data: Dict[str, Any],
        news_data: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> AnalysisResponse:
        """
        Analyze event impact using multiple data sources
        """
        try:
            # Analyze event characteristics
            event_analysis = await self._analyze_event(
                event_data, news_data
            )
            
            # Analyze market reaction
            market_reaction = await self._analyze_market_reaction(
                event_analysis, market_data
            )
            
            # Find similar historical events
            similar_events = await self._find_similar_events(
                event_analysis, market_data
            )
            
            # Predict impact
            impact_prediction = await self._predict_impact(
                event_analysis,
                market_reaction,
                similar_events
            )
            
            # Generate analysis and recommendations
            detailed_analysis = self._generate_impact_analysis(
                event_analysis,
                market_reaction,
                similar_events,
                impact_prediction
            )
            
            recommendations = self._generate_impact_recommendations(
                impact_prediction,
                event_analysis
            )

            return AnalysisResponse(
                request_id=metadata.get('request_id', ''),
                analysis_type=AnalysisType.EVENT_IMPACT,
                summary=self._generate_impact_summary(
                    event_analysis,
                    impact_prediction
                ),
                detailed_analysis=detailed_analysis,
                recommendations=recommendations,
                confidence_score=impact_prediction.confidence_score,
                metadata={
                    **metadata,
                    'event_type': event_analysis.event_type.value,
                    'severity': event_analysis.severity.value,
                    'timeframe': event_analysis.timeframe.value,
                    'similar_events_count': len(similar_events)
                },
                processing_time=metadata.get('processing_time', 0),
                timestamp=datetime.now()
            )

        except Exception as e:
            raise AnalysisError(f"Event impact analysis failed: {str(e)}")

    async def _analyze_event(
        self,
        event_data: Dict[str, Any],
        news_data: List[Dict[str, Any]]
    ) -> EventImpact:
        """Analyze event characteristics and initial impact"""
        try:
            # Classify event type
            event_type = await self._classify_event_type(
                event_data, news_data
            )
            
            # Determine event severity
            severity = await self._determine_severity(
                event_data, event_type
            )
            
            # Identify affected symbols
            affected_symbols = await self._identify_affected_symbols(
                event_data, news_data
            )
            
            # Analyze expected timeframe
            timeframe = await self._analyze_timeframe(
                event_type, severity
            )
            
            # Generate event impact assessment
            return EventImpact(
                event_id=event_data.get('id', ''),
                event_type=event_type,
                description=event_data.get('description', ''),
                severity=severity,
                timeframe=timeframe,
                affected_symbols=affected_symbols,
                price_impact=0.0,  # Will be updated later
                volume_impact=0.0,  # Will be updated later
                volatility_impact=0.0,  # Will be updated later
                confidence_score=0.0,  # Will be updated later
                similar_events=[],  # Will be updated later
                timestamp=datetime.now(),
                metadata=event_data.get('metadata', {})
            )

        except Exception as e:
            raise AnalysisError(f"Event analysis failed: {str(e)}")

    async def _analyze_market_reaction(
        self,
        event: EventImpact,
        market_data: Dict[str, Any]
    ) -> MarketReaction:
        """Analyze immediate market reaction to event"""
        try:
            # Get relevant market data around event time
            pre_event_data = self._get_pre_event_data(
                market_data, event.timestamp
            )
            post_event_data = self._get_post_event_data(
                market_data, event.timestamp
            )
            
            # Calculate market reactions
            price_change = self._calculate_price_impact(
                pre_event_data, post_event_data
            )
            volume_change = self._calculate_volume_impact(
                pre_event_data, post_event_data
            )
            volatility_change = self._calculate_volatility_impact(
                pre_event_data, post_event_data
            )
            
            # Calculate broader market impacts
            sector_correlation = self._calculate_sector_correlation(
                event.affected_symbols, market_data
            )
            market_breadth = self._calculate_market_breadth(
                market_data, event.timestamp
            )
            sentiment_shift = self._calculate_sentiment_shift(
                pre_event_data, post_event_data
            )
            liquidity_impact = self._calculate_liquidity_impact(
                pre_event_data, post_event_data
            )
            
            return MarketReaction(
                price_change=price_change,
                volume_change=volume_change,
                volatility_change=volatility_change,
                sector_correlation=sector_correlation,
                market_breadth=market_breadth,
                sentiment_shift=sentiment_shift,
                liquidity_impact=liquidity_impact
            )

        except Exception as e:
            raise AnalysisError(f"Market reaction analysis failed: {str(e)}")

    async def _predict_impact(
        self,
        event: EventImpact,
        reaction: MarketReaction,
        similar_events: List[EventImpact]
    ) -> EventImpact:
        """Predict event impact using current reaction and historical events"""
        try:
            # Update impact predictions based on market reaction
            event.price_impact = self._predict_price_impact(
                event, reaction, similar_events
            )
            event.volume_impact = self._predict_volume_impact(
                event, reaction, similar_events
            )
            event.volatility_impact = self._predict_volatility_impact(
                event, reaction, similar_events
            )
            
            # Calculate confidence score
            event.confidence_score = self._calculate_impact_confidence(
                event, reaction, similar_events
            )
            
            return event

        except Exception as e:
            raise AnalysisError(f"Impact prediction failed: {str(e)}")

    async def _classify_event_type(
        self,
        event_data: Dict[str, Any],
        news_data: List[Dict[str, Any]]
    ) -> EventType:
        """Classify event type using event data and news context"""
        # Create prompt for LLM
        prompt = self._create_event_classification_prompt(
            event_data, news_data
        )
        
        # Get LLM response
        response = await self.llm_client.generate(prompt)
        
        # Parse LLM response
        return self._parse_event_type(response)

    async def _determine_severity(
        self,
        event_data: Dict[str, Any],
        event_type: EventType
    ) -> EventSeverity:
        """Determine event severity based on characteristics"""
        # Create prompt for LLM
        prompt = self._create_severity_analysis_prompt(
            event_data, event_type
        )
        
        # Get LLM response
        response = await self.llm_client.generate(prompt)
        
        # Parse LLM response
        return self._parse_severity(response)

    def _create_event_classification_prompt(
        self,
        event_data: Dict[str, Any],
        news_data: List[Dict[str, Any]]
    ) -> str:
        """Create prompt for event classification"""
        return f"""Analyze the following event and classify its type:

Event Description: {event_data.get('description', '')}
Event Time: {event_data.get('timestamp', '')}
Event Source: {event_data.get('source', '')}

Related News Headlines:
{self._format_news_headlines(news_data)}

Based on the above information, classify this event into one of the following categories:
- Economic (e.g., Fed meetings, GDP releases)
- Geopolitical (e.g., political events, conflicts)
- Corporate (e.g., earnings, M&A, management changes)
- Market (e.g., market structure events, circuit breakers)
- Regulatory (e.g., policy changes, regulations)
- Sector (e.g., sector-specific events)

Provide the classification and a brief explanation of your reasoning.
"""

    def _create_severity_analysis_prompt(
        self,
        event_data: Dict[str, Any],
        event_type: EventType
    ) -> str:
        """Create prompt for severity analysis"""
        return f"""Analyze the following {event_type.value} event and determine its severity:

Event Description: {event_data.get('description', '')}
Event Type: {event_type.value}
Scale: {event_data.get('scale', '')}
Scope: {event_data.get('scope', '')}

Calculate the event severity considering:
1. Potential market impact
2. Scope of affected assets/sectors
3. Historical precedent
4. Market conditions

Classify severity as:
- CRITICAL: Major market-wide impact
- HIGH: Significant sector/asset impact
- MEDIUM: Moderate localized impact
- LOW: Minor impact
- NEGLIGIBLE: Minimal to no impact

Provide severity classification and supporting reasoning.
"""

    def _format_news_headlines(
        self,
        news_data: List[Dict[str, Any]]
    ) -> str:
        """Format news headlines for prompt"""
        headlines = []
        for news in news_data[:5]:  # Use top 5 most relevant headlines
            headlines.append(
                f"- {news.get('timestamp', '')}: {news.get('title', '')}"
            )
        return "\n".join(headlines)

    def _calculate_impact_confidence(
        self,
        event: EventImpact,
        reaction: MarketReaction,
        similar_events: List[EventImpact]
    ) -> float:
        """Calculate confidence score for impact prediction"""
        confidence_factors = []
        
        # Historical similarity confidence
        if similar_events:
            similarity_scores = [e.confidence_score for e in similar_events]
            confidence_factors.append(sum(similarity_scores) / len(similarity_scores))
        
        # Market reaction confidence
        reaction_confidence = min(
            abs(reaction.price_change),
            abs(reaction.volume_change),
            abs(reaction.volatility_change)
        ) / 100
        confidence_factors.append(reaction_confidence)
        
        # Event type confidence
        type_confidence = {
            EventType.ECONOMIC: 0.9,
            EventType.CORPORATE: 0.8,
            EventType.MARKET: 0.8,
            EventType.REGULATORY: 0.7,
            EventType.SECTOR: 0.7,
            EventType.GEOPOLITICAL: 0.6
        }.get(event.event_type, 0.5)
        confidence_factors.append(type_confidence)
        
        # Overall confidence
        return sum(confidence_factors) / len(confidence_factors)

    def _generate_impact_analysis(
        self,
        event: EventImpact,
        reaction: MarketReaction,
        similar_events: List[EventImpact],
        prediction: EventImpact
    ) -> str:
        """Generate detailed impact analysis"""
        analysis_parts = []
        
        # Event Overview
        analysis_parts.append("## Event Analysis\n")
        analysis_parts.append(f"Event Type: {event.event_type.value}")
        analysis_parts.append(f"Severity: {event.severity.value}")
        analysis_parts.append(f"Expected Timeframe: {event.timeframe.value}\n")
        
        # Market Reaction
        analysis_parts.append("## Initial Market Reaction\n")
        analysis_parts.append(
            f"Price Impact: {reaction.price_change:.2f}%"
        )
        analysis_parts.append(
            f"Volume Change: {reaction.volume_change:.2f}%"
        )
        analysis_parts.append(
            f"Volatility Change: {reaction.volatility_change:.2f}%"
        )
        analysis_parts.append(
            f"Market Breadth: {reaction.market_breadth:.2f}"
        )
        analysis_parts.append(
            f"Sector Correlation: {reaction.sector_correlation:.2f}\n"
        )
        
        # Historical Comparison
        if similar_events:
            analysis_parts.append("## Historical Comparisons\n")
            for similar in similar_events[:3]:  #
