# services/generative_agent/application/analyzers/liquidity_analyzer.py

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import numpy as np

from ...domain.models import AnalysisResponse, AnalysisType, ConfidenceLevel
from ...domain.exceptions import AnalysisError

class LiquidityLevel(Enum):
    HIGHLY_LIQUID = "highly_liquid"
    LIQUID = "liquid"
    MODERATE = "moderate"
    ILLIQUID = "illiquid"
    HIGHLY_ILLIQUID = "highly_illiquid"

class LiquidityTrend(Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    CRITICAL = "critical"

@dataclass
class LiquidityMetrics:
    """Core liquidity metrics"""
    bid_ask_spread: float
    depth_weighted_spread: float
    turnover_ratio: float
    amihud_illiquidity: float
    volume_profile: Dict[str, float]
    realized_spread: float
    market_impact: float
    resilience: float
    trading_activity: float

@dataclass
class LiquidityRisk:
    """Liquidity risk assessment"""
    level: LiquidityLevel
    trend: LiquidityTrend
    risk_score: float
    confidence: float
    factors: List[str]
    metrics: LiquidityMetrics
    timestamp: datetime

class LiquidityAnalyzer:
    """
    Analyzes market liquidity using multiple metrics:
    - Bid-ask spread analysis
    - Market depth assessment
    - Trading volume patterns
    - Price impact measures
    - Market resilience
    """
    
    def __init__(
        self,
        min_data_points: int = 20,
        spread_threshold: float = 0.05,
        depth_levels: int = 5,
        resilience_window: int = 30,  # minutes
        volume_buckets: int = 48      # 30-minute buckets
    ):
        self.min_data_points = min_data_points
        self.spread_threshold = spread_threshold
        self.depth_levels = depth_levels
        self.resilience_window = resilience_window
        self.volume_buckets = volume_buckets

    async def analyze(
        self,
        market_data: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> AnalysisResponse:
        """Process liquidity analysis using market microstructure data"""
        try:
            # Validate input data
            self._validate_market_data(market_data)
            
            # Calculate core liquidity metrics
            liquidity_metrics = await self._calculate_liquidity_metrics(market_data)
            
            # Analyze liquidity risk
            liquidity_risk = await self._analyze_liquidity_risk(
                liquidity_metrics,
                market_data
            )
            
            # Generate detailed analysis
            detailed_analysis = self._generate_liquidity_analysis(
                liquidity_metrics,
                liquidity_risk
            )
            
            # Generate recommendations
            recommendations = self._generate_liquidity_recommendations(
                liquidity_risk
            )

            return AnalysisResponse(
                request_id=metadata.get('request_id', ''),
                analysis_type=AnalysisType.LIQUIDITY,
                summary=self._generate_liquidity_summary(liquidity_risk),
                detailed_analysis=detailed_analysis,
                recommendations=recommendations,
                confidence_score=liquidity_risk.confidence,
                metadata={
                    **metadata,
                    'liquidity_level': liquidity_risk.level.value,
                    'liquidity_trend': liquidity_risk.trend.value,
                    'risk_score': liquidity_risk.risk_score
                },
                processing_time=metadata.get('processing_time', 0),
                timestamp=datetime.now()
            )

        except Exception as e:
            raise AnalysisError(f"Liquidity analysis failed: {str(e)}")

    async def _calculate_liquidity_metrics(
        self,
        market_data: Dict[str, Any]
    ) -> LiquidityMetrics:
        """Calculate comprehensive liquidity metrics"""
        
        # Calculate bid-ask spread metrics
        bid_ask_spread = self._calculate_bid_ask_spread(
            market_data['bids'],
            market_data['asks']
        )
        
        # Calculate depth-weighted spread
        depth_weighted_spread = self._calculate_depth_weighted_spread(
            market_data['bids'],
            market_data['asks'],
            self.depth_levels
        )
        
        # Calculate turnover ratio
        turnover_ratio = self._calculate_turnover_ratio(
            market_data['volume'],
            market_data['float_shares']
        )
        
        # Calculate Amihud illiquidity ratio
        amihud_illiquidity = self._calculate_amihud_illiquidity(
            market_data['returns'],
            market_data['volume'],
            market_data['price']
        )
        
        # Analyze volume profile
        volume_profile = self._analyze_volume_profile(
            market_data['volume'],
            market_data['timestamps'],
            self.volume_buckets
        )
        
        # Calculate realized spread
        realized_spread = self._calculate_realized_spread(
            market_data['trades'],
            market_data['quotes']
        )
        
        # Estimate market impact
        market_impact = self._estimate_market_impact(
            market_data['trades'],
            market_data['price'],
            market_data['volume']
        )
        
        # Measure market resilience
        resilience = self._measure_market_resilience(
            market_data['price'],
            market_data['volume'],
            self.resilience_window
        )
        
        # Calculate trading activity
        trading_activity = self._calculate_trading_activity(
            market_data['trades'],
            market_data['timestamps']
        )
        
        return LiquidityMetrics(
            bid_ask_spread=bid_ask_spread,
            depth_weighted_spread=depth_weighted_spread,
            turnover_ratio=turnover_ratio,
            amihud_illiquidity=amihud_illiquidity,
            volume_profile=volume_profile,
            realized_spread=realized_spread,
            market_impact=market_impact,
            resilience=resilience,
            trading_activity=trading_activity
        )

    async def _analyze_liquidity_risk(
        self,
        metrics: LiquidityMetrics,
        market_data: Dict[str, Any]
    ) -> LiquidityRisk:
        """Analyze liquidity risk using multiple factors"""
        
        # Determine liquidity level
        level = self._determine_liquidity_level(metrics)
        
        # Analyze liquidity trend
        trend = self._analyze_liquidity_trend(
            metrics,
            market_data['historical_metrics']
        )
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(metrics, level, trend)
        
        # Calculate confidence
        confidence = self._calculate_confidence(metrics, market_data)
        
        # Identify risk factors
        factors = self._identify_risk_factors(metrics, level, trend)
        
        return LiquidityRisk(
            level=level,
            trend=trend,
            risk_score=risk_score,
            confidence=confidence,
            factors=factors,
            metrics=metrics,
            timestamp=datetime.now()
        )

    def _calculate_bid_ask_spread(
        self,
        bids: List[Dict[str, float]],
        asks: List[Dict[str, float]]
    ) -> float:
        """Calculate average bid-ask spread"""
        if not bids or not asks:
            return float('inf')
            
        spreads = []
        for bid, ask in zip(bids, asks):
            if bid['price'] > 0 and ask['price'] > 0:
                spread = (ask['price'] - bid['price']) / ask['price']
                spreads.append(spread)
                
        return np.mean(spreads) if spreads else float('inf')

    def _calculate_depth_weighted_spread(
        self,
        bids: List[Dict[str, float]],
        asks: List[Dict[str, float]],
        levels: int
    ) -> float:
        """Calculate depth-weighted spread"""
        if not bids or not asks:
            return float('inf')
            
        weighted_spreads = []
        total_volume = 0
        
        for i in range(min(levels, len(bids), len(asks))):
            volume = min(bids[i]['size'], asks[i]['size'])
            spread = (asks[i]['price'] - bids[i]['price']) / asks[i]['price']
            weighted_spreads.append(spread * volume)
            total_volume += volume
            
        return sum(weighted_spreads) / total_volume if total_volume > 0 else float('inf')

    def _calculate_turnover_ratio(
        self,
        volume: List[float],
        float_shares: float
    ) -> float:
        """Calculate turnover ratio"""
        if float_shares <= 0:
            return 0.0
        return sum(volume) / float_shares

    def _calculate_amihud_illiquidity(
        self,
        returns: List[float],
        volume: List[float],
        price: List[float]
    ) -> float:
        """Calculate Amihud illiquidity ratio"""
        if not returns or not volume or not price:
            return float('inf')
            
        daily_ratios = []
        for r, v, p in zip(returns, volume, price):
            if v > 0 and p > 0:
                ratio = abs(r) / (v * p)
                daily_ratios.append(ratio)
                
        return np.mean(daily_ratios) if daily_ratios else float('inf')

    def _analyze_volume_profile(
        self,
        volume: List[float],
        timestamps: List[datetime],
        num_buckets: int
    ) -> Dict[str, float]:
        """Analyze intraday volume profile"""
        if not volume or not timestamps:
            return {}
            
        # Create time buckets
        bucket_size = timedelta(minutes=30)
        buckets = {i: 0.0 for i in range(num_buckets)}
        
        # Aggregate volume into buckets
        for vol, ts in zip(volume, timestamps):
            bucket_idx = (ts.hour * 60 + ts.minute) // 30
            buckets[bucket_idx] += vol
            
        # Normalize volume profile
        total_volume = sum(buckets.values())
        if total_volume > 0:
            return {
                str(i): v / total_volume 
                for i, v in buckets.items()
            }
            
        return buckets

    def _calculate_realized_spread(
        self,
        trades: List[Dict[str, Any]],
        quotes: List[Dict[str, Any]]
    ) -> float:
        """Calculate realized spread"""
        if not trades or not quotes:
            return float('inf')
            
        realized_spreads = []
        for trade in trades:
            # Find corresponding quotes
            trade_quotes = self._find_surrounding_quotes(
                trade['timestamp'],
                quotes
            )
            if trade_quotes:
                mid_price = (
                    trade_quotes['bid_price'] + trade_quotes['ask_price']
                ) / 2
                realized_spread = abs(trade['price'] - mid_price) / mid_price
                realized_spreads.append(realized_spread)
                
        return np.mean(realized_spreads) if realized_spreads else float('inf')

    def _estimate_market_impact(
        self,
        trades: List[Dict[str, Any]],
        price: List[float],
        volume: List[float]
    ) -> float:
        """Estimate market impact coefficient"""
        if not trades or not price or not volume:
            return float('inf')
            
        impacts = []
        for i in range(len(trades)-1):
            if volume[i] > 0:
                price_change = abs(price[i+1] - price[i]) / price[i]
                volume_ratio = volume[i] / np.mean(volume)
                impacts.append(price_change / volume_ratio)
                
        return np.mean(impacts) if impacts else float('inf')

    def _measure_market_resilience(
        self,
        price: List[float],
        volume: List[float],
        window: int
    ) -> float:
        """Measure market resilience"""
        if not price or not volume or len(price) < window:
            return 0.0
            
        # Calculate price impact and recovery
        resilience_scores = []
        for i in range(len(price)-window):
            if volume[i] > np.mean(volume):
                price_impact = abs(price[i+1] - price[i]) / price[i]
                price_recovery = abs(
                    price[i+window] - price[i+1]
                ) / price[i+1]
                
                if price_impact > 0:
                    resilience = 1 - (price_recovery / price_impact)
                    resilience_scores.append(max(0, min(1, resilience)))
                    
        return np.mean(resilience_scores) if resilience_scores else 0.0

    def _calculate_trading_activity(
        self,
        trades: List[Dict[str, Any]],
        timestamps: List[datetime]
    ) -> float:
        """Calculate normalized trading activity"""
        if not trades or not timestamps:
            return 0.0
            
        # Calculate trade frequency and size distribution
        intervals = []
        sizes = []
        
        for i in range(len(trades)-1):
            interval = (
                timestamps[i+1] - timestamps[i]
            ).total_seconds()
            intervals.append(interval)
            sizes.append(trades[i]['size'])
            
        if not intervals or not sizes:
            return 0.0
            
        # Normalize metrics
        avg_interval = np.mean(intervals)
        avg_size = np.mean(sizes)
        
        if avg_interval > 0 and avg_size > 0:
            return (1 / avg_interval) * (avg_size / np.std(sizes))
        return 0.0
def _determine_liquidity_level(
        self,
        metrics: LiquidityMetrics
    ) -> LiquidityLevel:
        """Determine overall liquidity level"""
        # Calculate composite score
        score = 0.0
        weights = {
            'spread': 0.3,
            'depth': 0.2,
            'turnover': 0.15,
            'impact': 0.15,
            'resilience': 0.2
        }
        
        # Normalize and weight metrics
        score += weights['spread'] * (
            1 - min(1, metrics.bid_ask_spread / self.spread_threshold)
        )
        score += weights['depth'] * (
            1 - min(1, metrics.depth_weighted_spread / (2 * self.spread_threshold))
        )
        score += weights['turnover'] * min(1, metrics.turnover_ratio)
        score += weights['impact'] * (1 - min(1, metrics.market_impact))
        score += weights['resilience'] * metrics.resilience
        
        # Determine level based on score
        if score > 0.8:
            return LiquidityLevel.HIGHLY_LIQUID
        elif score > 0.6:
            return LiquidityLevel.LIQUID
        elif score > 0.4:
            return LiquidityLevel.MODERATE
        elif score > 0.2:
            return LiquidityLevel.ILLIQUID
        else:
            return LiquidityLevel.HIGHLY_ILLIQUID

    def _analyze_liquidity_trend(
        self,
        metrics: LiquidityMetrics,
        historical_metrics: List[Dict[str, Any]]
    ) -> LiquidityTrend:
        """Analyze trend in liquidity metrics"""
        if not historical_metrics:
            return LiquidityTrend.STABLE
            
        # Calculate percentage changes in key metrics
        changes = {
            'spread': self._calculate_metric_trend(
                [m['bid_ask_spread'] for m in historical_metrics],
                metrics.bid_ask_spread
            ),
            'depth': self._calculate_metric_trend(
                [m['depth_weighted_spread'] for m in historical_metrics],
                metrics.depth_weighted_spread
            ),
            'turnover': self._calculate_metric_trend(
                [m['turnover_ratio'] for m in historical_metrics],
                metrics.turnover_ratio
            ),
            'impact': self._calculate_metric_trend(
                [m['market_impact'] for m in historical_metrics],
                metrics.market_impact
            ),
            'resilience': self._calculate_metric_trend(
                [m['resilience'] for m in historical_metrics],
                metrics.resilience
            )
        }
        
        # Weight the changes
        weights = {
            'spread': 0.3,
            'depth': 0.2,
            'turnover': 0.15,
            'impact': 0.15,
            'resilience': 0.2
        }
        
        weighted_change = sum(
            changes[metric] * weight 
            for metric, weight in weights.items()
        )
        
        # Determine trend based on weighted change
        if weighted_change < -0.2:
            return LiquidityTrend.CRITICAL
        elif weighted_change < -0.1:
            return LiquidityTrend.DETERIORATING
        elif weighted_change > 0.1:
            return LiquidityTrend.IMPROVING
        else:
            return LiquidityTrend.STABLE

    def _calculate_metric_trend(
        self,
        historical_values: List[float],
        current_value: float
    ) -> float:
        """Calculate percentage change in metric"""
        if not historical_values:
            return 0.0
            
        avg_historical = np.mean(historical_values)
        if avg_historical == 0:
            return 0.0
            
        return (current_value - avg_historical) / avg_historical

    def _calculate_risk_score(
        self,
        metrics: LiquidityMetrics,
        level: LiquidityLevel,
        trend: LiquidityTrend
    ) -> float:
        """Calculate overall liquidity risk score"""
        # Base risk score from liquidity level
        level_scores = {
            LiquidityLevel.HIGHLY_LIQUID: 0.1,
            LiquidityLevel.LIQUID: 0.3,
            LiquidityLevel.MODERATE: 0.5,
            LiquidityLevel.ILLIQUID: 0.7,
            LiquidityLevel.HIGHLY_ILLIQUID: 0.9
        }
        
        # Trend adjustments
        trend_adjustments = {
            LiquidityTrend.IMPROVING: -0.1,
            LiquidityTrend.STABLE: 0.0,
            LiquidityTrend.DETERIORATING: 0.1,
            LiquidityTrend.CRITICAL: 0.2
        }
        
        # Calculate base score
        base_score = level_scores[level]
        
        # Adjust for trend
        adjusted_score = base_score + trend_adjustments[trend]
        
        # Additional risk factors
        if metrics.amihud_illiquidity > 0.1:
            adjusted_score += 0.1
        if metrics.market_impact > 0.05:
            adjusted_score += 0.1
        if metrics.resilience < 0.3:
            adjusted_score += 0.1
            
        return min(1.0, max(0.0, adjusted_score))

    def _calculate_confidence(
        self,
        metrics: LiquidityMetrics,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence in liquidity analysis"""
        confidence_factors = []
        
        # Data quality confidence
        data_points = len(market_data.get('trades', []))
        data_confidence = min(1.0, data_points / self.min_data_points)
        confidence_factors.append(data_confidence)
        
        # Metric stability confidence
        stability_confidence = self._calculate_metric_stability(
            market_data.get('historical_metrics', [])
        )
        confidence_factors.append(stability_confidence)
        
        # Market conditions confidence
        market_confidence = self._calculate_market_confidence(
            metrics, market_data
        )
        confidence_factors.append(market_confidence)
        
        return np.mean(confidence_factors)

    def _calculate_metric_stability(
        self,
        historical_metrics: List[Dict[str, Any]]
    ) -> float:
        """Calculate stability of metrics over time"""
        if not historical_metrics:
            return 0.5
            
        # Calculate coefficient of variation for key metrics
        variations = []
        
        for metric in ['bid_ask_spread', 'turnover_ratio', 'market_impact']:
            values = [m[metric] for m in historical_metrics if metric in m]
            if values:
                cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else float('inf')
                variations.append(min(1.0, cv))
                
        if not variations:
            return 0.5
            
        # Convert average variation to stability score
        avg_variation = np.mean(variations)
        return 1.0 - avg_variation

    def _calculate_market_confidence(
        self,
        metrics: LiquidityMetrics,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence based on market conditions"""
        confidence = 1.0
        
        # Reduce confidence for extreme values
        if metrics.bid_ask_spread > self.spread_threshold * 2:
            confidence *= 0.8
        if metrics.market_impact > 0.1:
            confidence *= 0.8
        if metrics.resilience < 0.2:
            confidence *= 0.8
            
        # Reduce confidence for irregular volume patterns
        volume_profile = metrics.volume_profile
        if volume_profile:
            avg_volume = np.mean(list(volume_profile.values()))
            max_volume = max(volume_profile.values())
            if max_volume > avg_volume * 3:
                confidence *= 0.9
                
        return confidence

    def _identify_risk_factors(
        self,
        metrics: LiquidityMetrics,
        level: LiquidityLevel,
        trend: LiquidityTrend
    ) -> List[str]:
        """Identify key liquidity risk factors"""
        factors = []
        
        # Spread-related risks
        if metrics.bid_ask_spread > self.spread_threshold:
            factors.append(
                "Wide bid-ask spread indicating high transaction costs"
            )
        
        # Depth-related risks
        if metrics.depth_weighted_spread > self.spread_threshold * 1.5:
            factors.append(
                "Limited market depth suggesting potential for price impact"
            )
        
        # Volume-related risks
        if metrics.turnover_ratio < 0.02:
            factors.append(
                "Low turnover ratio indicating potential trading difficulties"
            )
        
        # Impact-related risks
        if metrics.market_impact > 0.05:
            factors.append(
                "High market impact coefficient suggesting price sensitivity"
            )
        
        # Resilience-related risks
        if metrics.resilience < 0.3:
            factors.append(
                "Low market resilience indicating slow recovery from trades"
            )
        
        # Trend-related risks
        if trend == LiquidityTrend.DETERIORATING:
            factors.append(
                "Deteriorating liquidity conditions"
            )
        elif trend == LiquidityTrend.CRITICAL:
            factors.append(
                "Critical decline in liquidity metrics"
            )
            
        return factors

    def _generate_liquidity_analysis(
        self,
        metrics: LiquidityMetrics,
        risk: LiquidityRisk
    ) -> str:
        """Generate detailed liquidity analysis"""
        analysis_parts = []
        
        # Overall assessment
        analysis_parts.append("## Liquidity Analysis\n")
        analysis_parts.append(
            f"Current Liquidity Level: {risk.level.value}\n"
            f"Liquidity Trend: {risk.trend.value}\n"
            f"Risk Score: {risk.risk_score:.2f}\n"
            f"Analysis Confidence: {risk.confidence:.2f}\n"
        )
        
        # Metric breakdown
        analysis_parts.append("\n## Core Metrics\n")
        analysis_parts.append(
            f"Bid-Ask Spread: {metrics.bid_ask_spread:.4f}\n"
            f"Depth-Weighted Spread: {metrics.depth_weighted_spread:.4f}\n"
            f"Turnover Ratio: {metrics.turnover_ratio:.4f}\n"
            f"Market Impact: {metrics.market_impact:.4f}\n"
            f"Market Resilience: {metrics.resilience:.4f}\n"
        )
        
        # Volume profile
        analysis_parts.append("\n## Volume Distribution\n")
        for period, volume in metrics.volume_profile.items():
            analysis_parts.append(f"Period {period}: {volume:.2%}\n")
        
        # Risk factors
        if risk.factors:
            analysis_parts.append("\n## Risk Factors\n")
            for factor in risk.factors:
                analysis_parts.append(f"- {factor}\n")
                
        return "\n".join(analysis_parts)

    def _generate_liquidity_recommendations(
        self,
        risk: LiquidityRisk
    ) -> List[str]:
        """Generate liquidity-based recommendations"""
        recommendations = []
        
        # Basic recommendations based on liquidity level
        if risk.level in [LiquidityLevel.HIGHLY_ILLIQUID, LiquidityLevel.ILLIQUID]:
            recommendations.extend([
                "Consider breaking large orders into smaller parts",
                "Implement careful position sizing",
                "Use limit orders instead of market orders",
                "Monitor execution costs closely"
            ])
            
        # Trend-based recommendations
        if risk.trend == LiquidityTrend.DETERIORATING:
            recommendations.extend([
                "Increase focus on risk management",
                "Review position sizes and adjust if necessary",
                "Consider reducing exposure in less liquid positions"
            ])
        elif risk.trend == LiquidityTrend.CRITICAL:
            recommendations.extend([
                "Immediate review of all positions required",
                "Consider temporary trading suspension",
                "Prepare contingency plans for further deterioration"
            ])
            
        # Risk factor specific recommendations
        for factor in risk.factors:
            if "bid-ask spread" in factor.lower():
                recommendations.append(
                    "Use more limit orders to minimize spread costs"
                )
            elif "market depth" in factor.lower():
                recommendations.append(
                    "Monitor order book depth before large trades"
                )
            elif "market impact" in factor.lower():
                recommendations.append(
                    "Implement advanced execution algorithms"
                )
                
        return recommendations

    def _generate_liquidity_summary(
        self,
        risk: LiquidityRisk
    ) -> str:
        """Generate concise liquidity summary"""
        summary_parts = []
        
        # Overall status
        summary_parts.append(
            f"Liquidity Status: {risk.level.value} "
            f"(Trend: {risk.trend.value})"
        )
        
        # Risk assessment
        summary_parts.append(
            f"\nRisk Score: {risk.risk_score:.2f} "
            f"(Confidence: {risk.confidence:.2f})"
        )
        
        # Key risk factors
        if risk.factors:
            summary_parts.append("\nKey Risk Factors:")
            for factor in risk.factors[:3]:  # Top 3 factors
                summary_parts.append(f"- {factor}")
                
        # Critical warnings
        if risk.trend == LiquidityTrend.CRITICAL:
            summary_parts.append(
                "\nWARNING: Critical deterioration in liquidity conditions"
            )
            
        return "\n".join(summary_parts)


    def _validate_market_data(self, market_data: Dict[str, Any]) -> None:
        """Validate required market data fields"""
        required_fields = [
            'bids', 'asks', 'trades', 'volume', 'price',
            'timestamps', 'float_shares'
        ]
        
        missing_fields = [
            field for field in required_fields
            if field not in market_data
        ]
        
        if missing_fields:
            raise AnalysisError(
                f"Missing required market data fields: {missing_fields}"
            )
            
        # Validate data points
        if len(market_data['trades']) < self.min_data_points:
            raise AnalysisError(
                f"Insufficient data points. Required: {self.min_data_points}, "
                f"Got: {len(market_data['trades'])}"
            )

    def _find_surrounding_quotes(
        self,
        trade_timestamp: datetime,
        quotes: List[Dict[str, Any]],
        time_window: timedelta = timedelta(seconds=1)
    ) -> Optional[Dict[str, Any]]:
        """Find closest quotes around a trade timestamp"""
        if not quotes:
            return None
            
        # Find closest quote before and after trade
        before_quotes = [
            q for q in quotes
            if q['timestamp'] <= trade_timestamp
            and trade_timestamp - q['timestamp'] <= time_window
        ]
        
        after_quotes = [
            q for q in quotes
            if q['timestamp'] > trade_timestamp
            and q['timestamp'] - trade_timestamp <= time_window
        ]
        
        if not before_quotes or not after_quotes:
            return None
            
        # Get closest quotes
        before_quote = max(before_quotes, key=lambda q: q['timestamp'])
        after_quote = min(after_quotes, key=lambda q: q['timestamp'])
        
        # Interpolate quote prices
        time_diff_before = (trade_timestamp - before_quote['timestamp']).total_seconds()
        time_diff_after = (after_quote['timestamp'] - trade_timestamp).total_seconds()
        total_time_diff = time_diff_before + time_diff_after
        
        if total_time_diff == 0:
            weight_before = 0.5
        else:
            weight_before = 1 - (time_diff_before / total_time_diff)
            
        weight_after = 1 - weight_before
        
        return {
            'bid_price': (
                weight_before * before_quote['bid_price'] +
                weight_after * after_quote['bid_price']
            ),
            'ask_price': (
                weight_before * before_quote['ask_price'] +
                weight_after * after_quote['ask_price']
            ),
            'timestamp': trade_timestamp
        }

    def _calculate_order_book_imbalance(
        self,
        bids: List[Dict[str, Any]],
        asks: List[Dict[str, Any]],
        levels: int = 5
    ) -> float:
        """Calculate order book imbalance"""
        bid_volume = sum(
            bid['size']
            for bid in bids[:levels]
        )
        
        ask_volume = sum(
            ask['size']
            for ask in asks[:levels]
        )
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
            
        return (bid_volume - ask_volume) / total_volume

    def _calculate_effective_spread(
        self,
        trades: List[Dict[str, Any]],
        quotes: List[Dict[str, Any]]
    ) -> float:
        """Calculate effective spread from trades and quotes"""
        spreads = []
        
        for trade in trades:
            surrounding_quotes = self._find_surrounding_quotes(
                trade['timestamp'],
                quotes
            )
            
            if surrounding_quotes:
                mid_price = (
                    surrounding_quotes['bid_price'] +
                    surrounding_quotes['ask_price']
                ) / 2
                
                effective_spread = abs(
                    trade['price'] - mid_price
                ) / mid_price
                
                spreads.append(effective_spread)
                
        return np.mean(spreads) if spreads else float('inf')

    def _calculate_kyle_lambda(
        self,
        trades: List[Dict[str, Any]],
        price: List[float],
        volume: List[float]
    ) -> float:
        """Calculate Kyle's lambda (price impact coefficient)"""
        if len(trades) < 2:
            return float('inf')
            
        # Calculate price changes and signed volumes
        price_changes = np.diff(price)
        signed_volumes = np.array([
            trade['size'] * (1 if trade['side'] == 'buy' else -1)
            for trade in trades[:-1]
        ])
        
        # Regress price changes on signed volumes
        if len(price_changes) == len(signed_volumes):
            try:
                lambda_coef = np.polyfit(
                    signed_volumes,
                    price_changes,
                    deg=1
                )[0]
                return abs(lambda_coef)
            except:
                return float('inf')
        return float('inf')

    def _calculate_liquidity_timing(
        self,
        volume_profile: Dict[str, float],
        target_volume: float
    ) -> List[Dict[str, Any]]:
        """Calculate optimal trading times based on volume profile"""
        sorted_periods = sorted(
            volume_profile.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        remaining_volume = target_volume
        trading_schedule = []
        
        for period, relative_volume in sorted_periods:
            if remaining_volume <= 0:
                break
                
            # Calculate optimal volume for this period
            period_volume = min(
                remaining_volume,
                target_volume * relative_volume * 1.5  # Allow 50% more than profile
            )
            
            if period_volume > 0:
                trading_schedule.append({
                    'period': period,
                    'target_volume': period_volume,
                    'expected_volume': target_volume * relative_volume,
                    'volume_share': relative_volume
                })
                
            remaining_volume -= period_volume
            
        return trading_schedule

    def _calculate_market_quality(
        self,
        metrics: LiquidityMetrics,
        historical_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overall market quality indicators"""
        # Current quality metrics
        quality_metrics = {
            'spread_quality': 1 - min(
                1,
                metrics.bid_ask_spread / self.spread_threshold
            ),
            'depth_quality': 1 - min(
                1,
                metrics.depth_weighted_spread / (2 * self.spread_threshold)
            ),
            'resilience_quality': metrics.resilience,
            'impact_quality': 1 - min(1, metrics.market_impact)
        }
        
        # Historical comparison
        if historical_metrics:
            for metric in quality_metrics.keys():
                historical_values = [
                    m.get(metric, 0.0) for m in historical_metrics
                ]
                if historical_values:
                    avg_historical = np.mean(historical_values)
                    if avg_historical > 0:
                        quality_metrics[f'{metric}_trend'] = (
                            quality_metrics[metric] / avg_historical - 1
                        )
                    else:
                        quality_metrics[f'{metric}_trend'] = 0.0
                        
        # Overall quality score
        weights = {
            'spread_quality': 0.3,
            'depth_quality': 0.3,
            'resilience_quality': 0.2,
            'impact_quality': 0.2
        }
        
        quality_metrics['overall_score'] = sum(
            metric * weights[name]
            for name, metric in quality_metrics.items()
            if name in weights
        )
        
        return quality_metrics

    def _get_execution_strategy(
        self,
        risk: LiquidityRisk,
        target_volume: float
    ) -> Dict[str, Any]:
        """Generate execution strategy recommendations"""
        strategy = {
            'urgency': 'normal',
            'order_types': ['limit'],
            'timing_strategy': 'volume_weighted',
            'max_participation_rate': 0.15,
            'price_limit_buffer': 0.01
        }
        
        # Adjust based on liquidity level
        if risk.level == LiquidityLevel.HIGHLY_LIQUID:
            strategy.update({
                'urgency': 'high',
                'order_types': ['limit', 'market'],
                'max_participation_rate': 0.25,
                'price_limit_buffer': 0.005
            })
        elif risk.level == LiquidityLevel.ILLIQUID:
            strategy.update({
                'urgency': 'low',
                'order_types': ['limit'],
                'max_participation_rate': 0.10,
                'price_limit_buffer': 0.02,
                'require_blocks': True
            })
            
        # Adjust for trend
        if risk.trend == LiquidityTrend.DETERIORATING:
            strategy['urgency'] = 'high'
            strategy['max_participation_rate'] *= 0.8
        elif risk.trend == LiquidityTrend.CRITICAL:
            strategy['urgency'] = 'immediate'
            strategy['order_types'] = ['limit', 'market']
            strategy['max_participation_rate'] *= 0.5
            
        # Calculate trading schedule
        if 'volume_profile' in risk.metrics.__dict__:
            strategy['trading_schedule'] = self._calculate_liquidity_timing(
                risk.metrics.volume_profile,
                target_volume
            )
            
        # Add cost estimates
        strategy['cost_estimate'] = {
            'spread_cost': risk.metrics.bid_ask_spread / 2,
            'market_impact': risk.metrics.market_impact * target_volume,
            'timing_cost': 0.01 if strategy['urgency'] == 'immediate' else 0.0
        }
        
        return strategy

    def _generate_monitoring_thresholds(
        self,
        metrics: LiquidityMetrics,
        risk: LiquidityRisk
    ) -> Dict[str, float]:
        """Generate monitoring thresholds for liquidity metrics"""
        return {
            'max_spread': metrics.bid_ask_spread * 1.5,
            'min_depth': metrics.depth_weighted_spread * 0.5,
            'max_impact': metrics.market_impact * 1.3,
            'min_resilience': metrics.resilience * 0.7,
            'max_turnover': metrics.turnover_ratio * 2,
            'risk_score_threshold': risk.risk_score + 0.1,
            'update_interval_seconds': 300 if risk.trend != LiquidityTrend.CRITICAL else 60
        }

    async def get_real_time_metrics(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate real-time liquidity metrics for monitoring"""
        try:
            metrics = await self._calculate_liquidity_metrics(market_data)
            risk = await self._analyze_liquidity_risk(
                metrics,
                market_data
            )
            
            return {
                'current_metrics': {
                    'bid_ask_spread': metrics.bid_ask_spread,
                    'depth_weighted_spread': metrics.depth_weighted_spread,
                    'market_impact': metrics.market_impact,
                    'resilience': metrics.resilience,
                    'risk_score': risk.risk_score
                },
                'thresholds': self._generate_monitoring_thresholds(
                    metrics,
                    risk
                ),
                'status': {
                    'liquidity_level': risk.level.value,
                    'trend': risk.trend.value,
                    'warning_level': 'critical' if risk.trend == LiquidityTrend.CRITICAL
                        else 'warning' if risk.trend == LiquidityTrend.DETERIORATING
                        else 'normal'
                }
            }
            
        except Exception as e:
            raise AnalysisError(f"Real-time metrics calculation failed: {str(e)}")

    async def simulate_trade_impact(
        self,
        market_data: Dict[str, Any],
        trade_size: float,
        trade_direction: str
    ) -> Dict[str, Any]:
        """Simulate market impact of a potential trade"""
        try:
            # Get current metrics
            metrics = await self._calculate_liquidity_metrics(market_data)
            
            # Estimate price impact
            estimated_impact = (
                metrics.market_impact *
                trade_size *
                (1 if trade_direction == 'buy' else -1)
            )
            
            # Estimate spread cost
            spread_cost = metrics.bid_ask_spread / 2
            
            # Estimate market depth impact
            depth_impact = (
                metrics.depth_weighted_spread *
                trade_size /
                sum(bid['size'] for bid in market_data['bids'][:self.depth_levels])
            )
            
            return {
                'estimated_price_impact': estimated_impact,
                'estimated_spread_cost': spread_cost,
                'estimated_depth_impact': depth_impact,
                'total_cost_estimate': estimated_impact + spread_cost + depth_impact,
                'execution_risk': 'high' if depth_impact > 0.02 else 'medium' if depth_impact > 0.01 else 'low',
                'recommended_sizing': {
                    'max_single_trade': sum(bid['size'] for bid in market_data['bids'][:3]) * 0.1,
                    'max_total_size': sum(bid['size'] for bid in market_data['bids']) * 0.2,
                    'recommended_splits': max(1, int(trade_size / (sum(bid['size'] for bid in market_data['bids'][:3]) * 0.05)))
                }
            }
            
        except Exception as e:
            raise AnalysisError(f"Trade impact simulation failed: {str(e)}")
