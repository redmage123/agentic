# services/generative_agent/application/analyzers/technical_analyzer.py

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import numpy as np
from dataclasses import dataclass

from ...domain.models import AnalysisResponse, AnalysisType, ConfidenceLevel
from ...domain.exceptions import AnalysisError

class TrendDirection(Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

class PatternType(Enum):
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    CONSOLIDATION = "consolidation"

@dataclass
class ChartPattern:
    pattern_name: str
    pattern_type: PatternType
    confidence: float
    price_target: Optional[float]
    support_level: Optional[float]
    resistance_level: Optional[float]
    volume_confirms: bool
    timeframe: str

class TechnicalAnalyzer:
    """
    Analyzes market data using technical analysis techniques including:
    - Trend analysis using multiple timeframes
    - Moving averages and momentum indicators
    - Volume analysis and price patterns
    - Support/resistance levels
    - Technical chart patterns
    """
    
    def __init__(
        self,
        short_term_period: int = 20,
        medium_term_period: int = 50,
        long_term_period: int = 200,
        rsi_period: int = 14,
        macd_periods: Tuple[int, int, int] = (12, 26, 9),
        volume_ma_period: int = 20,
        confidence_threshold: float = 0.7
    ):
        self.short_term_period = short_term_period
        self.medium_term_period = medium_term_period
        self.long_term_period = long_term_period
        self.rsi_period = rsi_period
        self.macd_fast, self.macd_slow, self.macd_signal = macd_periods
        self.volume_ma_period = volume_ma_period
        self.confidence_threshold = confidence_threshold

    async def analyze(
        self,
        market_data: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> AnalysisResponse:
        """
        Process technical analysis using multiple indicators and timeframes
        """
        try:
            # Extract price and volume data
            prices = np.array(market_data['prices'])
            volumes = np.array(market_data['volumes'])
            timestamps = market_data['timestamps']

            # Perform various technical analyses
            trend_analysis = await self._analyze_trends(prices, volumes)
            moving_averages = await self._analyze_moving_averages(prices)
            momentum_indicators = await self._analyze_momentum(prices)
            volume_analysis = await self._analyze_volume(volumes, prices)
            support_resistance = await self._analyze_support_resistance(prices)
            patterns = await self._identify_patterns(prices, volumes)

            # Combine analyses
            combined_analysis = await self._combine_analyses(
                trend_analysis,
                moving_averages,
                momentum_indicators,
                volume_analysis,
                support_resistance,
                patterns
            )

            # Generate analysis and recommendations
            detailed_analysis = self._generate_technical_analysis(
                combined_analysis,
                timestamps
            )

            recommendations = self._generate_technical_recommendations(
                combined_analysis
            )

            return AnalysisResponse(
                request_id=metadata.get('request_id', ''),
                analysis_type=AnalysisType.TECHNICAL,
                summary=self._generate_technical_summary(combined_analysis),
                detailed_analysis=detailed_analysis,
                recommendations=recommendations,
                confidence_score=combined_analysis['confidence'],
                metadata={
                    **metadata,
                    'trend_direction': combined_analysis['trend'].value,
                    'key_levels': combined_analysis['key_levels'],
                    'pattern_count': len(combined_analysis['patterns'])
                },
                processing_time=metadata.get('processing_time', 0),
                timestamp=datetime.now()
            )

        except Exception as e:
            raise AnalysisError(f"Technical analysis failed: {str(e)}")

    async def _analyze_trends(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze price trends across multiple timeframes"""
        
        # Calculate trend indicators
        short_trend = self._calculate_trend_direction(
            prices, self.short_term_period
        )
        medium_trend = self._calculate_trend_direction(
            prices, self.medium_term_period
        )
        long_trend = self._calculate_trend_direction(
            prices, self.long_term_period
        )

        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(
            prices, volumes, self.medium_term_period
        )

        # Determine overall trend
        overall_trend = self._determine_overall_trend(
            short_trend,
            medium_trend,
            long_trend,
            trend_strength
        )

        return {
            'short_term': short_trend,
            'medium_term': medium_trend,
            'long_term': long_trend,
            'strength': trend_strength,
            'overall_trend': overall_trend,
            'confidence': self._calculate_trend_confidence(
                short_trend,
                medium_trend,
                long_trend,
                trend_strength
            )
        }

    async def _analyze_moving_averages(
        self,
        prices: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze moving averages and their relationships"""
        
        # Calculate various MAs
        sma_short = self._calculate_sma(prices, self.short_term_period)
        sma_medium = self._calculate_sma(prices, self.medium_term_period)
        sma_long = self._calculate_sma(prices, self.long_term_period)
        
        ema_short = self._calculate_ema(prices, self.short_term_period)
        ema_medium = self._calculate_ema(prices, self.medium_term_period)
        ema_long = self._calculate_ema(prices, self.long_term_period)

        # Analyze MA crossovers
        crossovers = self._analyze_ma_crossovers(
            sma_short, sma_medium, sma_long,
            ema_short, ema_medium, ema_long
        )

        # Calculate price relative to MAs
        price_relative = self._analyze_price_relative_to_mas(
            prices,
            sma_short, sma_medium, sma_long,
            ema_short, ema_medium, ema_long
        )

        return {
            'moving_averages': {
                'sma': {
                    'short': sma_short[-1],
                    'medium': sma_medium[-1],
                    'long': sma_long[-1]
                },
                'ema': {
                    'short': ema_short[-1],
                    'medium': ema_medium[-1],
                    'long': ema_long[-1]
                }
            },
            'crossovers': crossovers,
            'price_relative': price_relative,
            'confidence': self._calculate_ma_confidence(
                prices, sma_short, sma_medium, sma_long
            )
        }

    async def _analyze_momentum(
        self,
        prices: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze momentum indicators"""
        
        # Calculate RSI
        rsi = self._calculate_rsi(prices, self.rsi_period)
        
        # Calculate MACD
        macd, signal, histogram = self._calculate_macd(
            prices,
            self.macd_fast,
            self.macd_slow,
            self.macd_signal
        )

        # Calculate Rate of Change
        roc = self._calculate_roc(prices, self.short_term_period)

        # Momentum strength and divergences
        momentum_strength = self._calculate_momentum_strength(
            rsi, macd, histogram, roc
        )
        divergences = self._identify_divergences(
            prices, rsi, macd, volumes
        )

        return {
            'indicators': {
                'rsi': float(rsi[-1]),
                'macd': float(macd[-1]),
                'macd_signal': float(signal[-1]),
                'macd_histogram': float(histogram[-1]),
                'roc': float(roc[-1])
            },
            'momentum_strength': momentum_strength,
            'divergences': divergences,
            'confidence': self._calculate_momentum_confidence(
                rsi, macd, histogram, roc
            )
        }

    async def _analyze_volume(
        self,
        volumes: np.ndarray,
        prices: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze volume patterns and trends"""
        
        # Calculate volume metrics
        volume_ma = self._calculate_sma(volumes, self.volume_ma_period)
        volume_trend = self._calculate_volume_trend(volumes)
        
        # Volume/price relationship
        price_volume_correlation = self._calculate_price_volume_correlation(
            prices, volumes
        )
        
        # Identify volume patterns
        volume_patterns = self._identify_volume_patterns(
            volumes, prices, volume_ma
        )

        # Calculate buying/selling pressure
        pressure = self._calculate_buying_selling_pressure(
            volumes, prices
        )

        return {
            'metrics': {
                'current_volume': float(volumes[-1]),
                'volume_ma': float(volume_ma[-1]),
                'volume_trend': volume_trend
            },
            'patterns': volume_patterns,
            'price_volume_correlation': price_volume_correlation,
            'pressure': pressure,
            'confidence': self._calculate_volume_confidence(
                volumes, volume_ma, price_volume_correlation
            )
        }

    async def _analyze_support_resistance(
        self,
        prices: np.ndarray
    ) -> Dict[str, Any]:
        """Identify support and resistance levels"""
        
        # Find key price levels
        support_levels = self._identify_support_levels(prices)
        resistance_levels = self._identify_resistance_levels(prices)
        
        # Calculate level strength
        level_strength = self._calculate_level_strength(
            prices, support_levels, resistance_levels
        )
        
        # Identify price channels
        channels = self._identify_price_channels(prices)

        return {
            'levels': {
                'support': support_levels,
                'resistance': resistance_levels
            },
            'strength': level_strength,
            'channels': channels,
            'confidence': self._calculate_level_confidence(
                prices, support_levels, resistance_levels
            )
        }

    async def _identify_patterns(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> List[ChartPattern]:
        """Identify technical chart patterns"""
        patterns = []
        
        # Identify reversal patterns
        reversal_patterns = self._identify_reversal_patterns(
            prices, volumes
        )
        patterns.extend(reversal_patterns)
        
        # Identify continuation patterns
        continuation_patterns = self._identify_continuation_patterns(
            prices, volumes
        )
        patterns.extend(continuation_patterns)
        
        # Identify consolidation patterns
        consolidation_patterns = self._identify_consolidation_patterns(
            prices, volumes
        )
        patterns.extend(consolidation_patterns)

        return sorted(
            patterns,
            key=lambda x: x.confidence,
            reverse=True
        )

    def _calculate_sma(
        self,
        data: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Calculate Simple Moving Average"""
        return np.convolve(
            data,
            np.ones(period) / period,
            mode='valid'
        )

    def _calculate_ema(
        self,
        data: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        return np.array([
            data[0] if i == 0
            else alpha * data[i] + (1 - alpha) * ema[i-1]
            for i, _ in enumerate(data)
        ])

    def _calculate_rsi(
        self,
        prices: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Calculate Relative Strength Index"""
        delta = np.diff(prices)
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        
        avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
        
        rs = avg_gain / np.where(avg_loss == 0, 1, avg_loss)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_macd(
        self,
        prices: np.ndarray,
        fast_period: int,
        slow_period: int,
        signal_period: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD, Signal, and Histogram"""
        ema_fast = self._calculate_ema(prices, fast_period)
        ema_slow = self._calculate_ema(prices, slow_period)
        
        macd = ema_fast - ema_slow
        signal = self._calculate_ema(macd, signal_period)
        histogram = macd - signal
        
        return macd, signal, histogram

    def _calculate_trend_strength(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        period: int
    ) -> float:
        """Calculate trend strength using price and volume"""
        price_change = np.abs(prices[-1] - prices[-period]) / prices[-period]
        volume_change = np.mean(volumes[-period:]) / np.mean(volumes[-period*2:-period])
        
        return (price_change * volume_change) ** 0.5

    def _determine_overall_trend(
        self,
        short_trend: TrendDirection,
        medium_trend: TrendDirection,
        long_trend: TrendDirection,
        trend_strength: float
    ) -> TrendDirection:
        """Determine overall trend direction"""
        # Convert trend directions to numerical values
        trend_values = {
            TrendDirection.STRONG_UPTREND: 2,
            TrendDirection.UPTREND: 1,
            TrendDirection.SIDEWAYS: 0,
            TrendDirection.DOWNTREND: -1,
            TrendDirection.STRONG_DOWNTREND: -2
        }
        
        # Calculate weighted average
        weights = [0.5, 0.3, 0.2]  # Short, medium, long weights
        trends = [short_trend, medium_trend, long_trend]
        
        weighted_trend = sum(
            trend_values[trend] * weight
            for trend, weight in zip(trends, weights)
        )
        
        # Adjust for trend strength
        weighted_trend *= trend_strength
        
        # Convert back to TrendDirection
        if weighted_trend > 1.5:
            return TrendDirection.STRONG_UPTREND
        elif weighted_trend > 0.5:
            return TrendDirection.UPTREND
        elif weighted_trend > -0.
