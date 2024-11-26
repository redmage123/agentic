# services/generative_agent/application/analyzers/volatility_analyzer.py

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar

from ...domain.models import AnalysisResponse, AnalysisType, ConfidenceLevel
from ...domain.exceptions import AnalysisError

class VolatilityRegime(Enum):
    VERY_LOW = "very_low"          # < 10%
    LOW = "low"                    # 10-20%
    MODERATE = "moderate"          # 20-30%
    HIGH = "high"                  # 30-50%
    VERY_HIGH = "very_high"        # > 50%

class VolatilityTrend(Enum):
    INCREASING = "increasing"
    STABLE = "stable"
    DECREASING = "decreasing"
    REGIME_CHANGE = "regime_change"

@dataclass
class VolatilityMetrics:
    """Core volatility metrics"""
    historical_vol: float              # Historical volatility
    implied_vol_atm: float            # At-the-money implied volatility
    implied_vol_surface: Dict[str, float]  # Vol surface by strike/expiry
    realized_vol: float               # Realized volatility
    forward_vol: float                # Forward-looking volatility
    vol_of_vol: float                 # Volatility of volatility
    skew: float                       # Volatility skew
    term_structure: Dict[str, float]  # Term structure of volatility

@dataclass
class VolatilityCone:
    """Volatility cone analysis"""
    percentiles: Dict[str, List[float]]  # Volatility percentiles by tenor
    current_vol: Dict[str, float]      # Current volatility by tenor
    zscore: Dict[str, float]           # Z-scores by tenor

@dataclass
class BlackScholesInputs:
    """Inputs for Black-Scholes calculations"""
    spot: float
    strike: float
    rate: float
    time_to_expiry: float
    option_price: float
    option_type: str  # 'call' or 'put'

class VolatilityAnalyzer:
    """
    Analyzes market volatility using multiple approaches:
    - Historical volatility calculation
    - Black-Scholes implied volatility
    - Volatility surface modeling
    - Regime detection
    - Forward volatility estimation
    """
    
    def __init__(
        self,
        lookback_window: int = 252,     # Trading days
        vol_window: int = 20,           # Days for historical vol
        confidence_level: float = 0.95,
        min_data_points: int = 30
    ):
        self.lookback_window = lookback_window
        self.vol_window = vol_window
        self.confidence_level = confidence_level
        self.min_data_points = min_data_points

    async def analyze(
        self,
        market_data: Dict[str, Any],
        options_data: Optional[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None
    ) -> AnalysisResponse:
        """Process volatility analysis using market and options data"""
        try:
            # Calculate historical volatility metrics
            vol_metrics = await self._calculate_volatility_metrics(
                market_data,
                options_data
            )
            
            # Analyze volatility regime
            regime_analysis = await self._analyze_volatility_regime(
                vol_metrics,
                market_data['historical_metrics']
            )
            
            # Generate volatility cone
            vol_cone = await self._generate_volatility_cone(
                market_data['returns'],
                market_data['timestamps']
            )
            
            # Generate detailed analysis
            detailed_analysis = self._generate_volatility_analysis(
                vol_metrics,
                regime_analysis,
                vol_cone
            )
            
            # Generate recommendations
            recommendations = self._generate_volatility_recommendations(
                vol_metrics,
                regime_analysis,
                vol_cone
            )

            return AnalysisResponse(
                request_id=metadata.get('request_id', ''),
                analysis_type=AnalysisType.VOLATILITY,
                summary=self._generate_volatility_summary(
                    vol_metrics,
                    regime_analysis
                ),
                detailed_analysis=detailed_analysis,
                recommendations=recommendations,
                confidence_score=self._calculate_confidence(
                    vol_metrics,
                    market_data
                ),
                metadata={
                    **metadata,
                    'regime': regime_analysis['regime'].value,
                    'trend': regime_analysis['trend'].value,
                    'metrics': vol_metrics.__dict__
                }
            )

        except Exception as e:
            raise AnalysisError(f"Volatility analysis failed: {str(e)}")

    def _calculate_black_scholes_call(
        self,
        inputs: BlackScholesInputs
    ) -> float:
        """Calculate Black-Scholes call option price"""
        S = inputs.spot
        K = inputs.strike
        r = inputs.rate
        t = inputs.time_to_expiry
        sigma = inputs.implied_vol if hasattr(inputs, 'implied_vol') else 0.2
        
        d1 = (np.log(S/K) + (r + sigma**2/2)*t) / (sigma*np.sqrt(t))
        d2 = d1 - sigma*np.sqrt(t)
        
        call = S*norm.cdf(d1) - K*np.exp(-r*t)*norm.cdf(d2)
        return call

    def _calculate_black_scholes_put(
        self,
        inputs: BlackScholesInputs
    ) -> float:
        """Calculate Black-Scholes put option price"""
        S = inputs.spot
        K = inputs.strike
        r = inputs.rate
        t = inputs.time_to_expiry
        sigma = inputs.implied_vol if hasattr(inputs, 'implied_vol') else 0.2
        
        d1 = (np.log(S/K) + (r + sigma**2/2)*t) / (sigma*np.sqrt(t))
        d2 = d1 - sigma*np.sqrt(t)
        
        put = K*np.exp(-r*t)*norm.cdf(-d2) - S*norm.cdf(-d1)
        return put

    def _calculate_implied_volatility(
        self,
        inputs: BlackScholesInputs
    ) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        def objective(sigma):
            inputs.implied_vol = sigma
            if inputs.option_type == 'call':
                price = self._calculate_black_scholes_call(inputs)
            else:
                price = self._calculate_black_scholes_put(inputs)
            return abs(price - inputs.option_price)

        result = minimize_scalar(
            objective,
            bounds=(0.0001, 5.0),
            method='bounded'
        )
        
        return result.x if result.success else None

    async def _calculate_volatility_metrics(
        self,
        market_data: Dict[str, Any],
        options_data: Optional[Dict[str, Any]] = None
    ) -> VolatilityMetrics:
        """Calculate comprehensive volatility metrics"""
        
        # Calculate historical volatility
        returns = market_data['returns']
        hist_vol = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Calculate realized volatility
        realized_vol = self._calculate_realized_volatility(
            returns,
            self.vol_window
        )
        
        # Calculate implied volatility if options data available
        implied_vol_atm = 0.0
        implied_vol_surface = {}
        if options_data:
            implied_vol_atm = await self._calculate_atm_implied_vol(
                options_data
            )
            implied_vol_surface = await self._calculate_vol_surface(
                options_data
            )
            
        # Calculate forward volatility
        forward_vol = self._calculate_forward_volatility(
            returns,
            realized_vol
        )
        
        # Calculate volatility of volatility
        vol_of_vol = self._calculate_vol_of_vol(returns)
        
        # Calculate volatility skew
        skew = self._calculate_volatility_skew(options_data)
        
        # Calculate term structure
        term_structure = self._calculate_term_structure(options_data)
        
        return VolatilityMetrics(
            historical_vol=hist_vol,
            implied_vol_atm=implied_vol_atm,
            implied_vol_surface=implied_vol_surface,
            realized_vol=realized_vol,
            forward_vol=forward_vol,
            vol_of_vol=vol_of_vol,
            skew=skew,
            term_structure=term_structure
        )

    def _calculate_realized_volatility(
        self,
        returns: List[float],
        window: int
    ) -> float:
        """Calculate realized volatility over a specific window"""
        if len(returns) < window:
            return 0.0
            
        rolling_vol = np.std(returns[-window:]) * np.sqrt(252)
        return rolling_vol

    async def _calculate_atm_implied_vol(
        self,
        options_data: Dict[str, Any]
    ) -> float:
        """Calculate at-the-money implied volatility"""
        if not options_data:
            return 0.0
            
        spot = options_data['spot_price']
        atm_options = [
            opt for opt in options_data['options']
            if 0.95 <= opt['strike'] / spot <= 1.05
        ]
        
        if not atm_options:
            return 0.0
            
        implied_vols = []
        for opt in atm_options:
            inputs = BlackScholesInputs(
                spot=spot,
                strike=opt['strike'],
                rate=options_data['risk_free_rate'],
                time_to_expiry=opt['time_to_expiry'],
                option_price=opt['price'],
                option_type=opt['type']
            )
            
            implied_vol = self._calculate_implied_volatility(inputs)
            if implied_vol:
                implied_vols.append(implied_vol)
                
        return np.mean(implied_vols) if implied_vols else 0.0

    async def _calculate_vol_surface(
        self,
        options_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate complete implied volatility surface"""
        if not options_data:
            return {}
            
        vol_surface = {}
        spot = options_data['spot_price']
        
        for opt in options_data['options']:
            moneyness = opt['strike'] / spot
            tenor = opt['time_to_expiry']
            
            inputs = BlackScholesInputs(
                spot=spot,
                strike=opt['strike'],
                rate=options_data['risk_free_rate'],
                time_to_expiry=tenor,
                option_price=opt['price'],
                option_type=opt['type']
            )
            
            implied_vol = self._calculate_implied_volatility(inputs)
            if implied_vol:
                key = f"{moneyness:.2f}_{tenor:.2f}"
                vol_surface[key] = implied_vol
                
        return vol_surface

    def _calculate_forward_volatility(
        self,
        returns: List[float],
        current_vol: float
    ) -> float:
        """Calculate forward-looking volatility estimate"""
        if len(returns) < self.min_data_points:
            return current_vol
            
        # Use EWMA for forward volatility
        lambda_param = 0.94
        forward_var = 0
        
        for i, ret in enumerate(reversed(returns)):
            weight = (1 - lambda_param) * lambda_param**i
            forward_var += weight * ret**2
            
        return np.sqrt(forward_var * 252)

    def _calculate_vol_of_vol(
        self,
        returns: List[float],
        window: int = 20
    ) -> float:
        """Calculate volatility of volatility"""
        if len(returns) < window + 5:
            return 0.0
            
        # Calculate rolling volatility series
        vol_series = []
        for i in range(len(returns) - window + 1):
            window_returns = returns[i:i+window]
            vol = np.std(window_returns) * np.sqrt(252)
            vol_series.append(vol)
            
        # Calculate volatility of volatility series
        return np.std(vol_series) * np.sqrt(252)

    def _calculate_volatility_skew(
        self,
        options_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate volatility skew"""
        if not options_data:
            return 0.0
            
        # Find ATM volatility
        spot = options_data['spot_price']
        atm_vol = None
        otm_put_vol = None
        otm_call_vol = None
        
        for opt in options_data['options']:
            moneyness = opt['strike'] / spot
            
            inputs = BlackScholesInputs(
                spot=spot,
                strike=opt['strike'],
                rate=options_data['risk_free_rate'],
                time_to_expiry=opt['time_to_expiry'],
                option_price=opt['price'],
                option_type=opt['type']
            )
            
            implied_vol = self._calculate_implied_volatility(inputs)
            
            if 0.98 <= moneyness <= 1.02:  # ATM
                atm_vol = implied_vol
            elif moneyness < 0.95 and opt['type'] == 'put':  # OTM Put
                otm_put_vol = implied_vol
            elif moneyness > 1.05 and opt['type'] == 'call':  # OTM Call
                otm_call_vol = implied_vol
                
        if atm_vol and otm_put_vol and otm_call_vol:
            return (otm_put_vol - otm_call_vol) / atm_vol
        return 0.0

    def _calculate_term_structure(
        self,
        options_data: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate volatility term structure"""
        term_structure = {}
        
        if not options_data:
            return term_structure
            
        spot = options_data['spot_price']
        tenors = sorted(list(set(opt['time_to_expiry'] for opt in options_data['options'])))
        
        for tenor in tenors:
            tenor_options = [
                opt for opt in options_data['options']
                if opt['time_to_expiry'] == tenor
                and 0.95 <= opt['strike'] / spot <= 1.05  # ATM options
            ]
            
            if tenor_options:
                vols = []
                for opt in tenor_options:
                    inputs = BlackScholesInputs(
                        spot=spot,
                        strike=opt['strike'],
                        rate=options_data['risk_free_rate'],
                        time_to_expiry=tenor,
                        option_price=opt['price'],
                        option_type=opt['type']
                    )
                    vol = self._calculate_implied_volatility(inputs)
                    if vol:
                        vols.append(vol)
                        
                if vols:
                    term_structure[str(tenor)] = np.mean(vols)
                    
        return term_structure

    async def _analyze_volatility_regime(
        self,
        metrics: VolatilityMetrics,
        historical_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze volatility regime and trends"""
        
        # Determine current regime
        regime = self._determine_volatility_regime(metrics.historical_vol)
        
        # Analyze trend
        trend = self._analyze_volatility_trend(
            metrics,
            historical_metrics
        )
        
        # Calculate regime stability
        stability = self._calculate_regime_stability(
            metrics.historical_vol,
            historical_metrics
        )
        
        # Identify regime characteristics
        characteristics = self._identify_regime_characteristics(
            metrics,
            regime
        )
        
        return {
            'regime': regime,
            'trend': trend,
            'stability': stability,
            'characteristics': characteristics,
            'confidence': self._calculate_regime_confidence(
                metrics,
                historical_metrics
            )
        }

    def _determine_volatility_regime(self, vol: float) -> VolatilityRegime:
        """Determine volatility regime based on annualized volatility"""
        if vol < 0.10:  # 10%
            return VolatilityRegime.VERY_LOW
        elif vol < 0.20:  # 20%
            return VolatilityRegime.LOW
        elif vol < 0.30:  # 30%
            return VolatilityRegime.MODERATE
        elif vol < 0.50:  # 50%
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.VERY_HIGH

    def _analyze_volatility_trend(
        self,
        metrics: VolatilityMetrics,
        historical_metrics: List[Dict[str, Any]]
    ) -> VolatilityTrend:
        """Analyze trend in volatility"""
        if not historical_metrics:
            return VolatilityTrend.STABLE
            
        # Calculate recent volatility changes
        recent_vols = [m['historical_vol'] for m in historical_metrics[-20:]]
        current_vol = metrics.historical_vol
        
        if not recent_vols:
            return VolatilityTrend.STABLE
            
        avg_vol = np.mean(recent_vols)
        std_vol = np.std(recent_vols)
        
        # Check for regime change
        if abs(current_vol - avg_vol) > 2 * std_vol:
            return VolatilityTrend.REGIME_CHANGE
            
        # Determine trend
        vol_change = (current_vol - avg_vol) / avg_vol
        if vol_change > 0.1:  # 10% increase
            return VolatilityTrend.INCREASING
        elif vol_change < -0.1:  # 10% decrease
            return VolatilityTrend.DECREASING
        else:
            return VolatilityTrend.STABLE

    def _calculate_regime_stability(
        self,
        current_vol: float,
        historical_metrics: List[Dict[str, Any]]
    ) -> float:
        """Calculate stability of current volatility regime"""
        if not historical_metrics:
            return 0.5
            
        recent_vols = [m['historical_vol'] for m in historical_metrics[-60:]]  # Last 60 periods
        if not recent_vols:
            return 0.5
            
        # Calculate how often volatility stayed within current regime bounds
        current_regime = self._determine_volatility_regime(current_vol)
        regime_vols = [
            vol for vol in recent_vols
            if self._determine_volatility_regime(vol) == current_regime
        ]
        
        stability = len(regime_vols) / len(recent_vols)
        return stability

    def _identify_regime_characteristics(
        self,
        metrics: VolatilityMetrics,
        regime: VolatilityRegime
    ) -> Dict[str, Any]:
        """Identify characteristics of current volatility regime"""
        characteristics = {
            'mean_reversion_speed': self._calculate_mean_reversion_speed(metrics),
            'volatility_clustering': self._calculate_volatility_clustering(metrics),
            'term_structure_slope': self._calculate_term_structure_slope(metrics),
            'skew_characteristics': self._analyze_skew_characteristics(metrics),
            'typical_duration': self._estimate_regime_duration(regime)
        }
        
        return characteristics

    def _calculate_mean_reversion_speed(
        self,
        metrics: VolatilityMetrics
    ) -> float:
        """Calculate speed of mean reversion in volatility"""
        # Simple AR(1) coefficient as mean reversion speed
        if not hasattr(metrics, 'historical_vol_series'):
            return 0.0
            
        series = metrics.historical_vol_series
        if len(series) < 2:
            return 0.0
            
        changes = np.diff(series)
        levels = series[:-1]
        
        # Regression of changes on levels
        beta = np.cov(changes, levels)[0,1] / np.var(levels)
        return -beta  # Negative beta indicates mean reversion speed

    def _calculate_volatility_clustering(
        self,
        metrics: VolatilityMetrics
    ) -> float:
        """Calculate degree of volatility clustering"""
        if not hasattr(metrics, 'returns'):
            return 0.0
            
        # Use autocorrelation of absolute returns as clustering measure
        abs_returns = np.abs(metrics.returns)
        if len(abs_returns) < 2:
            return 0.0
            
        autocorr = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0,1]
        return autocorr

    def _calculate_term_structure_slope(
        self,
        metrics: VolatilityMetrics
    ) -> float:
        """Calculate slope of volatility term structure"""
        term_structure = metrics.term_structure
        if not term_structure:
            return 0.0
            
        tenors = sorted([float(t) for t in term_structure.keys()])
        if len(tenors) < 2:
            return 0.0
            
        vols = [term_structure[str(t)] for t in tenors]
        
        # Calculate slope using linear regression
        coeffs = np.polyfit(tenors, vols, 1)
        return coeffs[0]  # Return slope

    def _analyze_skew_characteristics(
        self,
        metrics: VolatilityMetrics
    ) -> Dict[str, Any]:
        """Analyze characteristics of volatility skew"""
        return {
            'skew_level': metrics.skew,
            'skew_stability': self._calculate_skew_stability(metrics),
            'put_call_skew_ratio': self._calculate_put_call_skew_ratio(metrics)
        }

    def _estimate_regime_duration(
        self,
        regime: VolatilityRegime
    ) -> Dict[str, float]:
        """Estimate typical duration of volatility regime"""
        # Historical average durations (in trading days)
        typical_durations = {
            VolatilityRegime.VERY_LOW: 20,
            VolatilityRegime.LOW: 40,
            VolatilityRegime.MODERATE: 60,
            VolatilityRegime.HIGH: 30,
            VolatilityRegime.VERY_HIGH: 15
        }
        
        return {
            'typical_duration': typical_durations[regime],
            'confidence': 0.7  # Fixed confidence in duration estimate
        }

    async def _generate_volatility_cone(
        self,
        returns: List[float],
        timestamps: List[datetime]
    ) -> VolatilityCone:
        """Generate volatility cone analysis"""
        if len(returns) < self.min_data_points:
            raise AnalysisError("Insufficient data for volatility cone analysis")
            
        # Define lookback periods
        periods = [5, 10, 21, 42, 63, 126, 252]
        percentiles = [5, 25, 50, 75, 95]
        
        cone_data = {
            'percentiles': {},
            'current_vol': {},
            'zscore': {}
        }
        
        for period in periods:
            if len(returns) < period:
                continue
                
            # Calculate rolling volatilities
            rolling_vols = []
            for i in range(len(returns) - period + 1):
                window_returns = returns[i:i+period]
                vol = np.std(window_returns) * np.sqrt(252)
                rolling_vols.append(vol)
                
            if rolling_vols:
                # Calculate percentiles
                cone_data['percentiles'][str(period)] = [
                    np.percentile(rolling_vols, p)
                    for p in percentiles
                ]
                
                # Current volatility
                current_vol = np.std(returns[-period:]) * np.sqrt(252)
                cone_data['current_vol'][str(period)] = current_vol
                
                # Z-score
                mean_vol = np.mean(rolling_vols)
                std_vol = np.std(rolling_vols)
                if std_vol > 0:
                    zscore = (current_vol - mean_vol) / std_vol
                    cone_data['zscore'][str(period)] = zscore
                else:
                    cone_data['zscore'][str(period)] = 0.0
                    
        return VolatilityCone(
            percentiles=cone_data['percentiles'],
            current_vol=cone_data['current_vol'],
            zscore=cone_data['zscore']
        )

    def _generate_volatility_analysis(
        self,
        metrics: VolatilityMetrics,
        regime_analysis: Dict[str, Any],
        vol_cone: VolatilityCone
    ) -> str:
        """Generate detailed volatility analysis"""
        analysis_parts = []
        
        # Overview
        analysis_parts.append("## Volatility Analysis\n")
        analysis_parts.append(
            f"Current Regime: {regime_analysis['regime'].value}\n"
            f"Trend: {regime_analysis['trend'].value}\n"
            f"Regime Stability: {regime_analysis['stability']:.2f}\n"
        )
        
        # Core Metrics
        analysis_parts.append("\n## Volatility Metrics\n")
        analysis_parts.append(
            f"Historical Volatility: {metrics.historical_vol:.2%}\n"
            f"Realized Volatility: {metrics.realized_vol:.2%}\n"
            f"Implied Volatility (ATM): {metrics.implied_vol_atm:.2%}\n"
            f"Forward Volatility: {metrics.forward_vol:.2%}\n"
            f"Volatility of Volatility: {metrics.vol_of_vol:.2%}\n"
        )
        
        # Term Structure Analysis
        if metrics.term_structure:
            analysis_parts.append("\n## Term Structure Analysis\n")
            for tenor, vol in sorted(metrics.term_structure.items()):
                analysis_parts.append(f"Tenor {tenor}: {vol:.2%}\n")
        
        # Volatility Cone Analysis
        analysis_parts.append("\n## Volatility Cone Analysis\n")
        for period, current_vol in vol_cone.current_vol.items():
            percentiles = vol_cone.percentiles[period]
            zscore = vol_cone.zscore[period]
            analysis_parts.append(
                f"Period {period}:\n"
                f"  Current: {current_vol:.2%}\n"
                f"  Z-Score: {zscore:.2f}\n"
                f"  Percentiles (5,25,50,75,95): "
                f"{[f'{p:.2%}' for p in percentiles]}\n"
            )
        
        # Regime Characteristics
        analysis_parts.append("\n## Regime Characteristics\n")
        for name, value in regime_analysis['characteristics'].items():
            if isinstance(value, dict):
                analysis_parts.append(f"{name}:\n")
                for k, v in value.items():
                    analysis_parts.append(f"  {k}: {v}\n")
            else:
                analysis_parts.append(f"{name}: {value}\n")
        
        return "\n".join(analysis_parts)

    def _generate_volatility_recommendations(
        self,
        metrics: VolatilityMetrics,
        regime_analysis: Dict[str, Any],
        vol_cone: VolatilityCone
    ) -> List[str]:
        """Generate volatility-based recommendations"""
        recommendations = []
        
        # Regime-based recommendations
        regime = regime_analysis['regime']
        trend = regime_analysis['trend']
        stability = regime_analysis['stability']
        
        if regime in [VolatilityRegime.HIGH, VolatilityRegime.VERY_HIGH]:
            recommendations.extend([
                "Consider reducing position sizes due to elevated volatility",
                "Implement tighter stop-loss levels",
                "Consider volatility hedging strategies",
                "Review and adjust risk limits",
                "Consider options-based protection strategies"
            ])
            
        elif regime in [VolatilityRegime.VERY_LOW, VolatilityRegime.LOW]:
            recommendations.extend([
                "Consider increasing position sizes given low volatility",
                "Look for opportunities to sell expensive volatility",
                "Review carry trade opportunities",
                "Consider yield enhancement strategies"
            ])
            
        # Trend-based recommendations
        if trend == VolatilityTrend.INCREASING:
            recommendations.extend([
                "Review risk management practices",
                "Consider implementing volatility-based stops",
                "Prepare for potential regime shift",
                "Review option positions for vega exposure"
            ])
        elif trend == VolatilityTrend.REGIME_CHANGE:
            recommendations.extend([
                "Immediate review of all positions required",
                "Consider defensive positioning",
                "Review correlation assumptions",
                "Assess portfolio stress test scenarios"
            ])
            
        # Term structure recommendations
        slope = self._calculate_term_structure_slope(metrics)
        if slope > 0.05:
            recommendations.extend([
                "Term structure in contango - consider calendar spreads",
                "Evaluate short-dated vs long-dated option positions",
                "Review volatility roll-down strategies"
            ])
        elif slope < -0.05:
            recommendations.extend([
                "Term structure in backwardation - review hedging costs",
                "Consider extending option protection duration",
                "Evaluate opportunities in volatility curve steepener positions"
            ])
            
        # Skew-based recommendations
        if abs(metrics.skew) > 0.1:
            recommendations.extend([
                "Significant volatility skew - examine tail risk hedging",
                "Review risk reversal strategies",
                "Consider put spread collar strategies"
            ])
            
        # Cone-based recommendations
        for period, zscore in vol_cone.zscore.items():
            if zscore > 2:
                recommendations.extend([
                    f"Volatility extremely high for {period}-day window",
                    f"Consider mean reversion strategies for {period}-day horizon",
                    f"Review short volatility opportunities in {period}-day expiries"
                ])
            elif zscore < -2:
                recommendations.extend([
                    f"Volatility extremely low for {period}-day window",
                    f"Consider long volatility positions for {period}-day horizon",
                    f"Review tail risk protection for {period}-day expiries"
                ])
                
        return recommendations

    def _generate_volatility_summary(
        self,
        metrics: VolatilityMetrics,
        regime_analysis: Dict[str, Any]
    ) -> str:
        """Generate concise volatility summary"""
        summary_parts = []
        
        # Overall status
        summary_parts.append(
            f"Volatility Regime: {regime_analysis['regime'].value} "
            f"(Trend: {regime_analysis['trend'].value})"
        )
        
        # Key metrics
        summary_parts.append(
            f"\nKey Metrics:"
            f"\n- Historical Vol: {metrics.historical_vol:.1%}"
            f"\n- Implied Vol (ATM): {metrics.implied_vol_atm:.1%}"
            f"\n- Forward Vol: {metrics.forward_vol:.1%}"
        )
        
        # Risk signals
        risk_signals = []
        if metrics.skew > 0.1:
            risk_signals.append("Elevated put skew")
        if regime_analysis['trend'] == VolatilityTrend.REGIME_CHANGE:
            risk_signals.append("Regime transition in progress")
        if metrics.vol_of_vol > 0.1:
            risk_signals.append("High volatility of volatility")
            
        if risk_signals:
            summary_parts.append("\nRisk Signals:")
            for signal in risk_signals:
                summary_parts.append(f"- {signal}")
                
        # Term structure
        if metrics.term_structure:
            slope = self._calculate_term_structure_slope(metrics)
            summary_parts.append(
                f"\nTerm Structure: "
                f"{'Contango' if slope > 0 else 'Backwardation'} "
                f"({slope:.1%})"
            )
            
        return "\n".join(summary_parts)

    def _calculate_confidence(
        self,
        metrics: VolatilityMetrics,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence in volatility analysis"""
        confidence_factors = []
        
        # Data quality confidence
        data_points = len(market_data.get('returns', []))
        data_confidence = min(1.0, data_points / self.min_data_points)
        confidence_factors.append(data_confidence)
        
        # Market regime confidence
        regime_confidence = self._calculate_regime_confidence(
            metrics,
            market_data.get('historical_metrics', [])
        )
        confidence_factors.append(regime_confidence)
        
        # Metric stability confidence
        stability_confidence = self._calculate_metric_stability(metrics)
        confidence_factors.append(stability_confidence)
        
        return np.mean(confidence_factors)

    def _calculate_regime_confidence(
        self,
        metrics: VolatilityMetrics,
        historical_metrics: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in regime determination"""
        if not historical_metrics:
            return 0.5
            
        # Calculate how stable recent volatility has been
        recent_vols = [m['historical_vol'] for m in historical_metrics[-20:]]
        if not recent_vols:
            return 0.5
            
        current_vol = metrics.historical_vol
        vol_std = np.std(recent_vols)
        
        if vol_std == 0:
            return 1.0
            
        # Calculate z-score of current vol
        z_score = abs(current_vol - np.mean(recent_vols)) / vol_std
        
        # Higher z-score means less confidence
        confidence = max(0.0, 1.0 - (z_score / 4.0))
        return confidence

    def _calculate_metric_stability(
        self,
        metrics: VolatilityMetrics
    ) -> float:
        """Calculate stability of volatility metrics"""
        stability_scores = []
        
        # Compare different volatility measures
        vols = [
            metrics.historical_vol,
            metrics.realized_vol,
            metrics.implied_vol_atm,
            metrics.forward_vol
        ]
        
        vols = [v for v in vols if v > 0]
        if len(vols) >= 2:
            vol_cv = np.std(vols) / np.mean(vols)
            stability_scores.append(max(0.0, 1.0 - vol_cv))
            
        # Term structure stability
        if metrics.term_structure:
            term_vols = list(metrics.term_structure.values())
            term_cv = np.std(term_vols) / np.mean(term_vols)
            stability_scores.append(max(0.0, 1.0 - term_cv))
            
        # Surface stability
        if metrics.implied_vol_surface:
            surface_vols = list(metrics.implied_vol_surface.values())
            surface_cv = np.std(surface_vols) / np.mean(surface_vols)
            stability_scores.append(max(0.0, 1.0 - surface_cv))
            
        return np.mean(stability_scores) if stability_scores else 0.5

    async def get_real_time_metrics(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate real-time volatility metrics"""
        try:
            metrics = await self._calculate_volatility_metrics(market_data, None)
            
            return {
                'current_metrics': {
                    'historical_vol': metrics.historical_vol,
                    'realized_vol': metrics.realized_vol,
                    'forward_vol': metrics.forward_vol,
                    'vol_of_vol': metrics.vol_of_vol
                },
                'status': {
                    'regime': self._determine_volatility_regime(
                        metrics.historical_vol
                    ).value,
                    'trend': self._analyze_volatility_trend(
                        metrics,
                        market_data.get('historical_metrics', [])
                    ).value
                },
                'thresholds': {
                    'high_vol_threshold': 0.3,
                    'low_vol_threshold': 0.1,
                    'regime_change_threshold': 0.5
                }
            }
            
        except Exception as e:
            raise AnalysisError(f"Real-time metrics calculation failed: {str(e)}")

    async def simulate_volatility_shock(
        self,
        market_data: Dict[str, Any],
        shock_size: float = 0.5  # 50% increase in volatility
    ) -> Dict[str, Any]:
        """Simulate impact of volatility shock"""
        try:
            # Calculate pre-shock metrics
            initial_metrics = await self._calculate_volatility_metrics(
                market_data,
                None
            )
            
            # Apply shock to market data
            shocked_data = market_data.copy()
            shocked_data['returns'] = [
                r * (1 + shock_size) for r in market_data['returns']
            ]
            
            # Calculate post-shock metrics
            shocked_metrics = await self._calculate_volatility_metrics(
                shocked_data,
                None
            )
            
            return {
                'pre_shock': {
                    'historical_vol': initial_metrics.historical_vol,
                    'regime': self._determine_volatility_regime(
                        initial_metrics.historical_vol
                    ).value
                },
                'post_shock': {
                    'historical_vol': shocked_metrics.historical_vol,
                    'regime': self._determine_volatility_regime(
                        shocked_metrics.historical_vol
                    ).value
                },
                'impact': {
                    'vol_change': shocked_metrics.historical_vol - initial_metrics.historical_vol,
                    'regime_change': shocked_metrics.historical_vol / initial_metrics.historical_vol - 1,
                    'risk_implications': self._assess_shock_implications(
                        initial_metrics,
                        shocked_metrics
                    )
                }
            }
            
        except Exception as e:
            raise AnalysisError(f"Volatility shock simulation failed: {str(e)}")

    def _assess_shock_implications(
        self,
        initial_metrics: VolatilityMetrics,
        shocked_metrics: VolatilityMetrics
    ) -> List[str]:
        """Assess implications of volatility shock"""
        implications = []
        
        # Regime change implications
        initial_regime = self._determine_volatility_regime(initial_metrics.historical_vol)
        shocked_regime = self._determine_volatility_regime(shocked_metrics.historical_vol)
        
        if initial_regime != shocked_regime:
            implications.append(
                f"Regime shift from {initial_regime.value} to {shocked_regime.value}"
            )
            
        # Risk metric implications
        if shocked_metrics.vol_of_vol > initial_metrics.vol_of_vol * 1.5:
            implications.append("Significant increase in volatility uncertainty")
            
        if shocked_metrics.forward_vol > initial_metrics.forward_vol * 1.3:
            implications.append("Elevated forward volatility expectations")
            
        # Market impact implications
        if hasattr(shocked_metrics, 'market_impact'):
            if shocked_metrics.market_impact > initial_metrics.market_impact * 2:
                implications.append("Potential liquidity deterioration")
                
        return implications
