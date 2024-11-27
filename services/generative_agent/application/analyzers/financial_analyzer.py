 services/generative_agent/application/analyzers/financial_analyzer.py

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import torch
import numpy as np
from dataclasses import dataclass
from scipy import stats
from scipy.stats import linregress

from ...domain.models import AnalysisResponse, AnalysisType, ConfidenceLevel
from ...domain.exceptions import AnalysisError
from ..models.hamiltonian_model import HamiltonianNN, HNNConfig
from .mcts_analyzer import MarketMCTS, MCTSAnalysis, MarketState

@dataclass
class FinancialMetrics:
    """Core financial metrics used in analysis"""
    price_metrics: Dict[str, float]
    volume_metrics: Dict[str, float]
    momentum_metrics: Dict[str, float]
    conservation_metrics: Dict[str, float]
    derived_metrics: Dict[str, float]

class FinancialAnalyzer:
    """
    Financial analyzer integrating:
    - Hamiltonian Neural Network for market dynamics
    - Monte Carlo Tree Search for scenario exploration
    - Conservation law analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.service.hamiltonian.model.device)
        
        # Initialize models
        self.hnn_model = self._initialize_model()
        self.mcts = MarketMCTS(self.hnn_model, config.service.hamiltonian.mcts)
        
        # Setup GPU optimization if available
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream(device=self.device)
            torch.backends.cudnn.benchmark = config.service.hamiltonian.model.cuda_optimization.cudnn_benchmark

    def _initialize_model(self) -> HamiltonianNN:
        """Initialize GPU-optimized Hamiltonian model"""
        model_config = HNNConfig(**self.config.service.hamiltonian.model)
        model = HamiltonianNN(model_config)
        
        if self.config.service.hamiltonian.model.get('load_checkpoint', True):
            checkpoint = torch.load(
                self.config.service.integrations.hamiltonian.model_path,
                map_location=self.device
            )
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        return model

    async def analyze(
        self,
        market_data: Dict[str, Any]
    ) -> AnalysisResponse:
        """
        Analyze market data using combined HNN and MCTS approach
        """
        try:
            # Prepare data for analysis
            phase_space_data = self._prepare_phase_space(market_data)
            initial_state = self._create_market_state(market_data)
            
            # Perform HNN analysis
            hnn_predictions, conservation_metrics = await self._hnn_analysis(
                phase_space_data
            )
            
            # Perform MCTS analysis if enabled
            mcts_analysis = None
            if self.config.service.hamiltonian.mcts.enabled:
                mcts_analysis = await self.mcts.analyze(initial_state)
            
            # Calculate financial metrics
            financial_metrics = self._calculate_financial_metrics(
                market_data,
                hnn_predictions,
                conservation_metrics
            )
            
            # Generate analysis and recommendations
            return self._generate_analysis_response(
                market_data,
                hnn_predictions,
                conservation_metrics,
                financial_metrics,
                mcts_analysis
            )

        except Exception as e:
            raise AnalysisError(f"Financial analysis failed: {str(e)}")

    async def _hnn_analysis(
        self,
        phase_space_data: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform Hamiltonian Neural Network analysis"""
        try:
            with torch.cuda.stream(self.stream):
                with torch.cuda.amp.autocast(
                    enabled=self.config.service.hamiltonian.model.cuda_optimization.enable_amp
                ):
                    with torch.no_grad():
                        predictions, hamiltonian = self.hnn_model(
                            phase_space_data.to(
                                self.device,
                                non_blocking=True
                            )
                        )
                
                torch.cuda.current_stream().synchronize()
            
            conservation_metrics = self._calculate_conservation_metrics(
                phase_space_data,
                predictions,
                hamiltonian
            )
            
            return predictions, conservation_metrics

        except Exception as e:
            raise AnalysisError(f"HNN analysis failed: {str(e)}")

    def _prepare_phase_space(
        self,
        market_data: Dict[str, Any]
    ) -> torch.Tensor:
        """Transform market data into phase space representation"""
        # Position variables (q)
        positions = torch.tensor([
            market_data['prices'],
            market_data['volumes'],
            market_data['market_caps']
        ], dtype=torch.float32, pin_memory=True)
        
        # Momentum variables (p)
        momentums = torch.tensor([
            market_data['price_changes'],
            market_data['volume_changes'],
            market_data['flow_momentum']
        ], dtype=torch.float32, pin_memory=True)
        
        return torch.cat([positions, momentums], dim=-1)

    def _create_market_state(
        self,
        market_data: Dict[str, Any]
    ) -> MarketState:
        """Create market state for MCTS"""
        return MarketState(
            prices=np.array(market_data['prices']),
            volumes=np.array(market_data['volumes']),
            timestamp=market_data['timestamp'],
            metadata=market_data.get('metadata', {}),
            phase_space=self._prepare_phase_space(market_data)
        )

    def _calculate_conservation_metrics(
        self,
        initial_state: torch.Tensor,
        predictions: torch.Tensor,
        hamiltonian: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate conservation law metrics"""
        energy_conservation = float(torch.mean(
            (hamiltonian[1:] - hamiltonian[:-1]).abs()
        ))
        
        momentum_conservation = float(torch.mean(
            (predictions[:, 3:] - initial_state[:, 3:]).abs()
        ))
        
        phase_space_volume = float(torch.det(
            torch.matmul(
                predictions.T,
                initial_state
            )
        ))
        
        # Calculate Lyapunov exponent
        lyapunov = self._calculate_lyapunov_exponent(
            initial_state.cpu().numpy(),
            predictions.cpu().numpy()
        )
        
        return {
            'energy_conservation': energy_conservation,
            'momentum_conservation': momentum_conservation,
            'phase_space_volume': phase_space_volume,
            'lyapunov_exponent': lyapunov
        }

    def _calculate_lyapunov_exponent(
        self,
        initial_state: np.ndarray,
        predictions: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """Calculate largest Lyapunov exponent"""
        distances = np.linalg.norm(predictions - initial_state, axis=1)
        non_zero_distances = distances[distances > epsilon]
        if len(non_zero_distances) < 2:
            return 0.0
        
        time_steps = np.arange(len(non_zero_distances))
        log_distances = np.log(non_zero_distances)
        
        slope, _, _, _, _ = linregress(time_steps, log_distances)
        return float(slope)

    def _calculate_financial_metrics(
        self,
        market_data: Dict[str, Any],
        predictions: torch.Tensor,
        conservation_metrics: Dict[str, float]
    ) -> FinancialMetrics:
        """Calculate comprehensive financial metrics"""
        predictions_np = predictions.cpu().numpy()
        
        return FinancialMetrics(
            price_metrics=self._calculate_price_metrics(
                np.array(market_data['prices']),
                predictions_np[:, 0]
            ),
            volume_metrics=self._calculate_volume_metrics(
                np.array(market_data['volumes']),
                predictions_np[:, 1]
            ),
            momentum_metrics=self._calculate_momentum_metrics(
                predictions_np[:, 3:]
            ),
            conservation_metrics=conservation_metrics,
            derived_metrics=self._calculate_derived_metrics(
                market_data,
                predictions_np
            )
        )

    def _calculate_price_metrics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> Dict[str, float]:
        """Calculate price-related metrics"""
        # Basic accuracy metrics
        mae = float(np.mean(np.abs(predicted - actual)))
        mape = float(np.mean(np.abs((predicted - actual) / actual))) * 100
        
        # Direction accuracy
        actual_direction = np.sign(np.diff(actual))
        predicted_direction = np.sign(np.diff(predicted))
        direction_accuracy = float(np.mean(
            actual_direction == predicted_direction
        ))
        
        # Volatility metrics
        actual_volatility = float(np.std(actual))
        predicted_volatility = float(np.std(predicted))
        volatility_error = abs(predicted_volatility - actual_volatility)
        
        return {
            'mae': mae,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'volatility_error': volatility_error,
            'actual_volatility': actual_volatility,
            'predicted_volatility': predicted_volatility
        }

    def _calculate_volume_metrics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> Dict[str, float]:
        """Calculate volume-related metrics"""
        # Basic volume metrics
        volume_mae = float(np.mean(np.abs(predicted - actual)))
        volume_mape = float(np.mean(np.abs((predicted - actual) / actual))) * 100
        
        # Volume trend accuracy
        actual_trend = np.sign(np.diff(actual))
        predicted_trend = np.sign(np.diff(predicted))
        trend_accuracy = float(np.mean(actual_trend == predicted_trend))
        
        # Volume profile metrics
        actual_profile = actual / np.sum(actual)
        predicted_profile = predicted / np.sum(predicted)
        profile_error = float(np.mean(np.abs(actual_profile - predicted_profile)))
        
        return {
            'volume_mae': volume_mae,
            'volume_mape': volume_mape,
            'trend_accuracy': trend_accuracy,
            'profile_error': profile_error
        }

    def _calculate_momentum_metrics(
        self,
        momentum: np.ndarray
    ) -> Dict[str, float]:
        """Calculate momentum-related metrics"""
        # Momentum magnitude
        magnitude = float(np.mean(np.abs(momentum)))
        
        # Momentum stability
        stability = float(np.std(momentum))
        
        # Momentum autocorrelation
        autocorr = float(np.correlate(momentum[:, 0], momentum[:, 0])[0])
        
        # Momentum distribution metrics
        skew = float(stats.skew(momentum.flatten()))
        kurtosis = float(stats.kurtosis(momentum.flatten()))
        
        return {
            'magnitude': magnitude,
            'stability': stability,
            'autocorrelation': autocorr,
            'skewness': skew,
            'kurtosis': kurtosis
        }

    def _calculate_derived_metrics(
        self,
        market_data: Dict[str, Any],
        predictions: np.ndarray
    ) -> Dict[str, float]:
        """Calculate derived financial metrics"""
        # Trend strength
        trend_strength = float(np.mean(np.abs(np.diff(predictions[:, 0]))))
        
        # Market efficiency
        efficiency = self._calculate_market_efficiency(predictions)
        
        # Phase space metrics
        phase_space_density = float(np.mean(
            np.linalg.norm(predictions, axis=1)
        ))
        
        # Stability metrics
        stability_score = self._calculate_stability_score(predictions)
        
        return {
            'trend_strength': trend_strength,
            'market_efficiency': efficiency,
            'phase_space_density': phase_space_density,
            'stability_score': stability_score
        }
      def _calculate_market_efficiency(self, predictions: np.ndarray) -> float:
        """
        Calculate market efficiency using multiple metrics:
        - Hurst exponent
        - Entropy
        - Autocorrelation decay
        """
        # Calculate Hurst exponent
        hurst = self._calculate_hurst_exponent(predictions[:, 0])
        
        # Calculate entropy
        entropy = self._calculate_entropy(predictions[:, 0])
        
        # Calculate autocorrelation decay
        autocorr_decay = self._calculate_autocorrelation_decay(predictions[:, 0])
        
        # Combine metrics into efficiency score
        efficiency_score = (
            0.4 * (1 - abs(hurst - 0.5)) +  # 0.5 indicates random walk
            0.3 * entropy +                  # Higher entropy = more efficient
            0.3 * autocorr_decay            # Faster decay = more efficient
        )
        
        return float(efficiency_score)

    def _calculate_hurst_exponent(
        self,
        prices: np.ndarray,
        lags: Optional[List[int]] = None
    ) -> float:
        """Calculate Hurst exponent for price series"""
        if lags is None:
            lags = [2, 4, 8, 16, 32]
            
        log_rs = []
        log_lags = []
        
        for lag in lags:
            # Calculate returns
            returns = np.diff(prices)
            
            # Calculate R/S statistic
            mean_return = np.mean(returns[:lag])
            std_return = np.std(returns[:lag])
            
            if std_return == 0:
                continue
                
            cumulative = np.cumsum(returns[:lag] - mean_return)
            r = max(cumulative) - min(cumulative)
            s = std_return
            
            if s > 0:
                rs = r/s
                log_rs.append(np.log(rs))
                log_lags.append(np.log(lag))
        
        if not log_rs:
            return 0.5
            
        # Regression slope is Hurst exponent
        slope, _, _, _, _ = linregress(log_lags, log_rs)
        return float(slope)

    def _calculate_entropy(
        self,
        prices: np.ndarray,
        bins: int = 50
    ) -> float:
        """Calculate normalized entropy of price distribution"""
        hist, _ = np.histogram(prices, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        entropy = -np.sum(hist * np.log(hist))
        max_entropy = np.log(bins)
        return float(entropy / max_entropy)

    def _calculate_autocorrelation_decay(
        self,
        prices: np.ndarray,
        max_lag: int = 20
    ) -> float:
        """Calculate autocorrelation decay rate"""
        returns = np.diff(prices)
        autocorr = [1.]  # Lag 0
        
        for lag in range(1, min(max_lag, len(returns))):
            corr = np.corrcoef(returns[lag:], returns[:-lag])[0, 1]
            autocorr.append(abs(corr))
            
        # Fit exponential decay
        x = np.arange(len(autocorr))
        y = np.log(np.maximum(autocorr, 1e-10))
        slope, _, _, _, _ = linregress(x, y)
        
        return float(-slope)  # Negative slope = decay rate

    def _calculate_stability_score(self, predictions: np.ndarray) -> float:
        """Calculate stability score based on predictions"""
        # Variance of predictions
        variance = np.var(predictions, axis=0)
        
        # Trend consistency
        diff = np.diff(predictions, axis=0)
        sign_changes = np.sum(diff[1:] * diff[:-1] < 0, axis=0)
        
        # Combine into stability score
        stability = 1.0 / (1.0 + np.mean(variance) + np.mean(sign_changes))
        return float(stability)

    def _generate_summary(
        self,
        predictions: np.ndarray,
        financial_metrics: FinancialMetrics,
        mcts_analysis: Optional[MCTSAnalysis]
    ) -> str:
        """Generate concise analysis summary"""
        summary_parts = []
        
        # Overall market state
        summary_parts.append("Market State Analysis:")
        summary_parts.append(f"- Price Prediction Accuracy: {financial_metrics.price_metrics['direction_accuracy']:.1%}")
        summary_parts.append(f"- Market Efficiency Score: {financial_metrics.derived_metrics['market_efficiency']:.2f}")
        summary_parts.append(f"- Stability Score: {financial_metrics.derived_metrics['stability_score']:.2f}")
        
        # Conservation law status
        energy_violation = financial_metrics.conservation_metrics['energy_conservation']
        if energy_violation > 0.1:
            summary_parts.append("\nConservation Law Violations Detected:")
            summary_parts.append(f"- Energy Conservation Error: {energy_violation:.3f}")
            summary_parts.append("- Market may be entering unstable regime")
        
        # MCTS scenario summary
        if mcts_analysis:
            summary_parts.append("\nScenario Analysis:")
            summary_parts.append(f"- Most Likely Scenario Probability: {mcts_analysis.probability:.1%}")
            summary_parts.append(f"- Number of Alternative Scenarios: {len(mcts_analysis.alternative_scenarios)}")
            
        # Key metrics
        summary_parts.append("\nKey Metrics:")
        summary_parts.append(f"- Volatility: {financial_metrics.price_metrics['predicted_volatility']:.3f}")
        summary_parts.append(f"- Momentum Magnitude: {financial_metrics.momentum_metrics['magnitude']:.3f}")
        summary_parts.append(f"- Volume Profile Error: {financial_metrics.volume_metrics['profile_error']:.3f}")
        
        return "\n".join(summary_parts)

    def _generate_detailed_analysis(
        self,
        predictions: np.ndarray,
        financial_metrics: FinancialMetrics,
        conservation_metrics: Dict[str, float],
        mcts_analysis: Optional[MCTSAnalysis],
        market_data: Dict[str, Any]
    ) -> str:
        """Generate comprehensive detailed analysis"""
        analysis_parts = []
        
        # Conservation Law Analysis
        analysis_parts.append("## Conservation Law Analysis")
        analysis_parts.append("Energy and Momentum Conservation:")
        analysis_parts.append(f"- Energy Conservation Error: {conservation_metrics['energy_conservation']:.3f}")
        analysis_parts.append(f"- Momentum Conservation Error: {conservation_metrics['momentum_conservation']:.3f}")
        analysis_parts.append(f"- Phase Space Volume: {conservation_metrics['phase_space_volume']:.3f}")
        analysis_parts.append(f"- Lyapunov Exponent: {conservation_metrics['lyapunov_exponent']:.3f}")
        
        # Phase Space Analysis
        analysis_parts.append("\n## Phase Space Analysis")
        analysis_parts.append("Market Dynamics:")
        analysis_parts.append(f"- Phase Space Density: {financial_metrics.derived_metrics['phase_space_density']:.3f}")
        analysis_parts.append(f"- Stability Score: {financial_metrics.derived_metrics['stability_score']:.3f}")
        analysis_parts.append(f"- Market Efficiency: {financial_metrics.derived_metrics['market_efficiency']:.3f}")
        
        # Price Dynamics
        analysis_parts.append("\n## Price Dynamics")
        analysis_parts.append("Price Prediction Metrics:")
        analysis_parts.append(f"- MAE: {financial_metrics.price_metrics['mae']:.3f}")
        analysis_parts.append(f"- MAPE: {financial_metrics.price_metrics['mape']:.1f}%")
        analysis_parts.append(f"- Direction Accuracy: {financial_metrics.price_metrics['direction_accuracy']:.1%}")
        
        # Volume Analysis
        analysis_parts.append("\n## Volume Analysis")
        analysis_parts.append("Volume Metrics:")
        analysis_parts.append(f"- Volume MAE: {financial_metrics.volume_metrics['volume_mae']:.3f}")
        analysis_parts.append(f"- Volume Trend Accuracy: {financial_metrics.volume_metrics['trend_accuracy']:.1%}")
        analysis_parts.append(f"- Profile Error: {financial_metrics.volume_metrics['profile_error']:.3f}")
        
        # Momentum Analysis
        analysis_parts.append("\n## Momentum Analysis")
        analysis_parts.append("Momentum Characteristics:")
        analysis_parts.append(f"- Magnitude: {financial_metrics.momentum_metrics['magnitude']:.3f}")
        analysis_parts.append(f"- Stability: {financial_metrics.momentum_metrics['stability']:.3f}")
        analysis_parts.append(f"- Autocorrelation: {financial_metrics.momentum_metrics['autocorrelation']:.3f}")
        
        # MCTS Scenario Analysis
        if mcts_analysis:
            analysis_parts.append("\n## Scenario Analysis")
            analysis_parts.append("Most Likely Scenario:")
            analysis_parts.append(f"- Probability: {mcts_analysis.probability:.1%}")
            analysis_parts.append(f"- Confidence: {mcts_analysis.confidence:.1%}")
            
            if mcts_analysis.alternative_scenarios:
                analysis_parts.append("\nAlternative Scenarios:")
                for i, scenario in enumerate(mcts_analysis.alternative_scenarios[:3], 1):
                    analysis_parts.append(f"Scenario {i}:")
                    analysis_parts.append(f"- Probability: {scenario['probability']:.1%}")
                    analysis_parts.append(f"- Key Characteristics: {scenario.get('description', 'N/A')}")
        
        return "\n".join(analysis_parts)

    def _generate_recommendations(
        self,
        financial_metrics: FinancialMetrics,
        mcts_analysis: Optional[MCTSAnalysis]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Conservation-based recommendations
        energy_violation = financial_metrics.conservation_metrics['energy_conservation']
        if energy_violation > 0.1:
            recommendations.append(
                "ALERT: Significant conservation law violations detected. "
                "Consider reducing position sizes and increasing hedging."
            )
            
        # Stability-based recommendations
        stability = financial_metrics.derived_metrics['stability_score']
        if stability < 0.3:
            recommendations.append(
                "Market showing signs of instability. "
                "Consider implementing strict stop-loss orders."
            )
            
        # Efficiency-based recommendations
        efficiency = financial_metrics.derived_metrics['market_efficiency']
        if efficiency < 0.4:
            recommendations.append(
                "Market efficiency is low. "
                "Look for arbitrage opportunities and pricing inefficiencies."
            )
        elif efficiency > 0.8:
            recommendations.append(
                "Market is highly efficient. "
                "Focus on low-cost, passive strategies."
            )
            
        # Momentum-based recommendations
        momentum_mag = financial_metrics.momentum_metrics['magnitude']
        if momentum_mag > 0.8:
            recommendations.append(
                "Strong momentum detected. "
                "Consider trend-following strategies."
            )
            
        # Volume-based recommendations
        volume_error = financial_metrics.volume_metrics['profile_error']
        if volume_error > 0.2:
            recommendations.append(
                "Abnormal volume patterns detected. "
                "Monitor liquidity conditions carefully."
            )
            
        # MCTS-based recommendations
        if mcts_analysis:
            if mcts_analysis.probability < 0.3:
                recommendations.append(
                    "High uncertainty in scenario analysis. "
                    "Consider reducing exposure and increasing diversification."
                )
            
            # Add specific scenario-based recommendations
            for scenario in mcts_analysis.alternative_scenarios[:2]:
                if scenario['probability'] > 0.2:
                    recommendations.append(
                        f"Prepare for alternative scenario: {scenario.get('description', '')}"
                    )
        
        return recommendations

