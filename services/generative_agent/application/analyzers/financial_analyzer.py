# services/generative_agent/application/analyzers/financial_analyzer.py

from typing import List, Dict, Any, Tuple
from datetime import datetime
import asyncio

from ...domain.models import AnalysisResponse, AnalysisType
from ...domain.interfaces import NNClient


class FinancialAnalyzer:
    """
    Analyzes financial data using multiple neural networks and LLM responses
    """

    def __init__(
        self,
        hamiltonian_client: NNClient,
        fourier_client: NNClient,
        perturbation_client: NNClient,
        confidence_threshold: float = 0.7,
    ):
        self.hamiltonian_client = hamiltonian_client
        self.fourier_client = fourier_client
        self.perturbation_client = perturbation_client
        self.confidence_threshold = confidence_threshold

    async def analyze(
        self,
        market_data: Dict[str, Any],
        llm_responses: List[str],
        metadata: Dict[str, Any],
    ) -> AnalysisResponse:
        """
        Process financial analysis using NNs and LLM responses

        The neural networks are used as follows:
        - Hamiltonian NN: Identifies conservation laws and stable states in market dynamics
        - Fourier NN: Detects cyclical patterns and frequency components
        - Perturbation NN: Identifies regime changes and market state transitions
        """
        # Get predictions from all neural networks concurrently
        nn_results = await asyncio.gather(
            self._get_hamiltonian_analysis(market_data),
            self._get_fourier_analysis(market_data),
            self._get_perturbation_analysis(market_data),
        )

        hamiltonian_result, fourier_result, perturbation_result = nn_results

        # Combine NN insights with LLM analysis
        factors = await self._combine_factors(
            hamiltonian_result, fourier_result, perturbation_result, llm_responses
        )

        risks = await self._combine_risks(
            hamiltonian_result, fourier_result, perturbation_result, llm_responses
        )

        opportunities = await self._combine_opportunities(
            hamiltonian_result, fourier_result, perturbation_result, llm_responses
        )

        # Calculate combined confidence score
        confidence = self._calculate_combined_confidence(
            hamiltonian_result, fourier_result, perturbation_result, factors, risks
        )

        # Generate detailed analysis incorporating all sources
        detailed_analysis = self._generate_comprehensive_analysis(
            hamiltonian_result,
            fourier_result,
            perturbation_result,
            factors,
            risks,
            opportunities,
            llm_responses,
        )

        # Generate recommendations based on all sources
        recommendations = self._generate_combined_recommendations(
            hamiltonian_result,
            fourier_result,
            perturbation_result,
            factors,
            risks,
            opportunities,
            confidence,
        )

        return AnalysisResponse(
            request_id=metadata.get("request_id", ""),
            analysis_type=AnalysisType.FINANCIAL,
            summary=self._generate_summary(nn_results, factors, risks, opportunities),
            detailed_analysis=detailed_analysis,
            recommendations=recommendations,
            confidence_score=confidence,
            metadata={
                **metadata,
                "hamiltonian_confidence": hamiltonian_result["confidence"],
                "fourier_confidence": fourier_result["confidence"],
                "perturbation_confidence": perturbation_result["confidence"],
                "factor_count": len(factors),
                "risk_count": len(risks),
                "opportunity_count": len(opportunities),
            },
            processing_time=metadata.get("processing_time", 0),
            timestamp=datetime.now(),
        )

    async def _get_hamiltonian_analysis(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get Hamiltonian NN analysis focusing on market dynamics and conservation laws
        """
        response = await self.hamiltonian_client.analyze(market_data)
        return {
            "conservation_laws": response.get("laws", []),
            "stable_states": response.get("states", []),
            "energy_levels": response.get("energy", {}),
            "momentum_factors": response.get("momentum", {}),
            "confidence": response.get("confidence", 0.0),
        }

    async def _get_fourier_analysis(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get Fourier NN analysis focusing on cyclical patterns
        """
        response = await self.fourier_client.analyze(market_data)
        return {
            "cycles": response.get("cycles", []),
            "frequencies": response.get("frequencies", []),
            "amplitudes": response.get("amplitudes", {}),
            "phases": response.get("phases", {}),
            "confidence": response.get("confidence", 0.0),
        }

    async def _get_perturbation_analysis(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get Perturbation NN analysis focusing on regime changes
        """
        response = await self.perturbation_client.analyze(market_data)
        return {
            "regimes": response.get("regimes", []),
            "transitions": response.get("transitions", []),
            "stability": response.get("stability", {}),
            "anomalies": response.get("anomalies", []),
            "confidence": response.get("confidence", 0.0),
        }
