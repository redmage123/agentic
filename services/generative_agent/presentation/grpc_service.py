# services/generative_agent/presentation/grpc_servicer.py

import grpc
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from ..domain.models import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisType,
    MarketState,
    MCTSAnalysis,
)
from ..application.tca_service import GenerativeTCAService
from services.protos import generative_service_pb2 as pb2
from services.protos import generative_service_pb2_grpc as pb2_grpc


class GenerativeServicer(pb2_grpc.GenerativeServiceServicer):
    """
    gRPC servicer for the Generative Agent service.
    Handles all incoming analysis requests and integrates:
    - HNN-based analysis
    - MCTS scenario exploration
    - Market regime analysis
    - LLM-based analysis
    """

    def __init__(self, service: GenerativeTCAService):
        self.service = service

    async def Analyze(
        self, request: pb2.AnalysisRequest, context: grpc.aio.ServicerContext
    ) -> pb2.AnalysisResponse:
        """Handle market analysis requests"""
        try:
            # Convert gRPC request to domain model
            analysis_request = self._convert_request(request)

            # Process request through service
            analysis_result = await self.service.process_request(analysis_request)

            # Convert to gRPC response
            response = self._create_response(analysis_result)

            # Add MCTS scenarios if available
            if hasattr(analysis_result, "mcts_analysis"):
                self._add_mcts_data(response, analysis_result.mcts_analysis)

            return response

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Analysis failed: {str(e)}")
            raise

    async def GetStatus(
        self, request: pb2.StatusRequest, context: grpc.aio.ServicerContext
    ) -> pb2.StatusResponse:
        """Get service status"""
        try:
            # Get metrics and status
            metrics = await self.service.metrics_collector.get_metrics()
            agent_statuses = await self.service.agent_manager.list_agents()

            # Create response
            return pb2.StatusResponse(
                status="healthy",
                agent_statuses=[
                    self._convert_agent_status(agent) for agent in agent_statuses
                ],
                metrics=self._convert_metrics(metrics),
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Status check failed: {str(e)}")
            raise

    def _convert_request(self, request: pb2.AnalysisRequest) -> AnalysisRequest:
        """Convert gRPC request to domain model"""
        return AnalysisRequest(
            request_id=request.request_id or str(uuid.uuid4()),
            analysis_type=AnalysisType(request.analysis_type),
            market_data=self._parse_market_data(request.market_data),
            options_data=self._parse_options_data(request.options_data),
            metadata=dict(request.metadata),
        )

    def _create_response(self, result: AnalysisResponse) -> pb2.AnalysisResponse:
        """Convert domain response to gRPC response"""
        return pb2.AnalysisResponse(
            request_id=result.request_id,
            analysis_type=result.analysis_type.value,
            summary=result.summary,
            detailed_analysis=result.detailed_analysis,
            recommendations=result.recommendations,
            confidence_score=result.confidence_score,
            metadata=result.metadata,
            predictions=[self._convert_prediction(pred) for pred in result.predictions],
            processing_time=result.processing_time,
        )

    def _parse_market_data(self, market_data_proto: pb2.MarketData) -> Dict[str, Any]:
        """Parse market data from proto message"""
        return {
            "symbol": market_data_proto.symbol,
            "prices": list(market_data_proto.prices),
            "volumes": list(market_data_proto.volumes),
            "timestamps": [
                datetime.fromtimestamp(ts) for ts in market_data_proto.timestamps
            ],
            "indicators": dict(market_data_proto.indicators),
            "market_cap": market_data_proto.market_cap,
            "trading_data": {
                "high": list(market_data_proto.trading_data.high),
                "low": list(market_data_proto.trading_data.low),
                "open": list(market_data_proto.trading_data.open),
            },
        }

    def _parse_options_data(
        self, options_data_proto: Optional[pb2.OptionsData]
    ) -> Optional[Dict[str, Any]]:
        """Parse options data from proto message"""
        if not options_data_proto:
            return None

        return {
            "symbol": options_data_proto.symbol,
            "expiry": datetime.fromtimestamp(options_data_proto.expiry),
            "strike": options_data_proto.strike,
            "option_type": options_data_proto.option_type,
            "price": options_data_proto.price,
            "implied_vol": options_data_proto.implied_vol,
            "open_interest": options_data_proto.open_interest,
            "volume": options_data_proto.volume,
        }

    def _convert_prediction(self, prediction: Any) -> pb2.Prediction:
        """Convert prediction to proto message"""
        return pb2.Prediction(
            agent_id=prediction.agent_id,
            value=str(prediction.value),
            confidence=prediction.confidence,
            metadata=prediction.metadata,
        )

    def _add_mcts_data(
        self, response: pb2.AnalysisResponse, mcts_analysis: MCTSAnalysis
    ):
        """Add MCTS analysis data to response"""
        # Add most likely path
        response.most_likely_scenario.path.extend(
            [
                pb2.MarketState(
                    price=float(state.prices[-1]),
                    volume=float(state.volumes[-1]),
                    timestamp=state.timestamp.isoformat(),
                    metadata=state.metadata,
                )
                for state in mcts_analysis.most_likely_path
            ]
        )
        response.most_likely_scenario.probability = mcts_analysis.probability
        response.most_likely_scenario.confidence = mcts_analysis.confidence

        # Add alternative scenarios
        response.alternative_scenarios.extend(
            [
                pb2.Scenario(
                    path=[
                        pb2.MarketState(
                            price=float(state["price"]),
                            volume=float(state["volume"]),
                            timestamp=state["timestamp"].isoformat(),
                            metadata=state["metadata"],
                        )
                        for state in scenario["path"]
                    ],
                    probability=scenario["probability"],
                    confidence=scenario["confidence"],
                )
                for scenario in mcts_analysis.alternative_scenarios
            ]
        )

    def _convert_agent_status(self, agent: Any) -> pb2.AgentStatus:
        """Convert agent status to proto message"""
        return pb2.AgentStatus(
            agent_id=agent.agent_id,
            status=agent.status.value,
            load=agent.load,
            last_check=(
                int(agent.last_health_check.timestamp())
                if agent.last_health_check
                else 0
            ),
            metrics=agent.metrics,
        )

    def _convert_metrics(self, metrics: Dict[str, Any]) -> pb2.Metrics:
        """Convert metrics to proto message"""
        return pb2.Metrics(
            requests_total=metrics.get("requests_total", 0),
            requests_successful=metrics.get("requests_successful", 0),
            average_latency=metrics.get("average_latency", 0.0),
            error_rate=metrics.get("error_rate", 0.0),
            prediction_metrics=metrics.get("prediction_metrics", {}),
            system_metrics=metrics.get("system_metrics", {}),
        )
