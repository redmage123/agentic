# services/generative_agent/presentation/grpc_servicer.py

import grpc
from typing import List, Dict, Any
from datetime import datetime

from ..domain.models import (
    AnalysisRequest, AnalysisResponse, AnalysisType,
    PredictionRequest, AggregatedPrediction
)
from ..application.tca_service import GenerativeTCAService
from services.protos import generative_service_pb2 as pb2
from services.protos import generative_service_pb2_grpc as pb2_grpc

class GenerativeServicer(pb2_grpc.GenerativeServiceServicer):
    """
    gRPC servicer for the integrated Generative Agent + TCA service.
    Handles requests from the Flask backend.
    """
    
    def __init__(self, service: GenerativeTCAService):
        self.service = service

    async def Analyze(
        self,
        request: pb2.AnalysisRequest,
        context: grpc.aio.ServicerContext
    ) -> pb2.AnalysisResponse:
        """Handle market analysis requests"""
        try:
            # Convert gRPC request to domain model
            analysis_request = AnalysisRequest(
                request_id=request.request_id,
                analysis_type=AnalysisType(request.analysis_type),
                market_data=self._parse_market_data(request.market_data),
                options_data=self._parse_options_data(request.options_data),
                metadata=dict(request.metadata)
            )
            
            # Process request through service
            response = await self.service.process_request(analysis_request)
            
            # Convert to gRPC response
            return self._convert_to_grpc_response(response)
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Analysis failed: {str(e)}")
            raise

    async def GetStatus(
        self,
        request: pb2.StatusRequest,
        context: grpc.aio.ServicerContext
    ) -> pb2.StatusResponse:
        """Get service status"""
        try:
            metrics = await self.service.metrics_collector.get_metrics()
            agent_statuses = await self.service.agent_manager.list_agents()
            
            return pb2.StatusResponse(
                status="healthy",
                agent_statuses=[
                    self._convert_agent_status(agent)
                    for agent in agent_statuses
                ],
                metrics=self._convert_metrics(metrics)
            )
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Status check failed: {str(e)}")
            raise

    def _parse_market_data(
        self,
        market_data_proto: pb2.MarketData
    ) -> Dict[str, Any]:
        """Convert market data proto to dictionary"""
        return {
            'symbol': market_data_proto.symbol,
            'prices': list(market_data_proto.prices),
            'volumes': list(market_data_proto.volumes),
            'timestamps': [
                datetime.fromtimestamp(ts)
                for ts in market_data_proto.timestamps
            ],
            'indicators': dict(market_data_proto.indicators)
        }

    def _parse_options_data(
        self,
        options_data_proto: pb2.OptionsData
    ) -> Dict[str, Any]:
        """Convert options data proto to dictionary"""
        return {
            'symbol': options_data_proto.symbol,
            'expiry': datetime.fromtimestamp(options_data_proto.expiry),
            'strike': options_data_proto.strike,
            'option_type': options_data_proto.option_type,
            'price': options_data_proto.price,
            'implied_vol': options_data_proto.implied_vol
        }

    def _convert_to_grpc_response(
        self,
        response: AggregatedPrediction
    ) -> pb2.AnalysisResponse:
        """Convert domain response to gRPC response"""
        return pb2.AnalysisResponse(
            request_id=response.request_id,
            result=response.aggregated_result,
            confidence=response.confidence_score,
            processing_time=response.processing_time,
            metadata=response.metadata,
            agent_predictions=[
                pb2.AgentPrediction(
                    agent_id=pred.agent_id,
                    prediction=pred.prediction,
                    confidence=pred.confidence,
                    processing_time=pred.processing_time,
                    metadata=pred.metadata
                )
                for pred in response.predictions
            ]
        )

# services/client_service/infrastructure/generative_client.py

import grpc
from typing import Dict, Any
from datetime import datetime

from services.protos import generative_service_pb2 as pb2
from services.protos import generative_service_pb2_grpc as pb2_grpc

class GenerativeServiceClient:
    """
    Client for communicating with the Generative Agent service
    from the Flask backend
    """
    
    def __init__(self, host: str = "localhost", port: int = 50051):
        self.channel = grpc.aio.insecure_channel(f"{host}:{port}")
        self.stub = pb2_grpc.GenerativeServiceStub(self.channel)

    async def analyze_market(
        self,
        market_data: Dict[str, Any],
        options_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Send market analysis request"""
        try:
            # Create gRPC request
            request = pb2.AnalysisRequest(
                request_id=str(uuid.uuid4()),
                market_data=self._convert_market_data(market_data),
                options_data=self._convert_options_data(options_data) if options_data else None,
                metadata=metadata or {}
            )
            
            # Make gRPC call
            response = await self.stub.Analyze(request)
            
            # Convert response
            return {
                'request_id': response.request_id,
                'result': response.result,
                'confidence': response.confidence,
                'processing_time': response.processing_time,
                'metadata': dict(response.metadata),
                'agent_predictions': [
                    {
                        'agent_id': pred.agent_id,
                        'prediction': pred.prediction,
                        'confidence': pred.confidence,
                        'processing_time': pred.processing_time,
                        'metadata': dict(pred.metadata)
                    }
                    for pred in response.agent_predictions
                ]
            }
            
        except grpc.RpcError as e:
            raise Exception(f"RPC failed: {str(e)}")

    async def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        try:
            response = await self.stub.GetStatus(pb2.StatusRequest())
            
            return {
                'status': response.status,
                'agent_statuses': [
                    {
                        'agent_id': status.agent_id,
                        'status': status.status,
                        'load': status.load,
                        'last_check': datetime.fromtimestamp(
                            status.last_check
                        ).isoformat()
                    }
                    for status in response.agent_statuses
                ],
                'metrics': dict(response.metrics)
            }
            
        except grpc.RpcError as e:
            raise Exception(f"Status check failed: {str(e)}")

    def _convert_market_data(
        self,
        market_data: Dict[str, Any]
    ) -> pb2.MarketData:
        """Convert market data dict to proto message"""
        return pb2.MarketData(
            symbol=market_data['symbol'],
            prices=market_data['prices'],
            volumes=market_data['volumes'],
            timestamps=[
                int(ts.timestamp())
                for ts in market_data['timestamps']
            ],
            indicators=market_data.get('indicators', {})
        )

    def _convert_options_data(
        self,
        options_data: Dict[str, Any]
    ) -> pb2.OptionsData:
        """Convert options data dict to proto message"""
        return pb2.OptionsData(
            symbol=options_data['symbol'],
            expiry=int(options_data['expiry'].timestamp()),
            strike=options_data['strike'],
            option_type=options_data['option_type'],
            price=options_data['price'],
            implied_vol=options_data.get('implied_vol', 0.0)
        )

    async def close(self):
        """Close the gRPC channel"""
        await self.channel.close()
