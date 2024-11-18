# services/generative_agent/presentation/grpc_servicer.py
import grpc
from datetime import datetime
from typing import Dict, Any

from services.protos import generative_agent_pb2 as pb2
from services.protos import generative_agent_pb2_grpc as pb2_grpc
from ..application.service import GenerativeService
from ..domain.models import AnalysisRequest, AnalysisType
from ..domain.exceptions import GenerativeServiceError


class GenerativeServicer(pb2_grpc.GenerativeServiceServicer):
    def __init__(self, service: GenerativeService):
        self.service = service

    async def Analyze(
        self, request: pb2.AnalysisRequest, context
    ) -> pb2.AnalysisResponse:
        try:
            analysis_request = AnalysisRequest(
                request_id=request.request_id,
                query=request.query,
                analysis_type=AnalysisType(request.analysis_type),
                context=dict(request.context),
                metadata=dict(request.metadata),
            )

            result = await self.service.analyze(analysis_request)

            return pb2.AnalysisResponse(
                request_id=result.request_id,
                summary=result.summary,
                detailed_analysis=result.detailed_analysis,
                recommendations=result.recommendations,
                confidence_score=result.confidence_score,
                processing_time=result.processing_time,
                metadata=result.metadata,
            )

        except GenerativeServiceError as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.AnalysisResponse()

    async def GetHealth(
        self, request: pb2.HealthRequest, context
    ) -> pb2.HealthResponse:
        try:
            status = await self.service.get_health_status()
            return pb2.HealthResponse(
                status=status["status"],
                timestamp=status["last_check"],
                metrics=status["service_metrics"],
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.HealthResponse(status="unhealthy")
