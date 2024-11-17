# services/tca_service/presentation/grpc_servicer.py

import grpc
from datetime import datetime
from typing import Dict, Any

from services.protos import tca_service_pb2 as pb2
from services.protos import tca_service_pb2_grpc as pb2_grpc
from ..application.service import TCAService
from ..domain.models import (
    PredictionRequest, AgentInfo, RequestType, 
    AgentStatus, AgentPrediction
)

class TCAServicer(pb2_grpc.TrafficControlServiceServicer):
    """gRPC servicer implementation for Traffic Control Agent"""
    
    def __init__(self, service: TCAService):
        self.service = service

    async def ProcessRequest(
        self, 
        request: pb2.PredictionRequest, 
        context: grpc.aio.ServicerContext
    ) -> pb2.PredictionResponse:
        """
        Handle incoming prediction requests
        """
        try:
            # Convert gRPC request to domain model
            prediction_request = PredictionRequest(
                request_id=request.request_id,
                input_data=request.input_data,
                request_type=RequestType(request.type),
                metadata=dict(request.metadata),
                preferred_agents=list(request.preferred_agents)
            )
            
            # Process request through service
            result = await self.service.process_request(prediction_request)
            
            # Convert domain model to gRPC response
            agent_predictions = [
                pb2.AgentPrediction(
                    agent_id=pred.agent_id,
                    prediction=pred.prediction,
                    confidence=pred.confidence,
                    processing_time=int(pred.processing_time * 1000)  # Convert to milliseconds
                ) for pred in result.predictions
            ]
            
            return pb2.PredictionResponse(
                request_id=result.request_id,
                predictions=agent_predictions,
                aggregated_result=result.aggregated_result,
                confidence_score=result.confidence_score,
                metadata=result.metadata
            )
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error processing request: {str(e)}")
            raise

    async def RegisterAgent(
        self, 
        request: pb2.AgentRegistration, 
        context: grpc.aio.ServicerContext
    ) -> pb2.RegistrationResponse:
        """
        Handle agent registration requests
        """
        try:
            agent_info = AgentInfo(
                agent_id=request.agent_id,
                agent_type=request.agent_type,
                host=request.host,
                port=request.port,
                supported_types=[RequestType(t) for t in request.supported_types],
                capabilities=dict(request.capabilities),
                status=AgentStatus.ACTIVE,
                last_health_check=datetime.now()
            )
            
            success = await self.service.register_agent(agent_info)
            
            return pb2.RegistrationResponse(
                success=success,
                message="Agent registered successfully" if success else "Registration failed"
            )
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Registration failed: {str(e)}")
            return pb2.RegistrationResponse(
                success=False,
                message=f"Registration failed: {str(e)}"
            )

    async def DeregisterAgent(
        self, 
        request: pb2.DeregistrationRequest, 
        context: grpc.aio.ServicerContext
    ) -> pb2.DeregistrationResponse:
        """
        Handle agent deregistration requests
        """
        try:
            success = await self.service.deregister_agent(request.agent_id)
            
            return pb2.DeregistrationResponse(
                success=success,
                message="Agent deregistered successfully" if success else "Deregistration failed"
            )
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Deregistration failed: {str(e)}")
            return pb2.DeregistrationResponse(
                success=False,
                message=f"Deregistration failed: {str(e)}"
            )

    async def GetStatus(
        self, 
        request: pb2.StatusRequest, 
        context: grpc.aio.ServicerContext
    ) -> pb2.StatusResponse:
        """
        Handle status requests
        """
        try:
            if request.agent_id:
                # Get status for specific agent
                agent_status = await self.service.get_agent_status(request.agent_id)
                agent_statuses = [agent_status] if agent_status else []
            else:
                # Get status for all agents
                agent_statuses = await self.service.get_all_agent_status()
            
            # Get system metrics
            metrics = await self.service.get_metrics()
            
            # Convert to gRPC response
            status_responses = [
                pb2.AgentStatus(
                    agent_id=status.agent_id,
                    status=status.status.value,
                    load=status.load,
                    pending_requests=status.metrics.get('pending_requests', 0),
                    metrics=status.metrics
                ) for status in agent_statuses
            ]
            
            system_metrics = pb2.SystemMetrics(
                total_requests=metrics.get('total_requests', 0),
                active_requests=metrics.get('active_requests', 0),
                average_response_time=metrics.get('average_response_time', 0.0),
                agent_utilization=metrics.get('agent_utilization', {})
            )
            
            return pb2.StatusResponse(
                agents=status_responses,
                metrics=system_metrics
            )
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error getting status: {str(e)}")
            raise

    async def UpdateAgentHealth(
        self, 
        request: pb2.HealthUpdate, 
        context: grpc.aio.ServicerContext
    ) -> pb2.HealthResponse:
        """
        Handle agent health updates
        """
        try:
            success = await self.service.update_agent_health(
                agent_id=request.agent_id,
                status=request.status,
                metrics=dict(request.metrics)
            )
            
            return pb2.HealthResponse(
                acknowledged=success,
                message="Health update processed" if success else "Health update failed"
            )
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Health update failed: {str(e)}")
            return pb2.HealthResponse(
                acknowledged=False,
                message=f"Health update failed: {str(e)}"
            )
