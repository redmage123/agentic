# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import fourier_agent_pb2 as fourier__agent__pb2


class FourierPredictionServiceStub(object):
    """Service for Fourier Neural Network predictions
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Predict = channel.unary_unary(
                '/fourier_service.FourierPredictionService/Predict',
                request_serializer=fourier__agent__pb2.PredictionRequest.SerializeToString,
                response_deserializer=fourier__agent__pb2.PredictionResponse.FromString,
                )


class FourierPredictionServiceServicer(object):
    """Service for Fourier Neural Network predictions
    """

    def Predict(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FourierPredictionServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Predict': grpc.unary_unary_rpc_method_handler(
                    servicer.Predict,
                    request_deserializer=fourier__agent__pb2.PredictionRequest.FromString,
                    response_serializer=fourier__agent__pb2.PredictionResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'fourier_service.FourierPredictionService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class FourierPredictionService(object):
    """Service for Fourier Neural Network predictions
    """

    @staticmethod
    def Predict(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/fourier_service.FourierPredictionService/Predict',
            fourier__agent__pb2.PredictionRequest.SerializeToString,
            fourier__agent__pb2.PredictionResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
