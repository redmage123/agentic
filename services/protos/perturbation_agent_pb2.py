# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: perturbation_agent.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x18perturbation_agent.proto\x12\x14perturbation_service\"\'\n\x11PredictionRequest\x12\x12\n\ninput_data\x18\x01 \x01(\t\"5\n\x12PredictionResponse\x12\x0e\n\x06result\x18\x01 \x01(\t\x12\x0f\n\x07\x64\x65tails\x18\x02 \x01(\t2}\n\x1dPerturbationPredictionService\x12\\\n\x07Predict\x12\'.perturbation_service.PredictionRequest\x1a(.perturbation_service.PredictionResponseb\x06proto3')



_PREDICTIONREQUEST = DESCRIPTOR.message_types_by_name['PredictionRequest']
_PREDICTIONRESPONSE = DESCRIPTOR.message_types_by_name['PredictionResponse']
PredictionRequest = _reflection.GeneratedProtocolMessageType('PredictionRequest', (_message.Message,), {
  'DESCRIPTOR' : _PREDICTIONREQUEST,
  '__module__' : 'perturbation_agent_pb2'
  # @@protoc_insertion_point(class_scope:perturbation_service.PredictionRequest)
  })
_sym_db.RegisterMessage(PredictionRequest)

PredictionResponse = _reflection.GeneratedProtocolMessageType('PredictionResponse', (_message.Message,), {
  'DESCRIPTOR' : _PREDICTIONRESPONSE,
  '__module__' : 'perturbation_agent_pb2'
  # @@protoc_insertion_point(class_scope:perturbation_service.PredictionResponse)
  })
_sym_db.RegisterMessage(PredictionResponse)

_PERTURBATIONPREDICTIONSERVICE = DESCRIPTOR.services_by_name['PerturbationPredictionService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _PREDICTIONREQUEST._serialized_start=50
  _PREDICTIONREQUEST._serialized_end=89
  _PREDICTIONRESPONSE._serialized_start=91
  _PREDICTIONRESPONSE._serialized_end=144
  _PERTURBATIONPREDICTIONSERVICE._serialized_start=146
  _PERTURBATIONPREDICTIONSERVICE._serialized_end=271
# @@protoc_insertion_point(module_scope)
