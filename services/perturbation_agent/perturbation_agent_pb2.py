# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: perturbation_agent.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC, 5, 27, 2, "", "perturbation_agent.proto"
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b"\n\x18perturbation_agent.proto\x12\x14perturbation_service\"'\n\x11PredictionRequest\x12\x12\n\ninput_data\x18\x01 \x01(\t\"5\n\x12PredictionResponse\x12\x0e\n\x06result\x18\x01 \x01(\t\x12\x0f\n\x07\x64\x65tails\x18\x02 \x01(\t2}\n\x1dPerturbationPredictionService\x12\\\n\x07Predict\x12'.perturbation_service.PredictionRequest\x1a(.perturbation_service.PredictionResponseb\x06proto3"
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "perturbation_agent_pb2", _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    DESCRIPTOR._loaded_options = None
    _globals["_PREDICTIONREQUEST"]._serialized_start = 50
    _globals["_PREDICTIONREQUEST"]._serialized_end = 89
    _globals["_PREDICTIONRESPONSE"]._serialized_start = 91
    _globals["_PREDICTIONRESPONSE"]._serialized_end = 144
    _globals["_PERTURBATIONPREDICTIONSERVICE"]._serialized_start = 146
    _globals["_PERTURBATIONPREDICTIONSERVICE"]._serialized_end = 271
# @@protoc_insertion_point(module_scope)