# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: generative_agent.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC, 5, 27, 2, "", "generative_agent.proto"
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x16generative_agent.proto\x12\x12generative_service"\'\n\x11PredictionRequest\x12\x12\n\ninput_data\x18\x01 \x01(\t"5\n\x12PredictionResponse\x12\x0e\n\x06result\x18\x01 \x01(\t\x12\x0f\n\x07\x64\x65tails\x18\x02 \x01(\t2w\n\x1bGenerativePredictionService\x12X\n\x07Predict\x12%.generative_service.PredictionRequest\x1a&.generative_service.PredictionResponseb\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "generative_agent_pb2", _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    DESCRIPTOR._loaded_options = None
    _globals["_PREDICTIONREQUEST"]._serialized_start = 46
    _globals["_PREDICTIONREQUEST"]._serialized_end = 85
    _globals["_PREDICTIONRESPONSE"]._serialized_start = 87
    _globals["_PREDICTIONRESPONSE"]._serialized_end = 140
    _globals["_GENERATIVEPREDICTIONSERVICE"]._serialized_start = 142
    _globals["_GENERATIVEPREDICTIONSERVICE"]._serialized_end = 261
# @@protoc_insertion_point(module_scope)
