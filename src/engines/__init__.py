"""Inference engines package."""

from .base_engine import (
    BaseInferenceEngine,
    InferenceEngineError,
    ModelLoadError,
    InferenceError,
    WarmupError,
)
from .tensorflow_engine import TensorFlowEngine, create_tensorflow_engine
from .tflite_engine import TFLiteEngine, create_tflite_engine
from .onnx_engine import ONNXEngine, create_onnx_engine

# OpenVINO is only available on x86_64
try:
    from .openvino_engine import OpenVINOEngine, create_openvino_engine

    OPENVINO_AVAILABLE = True
except (ImportError, NotImplementedError):
    OPENVINO_AVAILABLE = False
    OpenVINOEngine = None
    create_openvino_engine = None

__all__ = [
    "BaseInferenceEngine",
    "InferenceEngineError",
    "ModelLoadError",
    "InferenceError",
    "WarmupError",
    "TensorFlowEngine",
    "create_tensorflow_engine",
    "TFLiteEngine",
    "create_tflite_engine",
    "ONNXEngine",
    "create_onnx_engine",
    "OpenVINOEngine",
    "create_openvino_engine",
    "OPENVINO_AVAILABLE",
]
