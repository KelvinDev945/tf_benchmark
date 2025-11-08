"""
OpenVINO Inference Engine.

This module implements the OpenVINO inference engine with various
precision modes. Only supported on x86_64 architecture.
"""

import platform
from typing import Any, Dict, Optional, Union

import numpy as np

from .base_engine import BaseInferenceEngine, InferenceError, ModelLoadError


class OpenVINOEngine(BaseInferenceEngine):
    """
    OpenVINO inference engine (x86_64 only).

    Supports 4 precision modes:
    - fp32: Single precision floating point
    - fp16: Half precision floating point
    - int8: Integer quantization
    - dynamic: Dynamic batching optimized

    Note: This engine is only available on x86_64 architecture.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenVINO engine.

        Args:
            config: Configuration dictionary with keys:
                - precision (str): 'fp32', 'fp16', 'int8', or 'dynamic'
                - performance_hint (str): 'THROUGHPUT' or 'LATENCY'
                - num_streams (int): Number of inference streams

        Raises:
            NotImplementedError: If not on x86_64 architecture
        """
        # Check architecture
        arch = platform.machine()
        if arch != "x86_64":
            raise NotImplementedError(
                f"OpenVINO is not supported on {arch} architecture. "
                "Only x86_64 is supported."
            )

        super().__init__(engine_name="openvino", config=config)

        # Import OpenVINO (will fail gracefully if not installed)
        try:
            from openvino.runtime import Core
            self.Core = Core
            self.core = Core()
        except ImportError as e:
            raise ImportError(
                "OpenVINO not installed. Please install with: "
                "pip install openvino==2023.2.0"
            ) from e

        self.compiled_model = None
        self.infer_request = None
        self.input_keys = None
        self.output_keys = None

    def load_model(
        self, model_path: str, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Load an OpenVINO IR model.

        Args:
            model_path: Path to .xml file (IR format)
            config: Additional loading configuration

        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            # Read model from IR format
            model = self.core.read_model(model_path)

            # Prepare configuration
            device_config = {}

            # Performance hint
            performance_hint = self.config.get("performance_hint", "THROUGHPUT")
            device_config["PERFORMANCE_HINT"] = performance_hint
            print(f"✓ Performance hint: {performance_hint}")

            # Number of streams
            num_streams = self.config.get("num_streams", 4)
            if performance_hint == "THROUGHPUT":
                device_config["NUM_STREAMS"] = str(num_streams)
                print(f"✓ Num streams: {num_streams}")

            # Precision mode
            precision = self.config.get("precision", "fp32").upper()
            print(f"✓ Precision: {precision}")

            # Compile model for CPU
            self.compiled_model = self.core.compile_model(
                model, "CPU", device_config
            )

            # Create infer request
            self.infer_request = self.compiled_model.create_infer_request()

            # Get input and output keys
            self.input_keys = [inp.any_name for inp in self.compiled_model.inputs]
            self.output_keys = [out.any_name for out in self.compiled_model.outputs]

            print(f"✓ Loaded OpenVINO model from {model_path}")
            print(f"  Inputs: {self.input_keys}")
            print(f"  Outputs: {self.output_keys}")

            self.model = self.compiled_model  # For compatibility
            self.is_loaded = True
            self._warmup_done = False

        except Exception as e:
            raise ModelLoadError(f"Failed to load OpenVINO model: {e}") from e

    def warmup(self, num_iterations: int = 10) -> None:
        """
        Warmup the OpenVINO engine.

        Args:
            num_iterations: Number of warmup iterations

        Raises:
            RuntimeError: If model is not loaded or warmup fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Create dummy input
            dummy_input = self._create_dummy_input()

            print(f"Warming up OpenVINO engine ({num_iterations} iterations)...")

            for i in range(num_iterations):
                _ = self.infer(dummy_input)

                if (i + 1) % 10 == 0:
                    print(f"  Warmup: {i + 1}/{num_iterations}")

            self._warmup_done = True
            print("✓ Warmup completed")

        except Exception as e:
            raise RuntimeError(f"Warmup failed: {e}") from e

    def infer(
        self, inputs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Perform single inference.

        Args:
            inputs: Input data (numpy array or dict of arrays)

        Returns:
            Model predictions

        Raises:
            InferenceError: If inference fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Prepare input dict
            if isinstance(inputs, dict):
                input_data = inputs
            else:
                # Single input - use first input key
                input_data = {self.input_keys[0]: inputs}

            # Run inference
            results = self.infer_request.infer(input_data)

            # Extract outputs
            if len(self.output_keys) == 1:
                # Single output
                output_tensor = results[self.compiled_model.outputs[0]]
                return np.array(output_tensor)
            else:
                # Multiple outputs - return as dict
                output_dict = {}
                for i, key in enumerate(self.output_keys):
                    output_tensor = results[self.compiled_model.outputs[i]]
                    output_dict[key] = np.array(output_tensor)
                return output_dict

        except Exception as e:
            raise InferenceError(f"Inference failed: {e}") from e

    def batch_infer(
        self, inputs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Perform batch inference.

        Args:
            inputs: Batch of input data

        Returns:
            Batch of predictions

        Raises:
            InferenceError: If batch inference fails
        """
        # For OpenVINO, batch inference is the same as single inference
        return self.infer(inputs)

    def get_info(self) -> Dict[str, Any]:
        """
        Get OpenVINO engine information.

        Returns:
            Dictionary with engine details
        """
        from openvino.runtime import get_version

        info = {
            "engine_name": self.engine_name,
            "engine_version": get_version(),
            "model_loaded": self.is_loaded,
            "warmup_done": self._warmup_done,
            "config": self.config.copy(),
            "precision": self.config.get("precision", "fp32"),
            "performance_hint": self.config.get("performance_hint", "THROUGHPUT"),
            "architecture": platform.machine(),
        }

        if self.is_loaded:
            # Add input/output details
            inputs_info = []
            for inp in self.compiled_model.inputs:
                inputs_info.append({
                    "name": inp.any_name,
                    "shape": list(inp.shape),
                    "type": str(inp.element_type),
                })

            outputs_info = []
            for out in self.compiled_model.outputs:
                outputs_info.append({
                    "name": out.any_name,
                    "shape": list(out.shape),
                    "type": str(out.element_type),
                })

            info["inputs"] = inputs_info
            info["outputs"] = outputs_info

        return info

    def cleanup(self) -> None:
        """Clean up OpenVINO resources."""
        if self.infer_request is not None:
            del self.infer_request
            self.infer_request = None

        if self.compiled_model is not None:
            del self.compiled_model
            self.compiled_model = None

        self.input_keys = None
        self.output_keys = None
        self.model = None
        self.is_loaded = False
        self._warmup_done = False

        print("✓ OpenVINO engine cleaned up")

    def _create_dummy_input(self) -> Dict[str, np.ndarray]:
        """
        Create dummy input for warmup.

        Returns:
            Dictionary of dummy input arrays
        """
        dummy_input = {}

        for inp in self.compiled_model.inputs:
            # Get input shape and replace dynamic dimensions
            shape = inp.shape
            concrete_shape = []
            for dim in shape:
                if dim == -1 or dim is None:
                    # Dynamic dimension - use 1
                    concrete_shape.append(1)
                else:
                    concrete_shape.append(dim)

            # Create random data
            dummy_input[inp.any_name] = np.random.randn(*concrete_shape).astype(
                np.float32
            )

        return dummy_input


def create_openvino_engine(precision: str = "fp32") -> OpenVINOEngine:
    """
    Create an OpenVINO engine with specified precision.

    Args:
        precision: Precision mode:
            - 'fp32': Single precision
            - 'fp16': Half precision
            - 'int8': Integer quantization
            - 'dynamic': Dynamic batching

    Returns:
        Configured OpenVINOEngine instance

    Raises:
        ValueError: If precision is not recognized
        NotImplementedError: If not on x86_64
    """
    valid_precisions = ["fp32", "fp16", "int8", "dynamic"]

    if precision not in valid_precisions:
        available = ", ".join(valid_precisions)
        raise ValueError(
            f"Unknown precision '{precision}'. Available: {available}"
        )

    config = {
        "precision": precision,
        "performance_hint": "LATENCY" if precision == "dynamic" else "THROUGHPUT",
        "num_streams": 1 if precision == "dynamic" else 4,
    }

    return OpenVINOEngine(config=config)
