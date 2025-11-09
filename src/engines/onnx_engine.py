"""
ONNX Runtime Inference Engine.

This module implements the ONNX Runtime inference engine with
various optimization configurations.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import onnxruntime as ort

from .base_engine import BaseInferenceEngine, InferenceError, ModelLoadError


class ONNXEngine(BaseInferenceEngine):
    """
    ONNX Runtime inference engine.

    Supports 3 optimization modes:
    - default: Standard ONNX Runtime settings
    - optimized: Graph optimization + parallelism
    - quantized: INT8 quantization (applied during conversion)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ONNX Runtime engine.

        Args:
            config: Configuration dictionary with keys:
                - mode (str): 'default', 'optimized', or 'quantized'
                - graph_optimization_level (str): Optimization level
                - inter_op_num_threads (int): Inter-op threads
                - intra_op_num_threads (int): Intra-op threads
                - execution_mode (str): 'SEQUENTIAL' or 'PARALLEL'
        """
        super().__init__(engine_name="onnxruntime", config=config)
        self.session = None
        self.input_names = None
        self.output_names = None

    def load_model(self, model_path: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Load an ONNX model.

        Args:
            model_path: Path to .onnx file
            config: Additional loading configuration

        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            # Create session options
            sess_options = ort.SessionOptions()

            # Apply configuration
            mode = self.config.get("mode", "default")

            if mode == "optimized" or mode == "quantized":
                # Graph optimization
                graph_opt_level = self.config.get("graph_optimization_level", "ENABLE_ALL")
                if graph_opt_level == "ENABLE_ALL":
                    sess_options.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    )
                elif graph_opt_level == "ENABLE_EXTENDED":
                    sess_options.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
                    )
                elif graph_opt_level == "ENABLE_BASIC":
                    sess_options.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                    )
                print(f"✓ Graph optimization: {graph_opt_level}")

                # Thread configuration
                inter_op_threads = self.config.get("inter_op_num_threads", 4)
                intra_op_threads = self.config.get("intra_op_num_threads", 8)

                sess_options.inter_op_num_threads = inter_op_threads
                sess_options.intra_op_num_threads = intra_op_threads
                print(f"✓ Threads: inter_op={inter_op_threads}, " f"intra_op={intra_op_threads}")

                # Execution mode
                execution_mode = self.config.get("execution_mode", "PARALLEL")
                if execution_mode == "PARALLEL":
                    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                else:
                    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                print(f"✓ Execution mode: {execution_mode}")

            # Set execution providers (CPU only)
            providers = ["CPUExecutionProvider"]

            # Create inference session
            self.session = ort.InferenceSession(model_path, sess_options, providers=providers)

            # Get input and output names
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]

            print(f"✓ Loaded ONNX model from {model_path}")
            print(f"  Inputs: {self.input_names}")
            print(f"  Outputs: {self.output_names}")

            self.model = self.session  # For compatibility
            self.is_loaded = True
            self._warmup_done = False

        except Exception as e:
            raise ModelLoadError(f"Failed to load ONNX model: {e}") from e

    def warmup(self, num_iterations: int = 10) -> None:
        """
        Warmup the ONNX Runtime session.

        Args:
            num_iterations: Number of warmup iterations

        Raises:
            RuntimeError: If session is not loaded or warmup fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Create dummy input
            dummy_input = self._create_dummy_input()

            print(f"Warming up ONNX engine ({num_iterations} iterations)...")

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
            # Prepare input feed dict
            if isinstance(inputs, dict):
                # Inputs already in dict format
                input_feed = inputs
            else:
                # Single input - use first input name
                input_feed = {self.input_names[0]: inputs}

            # Run inference
            outputs = self.session.run(self.output_names, input_feed)

            # Return outputs
            if len(outputs) == 1:
                return outputs[0]
            else:
                # Multiple outputs - return as dict
                return {name: output for name, output in zip(self.output_names, outputs)}

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
        # For ONNX, batch inference is the same as single inference
        return self.infer(inputs)

    def get_info(self) -> Dict[str, Any]:
        """
        Get ONNX Runtime engine information.

        Returns:
            Dictionary with engine details
        """
        info = {
            "engine_name": self.engine_name,
            "engine_version": ort.__version__,
            "model_loaded": self.is_loaded,
            "warmup_done": self._warmup_done,
            "config": self.config.copy(),
            "mode": self.config.get("mode", "default"),
            "execution_providers": ["CPUExecutionProvider"],
        }

        if self.is_loaded:
            # Add input/output details
            inputs_info = []
            for inp in self.session.get_inputs():
                inputs_info.append(
                    {
                        "name": inp.name,
                        "shape": inp.shape,
                        "type": inp.type,
                    }
                )

            outputs_info = []
            for out in self.session.get_outputs():
                outputs_info.append(
                    {
                        "name": out.name,
                        "shape": out.shape,
                        "type": out.type,
                    }
                )

            info["inputs"] = inputs_info
            info["outputs"] = outputs_info

        return info

    def cleanup(self) -> None:
        """Clean up ONNX Runtime resources."""
        if self.session is not None:
            del self.session
            self.session = None

        self.input_names = None
        self.output_names = None
        self.model = None
        self.is_loaded = False
        self._warmup_done = False

        print("✓ ONNX engine cleaned up")

    def _create_dummy_input(self) -> Dict[str, np.ndarray]:
        """
        Create dummy input for warmup.

        Returns:
            Dictionary of dummy input arrays
        """
        dummy_input = {}

        for inp in self.session.get_inputs():
            # Get input shape and replace dynamic dimensions
            shape = inp.shape
            concrete_shape = []
            for dim in shape:
                if isinstance(dim, str) or dim is None or dim < 0:
                    # Dynamic dimension - use 1
                    concrete_shape.append(1)
                else:
                    concrete_shape.append(dim)

            # Create random data
            dummy_input[inp.name] = np.random.randn(*concrete_shape).astype(np.float32)

        return dummy_input


def create_onnx_engine(mode: str = "default") -> ONNXEngine:
    """
    Create an ONNX Runtime engine with specified mode.

    Args:
        mode: Optimization mode:
            - 'default': Standard settings
            - 'optimized': Graph optimization + parallelism
            - 'quantized': INT8 quantization (model must be quantized)

    Returns:
        Configured ONNXEngine instance

    Raises:
        ValueError: If mode is not recognized
    """
    configs = {
        "default": {
            "mode": "default",
            "graph_optimization_level": "ENABLE_BASIC",
            "inter_op_num_threads": None,
            "intra_op_num_threads": None,
            "execution_mode": "SEQUENTIAL",
        },
        "optimized": {
            "mode": "optimized",
            "graph_optimization_level": "ENABLE_ALL",
            "inter_op_num_threads": 4,
            "intra_op_num_threads": 8,
            "execution_mode": "PARALLEL",
        },
        "quantized": {
            "mode": "quantized",
            "graph_optimization_level": "ENABLE_ALL",
            "inter_op_num_threads": 4,
            "intra_op_num_threads": 8,
            "execution_mode": "PARALLEL",
        },
    }

    if mode not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown mode '{mode}'. Available: {available}")

    return ONNXEngine(config=configs[mode])
