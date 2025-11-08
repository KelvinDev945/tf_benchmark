"""
TensorFlow Lite Inference Engine.

This module implements the TFLite inference engine with support
for various quantization strategies.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import tensorflow as tf

from .base_engine import BaseInferenceEngine, InferenceError, ModelLoadError


class TFLiteEngine(BaseInferenceEngine):
    """
    TensorFlow Lite inference engine.

    Supports 4 quantization modes:
    - float32: No quantization (baseline)
    - dynamic_range: Dynamic range quantization
    - int8: Full integer quantization
    - float16: Half-precision quantization
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TFLite engine.

        Args:
            config: Configuration dictionary with keys:
                - optimization (str): Quantization mode
                - num_threads (int): Number of CPU threads
        """
        super().__init__(engine_name="tflite", config=config)
        self.interpreter = None
        self.input_details = None
        self.output_details = None

    def load_model(
        self, model_path: Union[str, bytes], config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Load a TFLite model.

        Args:
            model_path: Path to .tflite file or model bytes
            config: Additional loading configuration

        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            # Get number of threads from config
            num_threads = self.config.get("num_threads", 4)

            # Load model
            if isinstance(model_path, bytes):
                # Model bytes provided directly
                self.interpreter = tf.lite.Interpreter(
                    model_content=model_path, num_threads=num_threads
                )
                print(f"✓ Loaded TFLite model from bytes")
            elif isinstance(model_path, str):
                # Load from file path
                self.interpreter = tf.lite.Interpreter(
                    model_path=model_path, num_threads=num_threads
                )
                print(f"✓ Loaded TFLite model from {model_path}")
            else:
                raise ModelLoadError(
                    f"Invalid model_path type: {type(model_path)}. "
                    "Expected str or bytes"
                )

            # Allocate tensors
            self.interpreter.allocate_tensors()

            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            print(f"  Num threads: {num_threads}")
            print(f"  Input shape: {self.input_details[0]['shape']}")
            print(f"  Output shape: {self.output_details[0]['shape']}")

            self.model = self.interpreter  # For compatibility
            self.is_loaded = True
            self._warmup_done = False

        except Exception as e:
            raise ModelLoadError(f"Failed to load TFLite model: {e}") from e

    def warmup(self, num_iterations: int = 10) -> None:
        """
        Warmup the TFLite interpreter.

        Args:
            num_iterations: Number of warmup iterations

        Raises:
            RuntimeError: If interpreter is not loaded or warmup fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Create dummy input
            input_shape = tuple(self.input_details[0]["shape"])
            input_dtype = self.input_details[0]["dtype"]

            # Create appropriate dummy data based on dtype
            if input_dtype == np.uint8:
                dummy_input = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
            elif input_dtype == np.int8:
                dummy_input = np.random.randint(-128, 128, size=input_shape, dtype=np.int8)
            else:
                dummy_input = np.random.randn(*input_shape).astype(input_dtype)

            print(f"Warming up TFLite engine ({num_iterations} iterations)...")

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
            # Handle dict inputs (use first input)
            if isinstance(inputs, dict):
                input_data = list(inputs.values())[0]
            else:
                input_data = inputs

            # Ensure input has correct dtype
            input_dtype = self.input_details[0]["dtype"]
            if input_data.dtype != input_dtype:
                input_data = input_data.astype(input_dtype)

            # Set input tensor
            self.interpreter.set_tensor(
                self.input_details[0]["index"], input_data
            )

            # Run inference
            self.interpreter.invoke()

            # Get output tensor
            output_data = self.interpreter.get_tensor(
                self.output_details[0]["index"]
            )

            return output_data

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
        # For TFLite, batch inference is the same as single inference
        return self.infer(inputs)

    def get_info(self) -> Dict[str, Any]:
        """
        Get TFLite engine information.

        Returns:
            Dictionary with engine details
        """
        info = {
            "engine_name": self.engine_name,
            "engine_version": tf.__version__,
            "model_loaded": self.is_loaded,
            "warmup_done": self._warmup_done,
            "config": self.config.copy(),
            "optimization": self.config.get("optimization", "float32"),
            "num_threads": self.config.get("num_threads", 4),
        }

        if self.is_loaded:
            # Add input/output details
            info["input_shape"] = tuple(self.input_details[0]["shape"])
            info["input_dtype"] = str(self.input_details[0]["dtype"])
            info["output_shape"] = tuple(self.output_details[0]["shape"])
            info["output_dtype"] = str(self.output_details[0]["dtype"])

            # Check if model is quantized
            input_quantization = self.input_details[0].get("quantization")
            output_quantization = self.output_details[0].get("quantization")

            if input_quantization:
                info["input_quantized"] = True
                info["input_scale"] = float(input_quantization[0]) if input_quantization[0] else None
                info["input_zero_point"] = int(input_quantization[1]) if input_quantization[1] else None

            if output_quantization:
                info["output_quantized"] = True
                info["output_scale"] = float(output_quantization[0]) if output_quantization[0] else None
                info["output_zero_point"] = int(output_quantization[1]) if output_quantization[1] else None

        return info

    def cleanup(self) -> None:
        """Clean up TFLite resources."""
        if self.interpreter is not None:
            del self.interpreter
            self.interpreter = None

        self.input_details = None
        self.output_details = None
        self.model = None
        self.is_loaded = False
        self._warmup_done = False

        print("✓ TFLite engine cleaned up")


def create_tflite_engine(optimization: str = "float32") -> TFLiteEngine:
    """
    Create a TFLite engine with specified optimization.

    Args:
        optimization: Quantization mode:
            - 'float32': No quantization
            - 'dynamic_range': Dynamic range quantization
            - 'int8': Full integer quantization
            - 'float16': Half-precision quantization

    Returns:
        Configured TFLiteEngine instance

    Raises:
        ValueError: If optimization is not recognized
    """
    valid_optimizations = ["float32", "dynamic_range", "int8", "float16"]

    if optimization not in valid_optimizations:
        available = ", ".join(valid_optimizations)
        raise ValueError(
            f"Unknown optimization '{optimization}'. Available: {available}"
        )

    config = {
        "optimization": optimization,
        "num_threads": 4,
    }

    return TFLiteEngine(config=config)
