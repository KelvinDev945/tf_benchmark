"""
TensorFlow Inference Engine.

This module implements the TensorFlow inference engine with multiple
optimization configurations.
"""

import os
from typing import Any, Dict, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from .base_engine import BaseInferenceEngine, InferenceError, ModelLoadError


class TensorFlowEngine(BaseInferenceEngine):
    """
    TensorFlow inference engine with optimization support.

    Supports 5 configuration modes:
    - baseline: Default TensorFlow settings
    - xla: XLA JIT compilation enabled
    - threads: Optimized thread configuration
    - mixed_precision: BFloat16 mixed precision
    - best_combo: All optimizations combined
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TensorFlow engine.

        Args:
            config: Configuration dictionary with keys:
                - xla (bool): Enable XLA JIT compilation
                - mixed_precision (bool): Enable mixed precision
                - inter_op_threads (int): Inter-op parallelism threads
                - intra_op_threads (int): Intra-op parallelism threads
        """
        super().__init__(engine_name="tensorflow", config=config)
        self._apply_config()

    def _apply_config(self) -> None:
        """Apply TensorFlow configuration settings."""
        # XLA configuration
        if self.config.get("xla", False):
            tf.config.optimizer.set_jit(True)
            os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
            print("✓ XLA JIT compilation enabled")

        # Thread configuration
        inter_op_threads = self.config.get("inter_op_threads")
        intra_op_threads = self.config.get("intra_op_threads")

        if inter_op_threads is not None:
            tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
            print(f"✓ Inter-op threads set to {inter_op_threads}")

        if intra_op_threads is not None:
            tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
            print(f"✓ Intra-op threads set to {intra_op_threads}")

        # Mixed precision configuration
        if self.config.get("mixed_precision", False):
            policy = mixed_precision.Policy("mixed_bfloat16")
            mixed_precision.set_global_policy(policy)
            print(f"✓ Mixed precision enabled: {policy.name}")

    def load_model(
        self, model_path: Union[str, tf.keras.Model], config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Load a TensorFlow/Keras model.

        Args:
            model_path: Path to SavedModel directory or Keras model object
            config: Additional loading configuration

        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            if isinstance(model_path, str):
                # Load from path
                if os.path.isdir(model_path):
                    # SavedModel format
                    self.model = tf.saved_model.load(model_path)
                    print(f"✓ Loaded TensorFlow SavedModel from {model_path}")
                else:
                    # Try to load as Keras model
                    self.model = tf.keras.models.load_model(model_path)
                    print(f"✓ Loaded Keras model from {model_path}")
            elif hasattr(model_path, '__call__') and hasattr(model_path, 'predict'):
                # Model object passed directly (Keras or HuggingFace Transformers)
                # Accept any callable TensorFlow model with predict method
                self.model = model_path
                model_type = type(model_path).__name__
                print(f"✓ Loaded TensorFlow model from object ({model_type})")
            else:
                raise ModelLoadError(
                    f"Invalid model_path type: {type(model_path).__name__}. "
                    "Expected str or callable model with predict method"
                )

            self.is_loaded = True
            self._warmup_done = False

        except Exception as e:
            raise ModelLoadError(f"Failed to load TensorFlow model: {e}") from e

    def warmup(self, num_iterations: int = 10) -> None:
        """
        Warmup the TensorFlow model.

        Performs multiple inference iterations to ensure:
        - Graph compilation (especially for XLA)
        - Cache warmup
        - Stable timing

        Args:
            num_iterations: Number of warmup iterations (use more for XLA)

        Raises:
            RuntimeError: If model is not loaded or warmup fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Use more iterations for XLA (requires compilation time)
        if self.config.get("xla", False) and num_iterations < 50:
            print(
                f"⚠ XLA enabled: increasing warmup iterations from {num_iterations} to 50"
            )
            num_iterations = 50

        try:
            # Create dummy input based on model input shape
            input_shape = self._get_input_shape()
            dummy_input = self._create_dummy_input(input_shape)

            print(f"Warming up TensorFlow engine ({num_iterations} iterations)...")

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
            # Convert numpy to TensorFlow tensors
            if isinstance(inputs, dict):
                tf_inputs = {k: tf.constant(v) for k, v in inputs.items()}
            else:
                tf_inputs = tf.constant(inputs)

            # Run inference
            outputs = self.model(tf_inputs, training=False)

            # Convert back to numpy
            if isinstance(outputs, dict):
                return {k: v.numpy() for k, v in outputs.items()}
            elif hasattr(outputs, "logits"):
                # Handle HuggingFace model outputs
                return outputs.logits.numpy()
            else:
                return outputs.numpy()

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
        # For TensorFlow, batch inference is the same as single inference
        return self.infer(inputs)

    def get_info(self) -> Dict[str, Any]:
        """
        Get TensorFlow engine information.

        Returns:
            Dictionary with engine details
        """
        info = {
            "engine_name": self.engine_name,
            "engine_version": tf.__version__,
            "model_loaded": self.is_loaded,
            "warmup_done": self._warmup_done,
            "config": self.config.copy(),
            "xla_enabled": self.config.get("xla", False),
            "mixed_precision_enabled": self.config.get("mixed_precision", False),
        }

        # Add thread configuration
        if self.config.get("inter_op_threads"):
            info["inter_op_threads"] = self.config["inter_op_threads"]
        if self.config.get("intra_op_threads"):
            info["intra_op_threads"] = self.config["intra_op_threads"]

        # Add GPU information (if available)
        gpus = tf.config.list_physical_devices("GPU")
        info["num_gpus"] = len(gpus)
        info["gpu_available"] = len(gpus) > 0

        return info

    def cleanup(self) -> None:
        """Clean up TensorFlow resources."""
        if self.model is not None:
            del self.model
            self.model = None

        # Clear session
        tf.keras.backend.clear_session()

        self.is_loaded = False
        self._warmup_done = False

        print("✓ TensorFlow engine cleaned up")

    def _get_input_shape(self) -> Union[tuple, Dict[str, tuple]]:
        """
        Get model input shape.

        Returns:
            Input shape tuple or dict of shapes
        """
        if hasattr(self.model, "input_shape"):
            return self.model.input_shape
        elif hasattr(self.model, "input"):
            if isinstance(self.model.input, dict):
                return {k: v.shape for k, v in self.model.input.items()}
            else:
                return self.model.input.shape
        else:
            # Default shape for testing
            return (1, 224, 224, 3)

    def _create_dummy_input(
        self, input_shape: Union[tuple, Dict[str, tuple]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Create dummy input for warmup.

        Args:
            input_shape: Input shape(s)

        Returns:
            Dummy input data
        """
        if isinstance(input_shape, dict):
            # Multiple inputs
            dummy_input = {}
            for name, shape in input_shape.items():
                # Remove None dimensions and use batch size 1
                concrete_shape = tuple(1 if s is None else s for s in shape)
                dummy_input[name] = np.random.randn(*concrete_shape).astype(
                    np.float32
                )
            return dummy_input
        else:
            # Single input
            # Remove None dimensions and use batch size 1
            concrete_shape = tuple(1 if s is None else s for s in input_shape)
            return np.random.randn(*concrete_shape).astype(np.float32)


def create_tensorflow_engine(config_name: str) -> TensorFlowEngine:
    """
    Create a TensorFlow engine with predefined configuration.

    Args:
        config_name: Name of the configuration:
            - 'baseline': Default settings
            - 'xla': XLA JIT compilation
            - 'threads': Optimized threading
            - 'mixed_precision': BFloat16 mixed precision
            - 'best_combo': All optimizations

    Returns:
        Configured TensorFlowEngine instance

    Raises:
        ValueError: If config_name is not recognized
    """
    configs = {
        "baseline": {
            "xla": False,
            "mixed_precision": False,
            "inter_op_threads": None,
            "intra_op_threads": None,
        },
        "xla": {
            "xla": True,
            "mixed_precision": False,
            "inter_op_threads": None,
            "intra_op_threads": None,
        },
        "threads": {
            "xla": False,
            "mixed_precision": False,
            "inter_op_threads": 4,
            "intra_op_threads": 8,
        },
        "mixed_precision": {
            "xla": False,
            "mixed_precision": True,
            "inter_op_threads": None,
            "intra_op_threads": None,
        },
        "best_combo": {
            "xla": True,
            "mixed_precision": True,
            "inter_op_threads": 4,
            "intra_op_threads": 8,
        },
    }

    if config_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(
            f"Unknown config '{config_name}'. Available: {available}"
        )

    return TensorFlowEngine(config=configs[config_name])
