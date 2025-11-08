"""
Base Inference Engine for TensorFlow Benchmark.

This module defines the abstract base class for all inference engines,
providing a unified interface for model loading and inference.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np


class BaseInferenceEngine(ABC):
    """
    Abstract base class for inference engines.

    All inference engines (TensorFlow, TFLite, ONNX, OpenVINO) must
    implement this interface to ensure consistent behavior across
    different backends.
    """

    def __init__(self, engine_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base inference engine.

        Args:
            engine_name: Name of the inference engine
            config: Engine configuration dictionary
        """
        self.engine_name = engine_name
        self.config = config or {}
        self.model = None
        self.is_loaded = False
        self._warmup_done = False

    @abstractmethod
    def load_model(
        self, model_path: Union[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Load a model for inference.

        Args:
            model_path: Path to model file or model object
            config: Additional configuration for loading

        Raises:
            RuntimeError: If model loading fails
        """
        pass

    @abstractmethod
    def warmup(self, num_iterations: int = 10) -> None:
        """
        Warmup the inference engine.

        Performs inference multiple times to ensure JIT compilation,
        cache warmup, and stable timing measurements.

        Args:
            num_iterations: Number of warmup iterations

        Raises:
            RuntimeError: If warmup fails
        """
        pass

    @abstractmethod
    def infer(
        self, inputs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Perform single inference.

        Args:
            inputs: Input data (array or dict of arrays)

        Returns:
            Output predictions (array or dict of arrays)

        Raises:
            RuntimeError: If inference fails
        """
        pass

    @abstractmethod
    def batch_infer(
        self, inputs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Perform batch inference.

        Args:
            inputs: Batch of input data

        Returns:
            Batch of output predictions

        Raises:
            RuntimeError: If batch inference fails
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the inference engine.

        Returns:
            Dictionary containing engine information:
                - engine_name: Name of the engine
                - engine_version: Version string
                - model_loaded: Whether a model is loaded
                - config: Current configuration
                - additional engine-specific info

        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up resources used by the engine.

        Releases memory, closes sessions, etc.
        Should be called when engine is no longer needed.
        """
        pass

    def is_model_loaded(self) -> bool:
        """
        Check if a model is currently loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        return self.is_loaded

    def is_warmup_done(self) -> bool:
        """
        Check if warmup has been completed.

        Returns:
            True if warmup is done, False otherwise
        """
        return self._warmup_done

    def reset_warmup(self) -> None:
        """Reset warmup status."""
        self._warmup_done = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()

    def __repr__(self) -> str:
        """String representation of the engine."""
        status = "loaded" if self.is_loaded else "not loaded"
        return f"{self.__class__.__name__}(name='{self.engine_name}', status='{status}')"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.engine_name} Inference Engine"


class InferenceEngineError(Exception):
    """Base exception for inference engine errors."""

    pass


class ModelLoadError(InferenceEngineError):
    """Exception raised when model loading fails."""

    pass


class InferenceError(InferenceEngineError):
    """Exception raised when inference fails."""

    pass


class WarmupError(InferenceEngineError):
    """Exception raised when warmup fails."""

    pass
