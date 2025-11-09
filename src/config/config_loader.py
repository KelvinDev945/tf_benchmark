"""
Configuration Loader for TensorFlow Benchmark.

This module provides functionality to load, validate, and manage
benchmark configuration from YAML files.
"""

import os
import platform
from pathlib import Path
from typing import Any, Dict, Union

import yaml


class ConfigLoader:
    """
    Configuration loader and validator.

    Handles loading benchmark configuration from YAML files,
    applying environment variable overrides, and validating
    configuration values.
    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize ConfigLoader.

        Args:
            config_path: Path to the configuration YAML file

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If configuration file is invalid YAML
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self.config = self._load_yaml()
        self._apply_environment_overrides()
        self._validate_config()
        self._apply_architecture_constraints()

    def _load_yaml(self) -> Dict[str, Any]:
        """
        Load YAML configuration file.

        Returns:
            Dictionary containing configuration

        Raises:
            yaml.YAMLError: If YAML is invalid
        """
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError("Configuration file is empty")

        return config

    def _apply_environment_overrides(self) -> None:
        """
        Apply environment variable overrides to configuration.

        Environment variables should be in format:
        BENCHMARK_<SECTION>_<KEY>=<VALUE>

        Examples:
            BENCHMARK_BENCHMARK_WARMUP_ITERATIONS=100
            BENCHMARK_OUTPUT_RESULTS_DIR=/custom/path
        """
        prefix = "BENCHMARK_"

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # Remove prefix and split into parts
            parts = key[len(prefix) :].lower().split("_")

            if len(parts) < 2:
                continue

            # Navigate to the target config section
            current = self.config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the value (attempt type conversion)
            final_key = parts[-1]
            current[final_key] = self._convert_type(value)

    def _convert_type(self, value: str) -> Union[str, int, float, bool]:
        """
        Convert string value to appropriate type.

        Args:
            value: String value from environment variable

        Returns:
            Converted value
        """
        # Try boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _validate_config(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate benchmark section
        if "benchmark" not in self.config:
            raise ValueError("Missing 'benchmark' section in configuration")

        benchmark = self.config["benchmark"]
        self._validate_positive_int(benchmark, "warmup_iterations")
        self._validate_positive_int(benchmark, "test_iterations")
        self._validate_positive_int(benchmark, "repeat_runs")

        # Validate dataset section
        if "dataset" not in self.config:
            raise ValueError("Missing 'dataset' section in configuration")

        # Validate models section
        if "models" not in self.config:
            raise ValueError("Missing 'models' section in configuration")

        models = self.config["models"]
        image_models = models.get("image", [])
        if not image_models:
            raise ValueError("At least one image model must be specified")

        # Validate batch sizes
        if "batch_sizes" not in self.config:
            raise ValueError("Missing 'batch_sizes' in configuration")

        batch_sizes = self.config["batch_sizes"]
        if not isinstance(batch_sizes, list) or not batch_sizes:
            raise ValueError("'batch_sizes' must be a non-empty list")

        if any(not isinstance(bs, int) or bs <= 0 for bs in batch_sizes):
            raise ValueError("All batch sizes must be positive integers")

        # Validate engines section
        if "engines" not in self.config:
            raise ValueError("Missing 'engines' section in configuration")

        # Validate output section
        if "output" not in self.config:
            raise ValueError("Missing 'output' section in configuration")

    def _validate_positive_int(self, section: Dict, key: str) -> None:
        """
        Validate that a configuration value is a positive integer.

        Args:
            section: Configuration section dictionary
            key: Key to validate

        Raises:
            ValueError: If value is not a positive integer
        """
        if key not in section:
            raise ValueError(f"Missing '{key}' in configuration section")

        value = section[key]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"'{key}' must be a positive integer, got {value}")

    def _apply_architecture_constraints(self) -> None:
        """
        Apply architecture-specific constraints.

        For example, disable OpenVINO on ARM64 architecture.
        """
        arch = platform.machine()

        # Disable OpenVINO on non-x86_64 architectures
        if arch != "x86_64" and "engines" in self.config:
            engines = self.config["engines"]
            if "openvino" in engines:
                engines["openvino"]["enabled"] = False
                print(
                    f"Warning: OpenVINO disabled on {arch} architecture "
                    "(only supported on x86_64)"
                )

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Path to configuration key using dots (e.g., 'benchmark.warmup_iterations')
            default: Default value if key doesn't exist

        Returns:
            Configuration value or default

        Examples:
            >>> config.get('benchmark.warmup_iterations')
            50
            >>> config.get('nonexistent.key', default=100)
            100
        """
        keys = key_path.split(".")
        current = self.config

        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]

        return current

    def get_mode_config(self, mode: str = "standard") -> Dict[str, Any]:
        """
        Get configuration for a specific testing mode.

        Args:
            mode: Testing mode ('quick', 'standard', or 'full')

        Returns:
            Configuration dictionary for the specified mode

        Raises:
            ValueError: If mode is not recognized
        """
        if "modes" not in self.config:
            raise ValueError("No 'modes' section in configuration")

        modes = self.config["modes"]
        if mode not in modes:
            raise ValueError(f"Unknown mode '{mode}'. Available modes: {list(modes.keys())}")

        # Create a copy of the main config
        mode_config = self.config.copy()

        # Apply mode-specific overrides
        mode_settings = modes[mode]

        if "warmup_iterations" in mode_settings:
            mode_config["benchmark"]["warmup_iterations"] = mode_settings["warmup_iterations"]

        if "test_iterations" in mode_settings:
            mode_config["benchmark"]["test_iterations"] = mode_settings["test_iterations"]

        if "repeat_runs" in mode_settings:
            mode_config["benchmark"]["repeat_runs"] = mode_settings["repeat_runs"]

        if "batch_sizes" in mode_settings:
            mode_config["batch_sizes"] = mode_settings["batch_sizes"]

        if "sequence_lengths" in mode_settings:
            mode_config["sequence_lengths"] = mode_settings["sequence_lengths"]

        if "num_samples_image" in mode_settings:
            mode_config["dataset"]["image"]["num_samples"] = mode_settings["num_samples_image"]

        if "num_samples_text" in mode_settings:
            text_dataset = mode_config.get("dataset", {}).get("text")
            if isinstance(text_dataset, dict):
                text_dataset["num_samples"] = mode_settings["num_samples_text"]

        return mode_config

    def get_enabled_engines(self) -> Dict[str, list]:
        """
        Get list of enabled engines and their configurations.

        Returns:
            Dictionary mapping engine names to their enabled configurations
        """
        enabled_engines = {}

        if "engines" not in self.config:
            return enabled_engines

        engines = self.config["engines"]

        for engine_name, engine_config in engines.items():
            if engine_config.get("enabled", False):
                enabled_engines[engine_name] = engine_config.get("configs", [])

        return enabled_engines

    def to_dict(self) -> Dict[str, Any]:
        """
        Get the full configuration as a dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()

    def save(self, output_path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.

        Args:
            output_path: Path where to save the configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        """String representation of ConfigLoader."""
        return f"ConfigLoader(config_path='{self.config_path}')"

    def __str__(self) -> str:
        """Human-readable string representation."""
        num_engines = len(self.config.get("engines", {}))
        num_models = len(self.config.get("models", {}).get("image", [])) + len(
            self.config.get("models", {}).get("text", [])
        )
        return (
            f"BenchmarkConfig(engines={num_engines}, models={num_models}, "
            f"source='{self.config_path}')"
        )
