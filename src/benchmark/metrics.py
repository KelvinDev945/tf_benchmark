"""
Metrics Collector for TensorFlow Benchmark.

This module provides functionality to collect and calculate
performance metrics (latency, throughput, resource usage).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class MetricsCollector:
    """
    Performance metrics collector.

    Collects latency, throughput, and resource usage metrics
    and calculates comprehensive statistics.
    """

    def __init__(self):
        """Initialize MetricsCollector."""
        # Latency samples (in milliseconds)
        self.latencies: List[float] = []

        # Resource usage samples
        self.cpu_usage: List[float] = []
        self.memory_usage: List[float] = []

        # Model information
        self.model_info: Dict = {}

        # Conversion information
        self.conversion_info: Dict = {}

        # Timing information
        self.total_time: float = 0.0
        self.num_samples: int = 0
        self.num_batches: int = 0
        self.batch_size: int = 1

    def add_latency(self, latency_ms: float) -> None:
        """
        Add a latency sample.

        Args:
            latency_ms: Latency in milliseconds
        """
        self.latencies.append(latency_ms)

    def add_latencies(self, latencies_ms: List[float]) -> None:
        """
        Add multiple latency samples.

        Args:
            latencies_ms: List of latencies in milliseconds
        """
        self.latencies.extend(latencies_ms)

    def add_resource_sample(self, cpu_percent: float, memory_mb: float) -> None:
        """
        Add a resource usage sample.

        Args:
            cpu_percent: CPU usage percentage
            memory_mb: Memory usage in MB
        """
        self.cpu_usage.append(cpu_percent)
        self.memory_usage.append(memory_mb)

    def set_model_info(self, info: Dict) -> None:
        """
        Set model information.

        Args:
            info: Dictionary with model metadata
        """
        self.model_info = info.copy()

    def set_conversion_info(self, info: Dict) -> None:
        """
        Set model conversion information.

        Args:
            info: Dictionary with conversion metadata
        """
        self.conversion_info = info.copy()

    def set_timing_info(
        self, total_time: float, num_samples: int, num_batches: int, batch_size: int = 1
    ) -> None:
        """
        Set timing information.

        Args:
            total_time: Total execution time in seconds
            num_samples: Total number of samples processed
            num_batches: Total number of batches processed
            batch_size: Batch size used
        """
        self.total_time = total_time
        self.num_samples = num_samples
        self.num_batches = num_batches
        self.batch_size = batch_size

    def calculate_statistics(self) -> Dict[str, float]:
        """
        Calculate comprehensive statistics.

        Returns:
            Dictionary containing all statistics
        """
        if not self.latencies:
            return self._get_empty_stats()

        latencies_array = np.array(self.latencies)

        # Latency statistics (in milliseconds)
        latency_stats = {
            "latency_mean": float(np.mean(latencies_array)),
            "latency_median": float(np.median(latencies_array)),
            "latency_std": float(np.std(latencies_array)),
            "latency_min": float(np.min(latencies_array)),
            "latency_max": float(np.max(latencies_array)),
            "latency_p50": float(np.percentile(latencies_array, 50)),
            "latency_p95": float(np.percentile(latencies_array, 95)),
            "latency_p99": float(np.percentile(latencies_array, 99)),
            "latency_p999": float(np.percentile(latencies_array, 99.9)),
        }

        # Throughput statistics
        throughput_stats = {}
        if self.total_time > 0:
            throughput_stats["throughput_samples_per_sec"] = self.num_samples / self.total_time
            throughput_stats["throughput_batches_per_sec"] = self.num_batches / self.total_time

            # For text models: tokens per second (approximation)
            if self.batch_size > 1:
                throughput_stats["throughput_items_per_sec"] = (
                    self.num_samples * self.batch_size / self.total_time
                )

        # Resource statistics
        resource_stats = {}
        if self.cpu_usage:
            cpu_array = np.array(self.cpu_usage)
            resource_stats["cpu_mean"] = float(np.mean(cpu_array))
            resource_stats["cpu_max"] = float(np.max(cpu_array))
            resource_stats["cpu_std"] = float(np.std(cpu_array))

        if self.memory_usage:
            memory_array = np.array(self.memory_usage)
            resource_stats["memory_mean_mb"] = float(np.mean(memory_array))
            resource_stats["memory_peak_mb"] = float(np.max(memory_array))
            resource_stats["memory_std_mb"] = float(np.std(memory_array))

        # Combine all statistics
        stats = {
            **latency_stats,
            **throughput_stats,
            **resource_stats,
            "num_latency_samples": len(self.latencies),
            "total_time_sec": self.total_time,
        }

        return stats

    def calculate_confidence_interval(
        self, confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals for metrics.

        Args:
            confidence_level: Confidence level (default: 0.95 for 95% CI)

        Returns:
            Dictionary mapping metric names to (lower, upper) bounds
        """
        if not self.latencies:
            return {}

        from scipy import stats as scipy_stats

        latencies_array = np.array(self.latencies)

        # Calculate confidence interval for mean latency
        mean = np.mean(latencies_array)
        std_err = scipy_stats.sem(latencies_array)
        ci = scipy_stats.t.interval(
            confidence_level,
            len(latencies_array) - 1,
            loc=mean,
            scale=std_err,
        )

        intervals = {
            "latency_mean_ci": (float(ci[0]), float(ci[1])),
        }

        # Throughput confidence interval
        if self.total_time > 0:
            throughput = self.num_samples / self.total_time
            throughput_std = np.std(1000.0 / latencies_array)  # Samples per second
            throughput_err = scipy_stats.sem(1000.0 / latencies_array)
            throughput_ci = scipy_stats.t.interval(
                confidence_level,
                len(latencies_array) - 1,
                loc=throughput,
                scale=throughput_err,
            )
            intervals["throughput_ci"] = (float(throughput_ci[0]), float(throughput_ci[1]))

        return intervals

    def detect_outliers(self, method: str = "iqr", threshold: float = 1.5) -> List[int]:
        """
        Detect outlier indices in latency measurements.

        Args:
            method: Outlier detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection (1.5 for IQR, 3.0 for z-score)

        Returns:
            List of indices corresponding to outliers
        """
        if not self.latencies:
            return []

        latencies_array = np.array(self.latencies)
        outlier_indices = []

        if method == "iqr":
            # Interquartile Range method
            q1 = np.percentile(latencies_array, 25)
            q3 = np.percentile(latencies_array, 75)
            iqr = q3 - q1

            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            outlier_indices = [
                i for i, val in enumerate(latencies_array)
                if val < lower_bound or val > upper_bound
            ]

        elif method == "zscore":
            # Z-score method
            mean = np.mean(latencies_array)
            std = np.std(latencies_array)

            if std > 0:
                z_scores = np.abs((latencies_array - mean) / std)
                outlier_indices = [
                    i for i, z in enumerate(z_scores) if z > threshold
                ]

        return outlier_indices

    def to_dict(self) -> Dict:
        """
        Export all metrics as a dictionary.

        Returns:
            Dictionary containing all metrics and metadata
        """
        result = {
            "statistics": self.calculate_statistics(),
            "model_info": self.model_info.copy(),
            "conversion_info": self.conversion_info.copy(),
            "num_latency_samples": len(self.latencies),
            "num_resource_samples": len(self.cpu_usage),
        }

        # Add confidence intervals if scipy is available
        try:
            result["confidence_intervals"] = self.calculate_confidence_interval()
        except ImportError:
            pass  # scipy not available

        # Add outlier information
        outlier_indices = self.detect_outliers()
        result["outlier_count"] = len(outlier_indices)
        result["outlier_percentage"] = (
            len(outlier_indices) / len(self.latencies) * 100 if self.latencies else 0.0
        )

        return result

    def reset(self) -> None:
        """Reset all collected metrics."""
        self.latencies.clear()
        self.cpu_usage.clear()
        self.memory_usage.clear()
        self.model_info.clear()
        self.conversion_info.clear()
        self.total_time = 0.0
        self.num_samples = 0
        self.num_batches = 0
        self.batch_size = 1

    def _get_empty_stats(self) -> Dict[str, float]:
        """Get empty statistics dictionary."""
        return {
            "latency_mean": 0.0,
            "latency_median": 0.0,
            "latency_std": 0.0,
            "latency_min": 0.0,
            "latency_max": 0.0,
            "latency_p50": 0.0,
            "latency_p95": 0.0,
            "latency_p99": 0.0,
            "latency_p999": 0.0,
            "num_latency_samples": 0,
            "total_time_sec": 0.0,
        }

    def __repr__(self) -> str:
        """String representation of the collector."""
        return (
            f"MetricsCollector(latency_samples={len(self.latencies)}, "
            f"resource_samples={len(self.cpu_usage)})"
        )
