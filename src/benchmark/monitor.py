"""
Resource Monitor for TensorFlow Benchmark.

This module provides real-time monitoring of system resources
(CPU and memory usage) during benchmark execution.
"""

import threading
import time
from typing import Dict, List, Optional

import psutil


class ResourceMonitor:
    """
    Real-time resource monitor for CPU and memory usage.

    Runs in a background thread to collect resource usage statistics
    during benchmark execution without interfering with measurements.
    """

    def __init__(self, sampling_interval: float = 0.1):
        """
        Initialize ResourceMonitor.

        Args:
            sampling_interval: Sampling interval in seconds (default: 0.1)
        """
        self.sampling_interval = sampling_interval

        # Storage for samples
        self.cpu_samples: List[float] = []
        self.cpu_per_core_samples: List[List[float]] = []
        self.memory_samples: List[float] = []

        # Monitoring state
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Process handle
        self.process = psutil.Process()

    def start(self) -> None:
        """
        Start resource monitoring in background thread.

        Raises:
            RuntimeError: If monitor is already running
        """
        if self.running:
            raise RuntimeError("Monitor is already running")

        # Reset samples
        self.cpu_samples.clear()
        self.cpu_per_core_samples.clear()
        self.memory_samples.clear()

        # Start monitoring thread
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """
        Stop resource monitoring.

        Waits for monitoring thread to finish.
        """
        if not self.running:
            return

        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.running:
            try:
                # Get CPU usage (overall)
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_samples.append(cpu_percent)

                # Get per-core CPU usage
                per_core = psutil.cpu_percent(interval=None, percpu=True)
                self.cpu_per_core_samples.append(per_core)

                # Get memory usage (RSS - Resident Set Size)
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
                self.memory_samples.append(memory_mb)

                # Sleep until next sample
                time.sleep(self.sampling_interval)

            except Exception as e:
                print(f"Warning: Resource monitoring error: {e}")
                # Continue monitoring despite errors

    def get_stats(self) -> Dict[str, float]:
        """
        Get resource usage statistics.

        Returns:
            Dictionary containing:
                - cpu_mean: Mean CPU usage (%)
                - cpu_std: Std dev of CPU usage
                - cpu_min: Minimum CPU usage
                - cpu_max: Maximum CPU usage
                - cpu_median: Median CPU usage
                - memory_mean: Mean memory usage (MB)
                - memory_std: Std dev of memory usage
                - memory_min: Minimum memory usage
                - memory_max: Maximum memory usage
                - memory_peak: Peak memory usage
                - num_samples: Number of samples collected
        """
        if not self.cpu_samples or not self.memory_samples:
            return {
                "cpu_mean": 0.0,
                "cpu_std": 0.0,
                "cpu_min": 0.0,
                "cpu_max": 0.0,
                "cpu_median": 0.0,
                "memory_mean": 0.0,
                "memory_std": 0.0,
                "memory_min": 0.0,
                "memory_max": 0.0,
                "memory_peak": 0.0,
                "num_samples": 0,
            }

        import numpy as np

        cpu_array = np.array(self.cpu_samples)
        memory_array = np.array(self.memory_samples)

        stats = {
            # CPU statistics
            "cpu_mean": float(np.mean(cpu_array)),
            "cpu_std": float(np.std(cpu_array)),
            "cpu_min": float(np.min(cpu_array)),
            "cpu_max": float(np.max(cpu_array)),
            "cpu_median": float(np.median(cpu_array)),

            # Memory statistics
            "memory_mean": float(np.mean(memory_array)),
            "memory_std": float(np.std(memory_array)),
            "memory_min": float(np.min(memory_array)),
            "memory_max": float(np.max(memory_array)),
            "memory_peak": float(np.max(memory_array)),

            # Sample count
            "num_samples": len(self.cpu_samples),
        }

        # Add per-core CPU stats if available
        if self.cpu_per_core_samples:
            num_cores = len(self.cpu_per_core_samples[0])
            for core_idx in range(num_cores):
                core_samples = [sample[core_idx] for sample in self.cpu_per_core_samples]
                stats[f"cpu_core_{core_idx}_mean"] = float(np.mean(core_samples))

        return stats

    def reset(self) -> None:
        """Reset monitoring data (clears all samples)."""
        if self.running:
            raise RuntimeError("Cannot reset while monitoring is running. Stop first.")

        self.cpu_samples.clear()
        self.cpu_per_core_samples.clear()
        self.memory_samples.clear()

    def is_running(self) -> bool:
        """Check if monitoring is currently running."""
        return self.running

    def __enter__(self):
        """Context manager entry - start monitoring."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop monitoring."""
        self.stop()

    def __repr__(self) -> str:
        """String representation of the monitor."""
        status = "running" if self.running else "stopped"
        num_samples = len(self.cpu_samples)
        return (
            f"ResourceMonitor(status='{status}', "
            f"samples={num_samples}, "
            f"interval={self.sampling_interval}s)"
        )
