"""
Benchmark Visualizer for TensorFlow Benchmark.

This module generates visualizations for benchmark results.
Simplified version for Phase 5.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class BenchmarkVisualizer:
    """
    Benchmark visualization generator.

    Creates charts and plots for benchmark analysis.
    """

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize BenchmarkVisualizer.

        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")

        self.color_palette = sns.color_palette("husl", 8)
        sns.set_palette(self.color_palette)

    def plot_throughput_comparison(self, data: pd.DataFrame, output_path: Path) -> None:
        """
        Generate throughput comparison bar chart.

        Args:
            data: Benchmark results DataFrame
            output_path: Path to save plot
        """
        plt.figure(figsize=(12, 6))

        # Extract throughput data
        engines = data["engine"].tolist()
        configs = data["config"].tolist()
        labels = [f"{e}_{c}" for e, c in zip(engines, configs)]

        throughputs = [
            row["statistics"].get("throughput_samples_per_sec", 0) for _, row in data.iterrows()
        ]

        # Create bar chart
        plt.bar(range(len(labels)), throughputs, color=self.color_palette[0])
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("Throughput (samples/sec)")
        plt.title("Throughput Comparison Across Engines")
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved throughput chart to {output_path}")

    def plot_latency_boxplot(self, data: pd.DataFrame, output_path: Path) -> None:
        """
        Generate latency distribution box plot.

        Args:
            data: Benchmark results DataFrame
            output_path: Path to save plot
        """
        plt.figure(figsize=(12, 6))

        # Prepare data for box plot
        latency_data = []
        labels = []

        for _, row in data.iterrows():
            stats = row["statistics"]
            engine = row["engine"]
            config = row["config"]

            # Create synthetic distribution from percentiles
            p50 = stats.get("latency_p50", 0)
            p95 = stats.get("latency_p95", 0)
            p99 = stats.get("latency_p99", 0)

            # Approximate distribution
            latency_data.append([p50 * 0.9, p50, p95, p99, p99 * 1.1])
            labels.append(f"{engine}_{config}")

        # Create box plot
        plt.boxplot(latency_data, labels=labels)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Latency (ms)")
        plt.title("Latency Distribution (P50/P95/P99)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved latency boxplot to {output_path}")

    def plot_resource_usage(self, data: pd.DataFrame, output_path: Path) -> None:
        """
        Generate resource usage chart.

        Args:
            data: Benchmark results DataFrame
            output_path: Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        engines = data["engine"].tolist()
        configs = data["config"].tolist()
        labels = [f"{e}_{c}" for e, c in zip(engines, configs)]

        # CPU usage
        cpu_usage = [row.get("resource_usage", {}).get("cpu_mean", 0) for _, row in data.iterrows()]

        ax1.bar(range(len(labels)), cpu_usage, color=self.color_palette[1])
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax1.set_ylabel("CPU Usage (%)")
        ax1.set_title("Average CPU Usage")
        ax1.grid(True, alpha=0.3)

        # Memory usage
        memory_usage = [
            row.get("resource_usage", {}).get("memory_mean_mb", 0) for _, row in data.iterrows()
        ]

        ax2.bar(range(len(labels)), memory_usage, color=self.color_palette[2])
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha="right")
        ax2.set_ylabel("Memory (MB)")
        ax2.set_title("Average Memory Usage")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved resource usage chart to {output_path}")

    def generate_all_plots(self, data: pd.DataFrame, output_dir: Path) -> List[Path]:
        """
        Generate all visualization plots.

        Args:
            data: Benchmark results DataFrame
            output_dir: Directory to save plots

        Returns:
            List of paths to generated plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_plots = []

        # Generate throughput comparison
        throughput_path = output_dir / "throughput_comparison.png"
        self.plot_throughput_comparison(data, throughput_path)
        generated_plots.append(throughput_path)

        # Generate latency boxplot
        latency_path = output_dir / "latency_boxplot.png"
        self.plot_latency_boxplot(data, latency_path)
        generated_plots.append(latency_path)

        # Generate resource usage
        resource_path = output_dir / "resource_usage.png"
        self.plot_resource_usage(data, resource_path)
        generated_plots.append(resource_path)

        print(f"\n✓ Generated {len(generated_plots)} plots in {output_dir}")
        return generated_plots

    def __repr__(self) -> str:
        """String representation."""
        return "BenchmarkVisualizer(style='seaborn')"
