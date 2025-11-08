"""
Data Processor for TensorFlow Benchmark.

This module processes benchmark results for analysis and visualization.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class DataProcessor:
    """
    Benchmark data processor.

    Loads, cleans, and processes benchmark results for reporting.
    """

    def __init__(self):
        """Initialize DataProcessor."""
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None

    def load_results(self, results_path: str) -> pd.DataFrame:
        """
        Load benchmark results from file.

        Args:
            results_path: Path to results file (JSON or directory)

        Returns:
            DataFrame with loaded results
        """
        results_path = Path(results_path)

        if results_path.is_file() and results_path.suffix == ".json":
            # Load from JSON file
            with open(results_path, "r") as f:
                data = json.load(f)
            self.raw_data = pd.DataFrame(data)

        elif results_path.is_dir():
            # Load from directory (find results.json)
            json_file = results_path / "results.json"
            if json_file.exists():
                with open(json_file, "r") as f:
                    data = json.load(f)
                self.raw_data = pd.DataFrame(data)
            else:
                raise FileNotFoundError(f"No results.json found in {results_path}")

        else:
            raise ValueError(f"Invalid results path: {results_path}")

        print(f"✓ Loaded {len(self.raw_data)} benchmark results")
        return self.raw_data

    def aggregate_by_engine(self) -> pd.DataFrame:
        """
        Aggregate results by engine.

        Returns:
            DataFrame aggregated by engine
        """
        if self.raw_data is None:
            raise RuntimeError("No data loaded. Call load_results() first.")

        # Group by engine and calculate mean statistics
        grouped = self.raw_data.groupby("engine").agg({
            "statistics": "mean",
        }).reset_index()

        return grouped

    def aggregate_by_model(self) -> pd.DataFrame:
        """
        Aggregate results by model.

        Returns:
            DataFrame aggregated by model
        """
        if self.raw_data is None:
            raise RuntimeError("No data loaded. Call load_results() first.")

        grouped = self.raw_data.groupby("model").agg({
            "statistics": "mean",
        }).reset_index()

        return grouped

    def calculate_speedup(self, baseline: str = "tensorflow_baseline") -> pd.DataFrame:
        """
        Calculate speedup relative to baseline.

        Args:
            baseline: Baseline configuration name

        Returns:
            DataFrame with speedup column added
        """
        if self.raw_data is None:
            raise RuntimeError("No data loaded. Call load_results() first.")

        # Find baseline performance
        baseline_row = self.raw_data[self.raw_data["config"] == baseline]

        if baseline_row.empty:
            print(f"Warning: Baseline '{baseline}' not found")
            return self.raw_data

        baseline_throughput = baseline_row.iloc[0]["statistics"].get(
            "throughput_samples_per_sec", 1.0
        )

        # Calculate speedup for each row
        def calc_speedup(row):
            throughput = row["statistics"].get("throughput_samples_per_sec", 0)
            return throughput / baseline_throughput if baseline_throughput > 0 else 0

        self.raw_data["speedup"] = self.raw_data.apply(calc_speedup, axis=1)

        return self.raw_data

    def rank_by_metric(self, metric: str = "throughput_samples_per_sec") -> pd.DataFrame:
        """
        Rank results by a specific metric.

        Args:
            metric: Metric name to rank by

        Returns:
            DataFrame sorted by metric
        """
        if self.raw_data is None:
            raise RuntimeError("No data loaded. Call load_results() first.")

        # Extract metric from statistics
        def get_metric(row):
            return row["statistics"].get(metric, 0)

        self.raw_data["_sort_metric"] = self.raw_data.apply(get_metric, axis=1)
        ranked = self.raw_data.sort_values("_sort_metric", ascending=False)
        ranked = ranked.drop("_sort_metric", axis=1)

        return ranked

    def export_summary_csv(self, output_path: str) -> None:
        """
        Export summary statistics to CSV.

        Args:
            output_path: Path to save CSV file
        """
        if self.raw_data is None:
            raise RuntimeError("No data loaded. Call load_results() first.")

        # Flatten statistics into columns
        rows = []
        for _, row in self.raw_data.iterrows():
            flat_row = {
                "model": row.get("model"),
                "engine": row.get("engine"),
                "config": row.get("config"),
                "batch_size": row.get("batch_size"),
            }

            # Add statistics
            stats = row.get("statistics", {})
            for key, value in stats.items():
                flat_row[key] = value

            rows.append(flat_row)

        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(output_path, index=False)
        print(f"✓ Summary exported to {output_path}")

    def __repr__(self) -> str:
        """String representation."""
        num_rows = len(self.raw_data) if self.raw_data is not None else 0
        return f"DataProcessor(loaded_results={num_rows})"
