#!/usr/bin/env python3
"""
Consolidated Benchmark Report Generator

This script aggregates results from multiple benchmark runs and generates
a comprehensive comparison report.

Usage:
    python scripts/generate_consolidated_report.py --input-dir ./results/full_benchmark --output ./results/report
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


class ConsolidatedReportGenerator:
    """
    Generate consolidated reports from multiple benchmark results.
    """

    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize report generator.

        Args:
            input_dir: Directory containing benchmark results
            output_dir: Output directory for consolidated report
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.all_results = []

    def collect_results(self):
        """Collect all benchmark results from subdirectories."""
        print(f"Collecting results from: {self.input_dir}")

        # Find all JSON result files
        json_files = list(self.input_dir.rglob("*results.json"))

        print(f"Found {len(json_files)} result files")

        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    results = json.load(f)

                    # Handle both single result and list of results
                    if isinstance(results, dict):
                        results = [results]

                    self.all_results.extend(results)
                    print(f"  ✓ Loaded {len(results)} results from {json_file.name}")

            except Exception as e:
                print(f"  ✗ Failed to load {json_file}: {e}")

        print(f"\nTotal results collected: {len(self.all_results)}")

    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate summary table from all results.

        Returns:
            DataFrame with summary statistics
        """
        if not self.all_results:
            print("No results to generate summary from")
            return pd.DataFrame()

        # Extract key metrics
        summary_data = []

        for result in self.all_results:
            stats = result.get("statistics", {})

            row = {
                "Model": result.get("model", "unknown"),
                "Engine": result.get("engine", "unknown"),
                "Batch Size": result.get("batch_size", 1),
                "Latency Mean (ms)": stats.get("latency_mean", 0),
                "Latency P50 (ms)": stats.get("latency_p50", 0),
                "Latency P95 (ms)": stats.get("latency_p95", 0),
                "Latency P99 (ms)": stats.get("latency_p99", 0),
                "Throughput (samples/sec)": stats.get("throughput_samples_per_sec", 0),
                "CPU Mean (%)": stats.get("cpu_mean", 0),
                "Memory Peak (MB)": stats.get("memory_peak_mb", 0),
            }

            summary_data.append(row)

        df = pd.DataFrame(summary_data)

        # Sort by model and latency
        df = df.sort_values(["Model", "Latency Mean (ms)"])

        return df

    def generate_bert_comparison_table(self) -> pd.DataFrame:
        """
        Generate BERT-specific comparison table.

        Returns:
            DataFrame with BERT comparison
        """
        # Filter BERT results
        bert_results = [
            r
            for r in self.all_results
            if "bert" in r.get("model", "").lower()
        ]

        if not bert_results:
            print("No BERT results found")
            return pd.DataFrame()

        comparison_data = []

        for result in bert_results:
            stats = result.get("statistics", {})

            row = {
                "Model": result.get("model", "unknown"),
                "Engine Type": self._get_engine_type(result.get("engine", "")),
                "Configuration": result.get("engine", "unknown"),
                "Latency Mean (ms)": stats.get("latency_mean", 0),
                "Latency P95 (ms)": stats.get("latency_p95", 0),
                "Throughput (samples/sec)": stats.get("throughput_samples_per_sec", 0),
                "Model Size Category": self._get_model_size_category(result.get("engine", "")),
            }

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by model type and latency
        df = df.sort_values(["Model", "Latency Mean (ms)"])

        return df

    def _get_engine_type(self, engine_name: str) -> str:
        """Get engine type from engine name."""
        engine_lower = engine_name.lower()

        if "tensorflow" in engine_lower:
            return "TensorFlow"
        elif "tflite" in engine_lower:
            return "TFLite (Quantized)"
        elif "onnx" in engine_lower:
            return "ONNX Runtime"
        elif "openvino" in engine_lower:
            return "OpenVINO"
        else:
            return "Unknown"

    def _get_model_size_category(self, engine_name: str) -> str:
        """Get model size category from engine name."""
        engine_lower = engine_name.lower()

        if "int8" in engine_lower:
            return "INT8 Quantized"
        elif "float16" in engine_lower or "fp16" in engine_lower:
            return "Float16 Quantized"
        elif "dynamic" in engine_lower:
            return "Dynamic Range Quantized"
        else:
            return "Full Precision"

    def generate_markdown_report(self):
        """Generate comprehensive Markdown report."""
        report_file = self.output_dir / "consolidated_report.md"

        with open(report_file, "w") as f:
            f.write("# TensorFlow Multi-Engine CPU Inference Benchmark\n\n")
            f.write("## Consolidated Report\n\n")

            # Summary table
            f.write("## All Results Summary\n\n")
            summary_df = self.generate_summary_table()

            if not summary_df.empty:
                f.write(summary_df.to_markdown(index=False))
                f.write("\n\n")

            # BERT comparison
            f.write("## BERT Model Comparison\n\n")
            f.write("### Base Model vs Quantized vs ONNX\n\n")

            bert_df = self.generate_bert_comparison_table()

            if not bert_df.empty:
                f.write(bert_df.to_markdown(index=False))
                f.write("\n\n")

                # Calculate speedups
                f.write("### Speedup Analysis\n\n")

                # Group by model
                for model_name in bert_df["Model"].unique():
                    model_results = bert_df[bert_df["Model"] == model_name]

                    # Find baseline (TensorFlow)
                    baseline = model_results[model_results["Engine Type"] == "TensorFlow"]

                    if not baseline.empty:
                        baseline_latency = baseline.iloc[0]["Latency Mean (ms)"]

                        f.write(f"#### {model_name}\n\n")
                        f.write("| Configuration | Speedup vs Baseline |\n")
                        f.write("|---------------|---------------------|\n")

                        for _, row in model_results.iterrows():
                            speedup = baseline_latency / row["Latency Mean (ms)"]
                            f.write(f"| {row['Configuration']} | {speedup:.2f}x |\n")

                        f.write("\n")

            # Best configurations
            f.write("## Best Configurations\n\n")

            if not summary_df.empty:
                # Fastest overall
                fastest = summary_df.loc[summary_df["Latency Mean (ms)"].idxmin()]
                f.write("### Fastest Configuration (Lowest Latency)\n\n")
                f.write(f"- **Model**: {fastest['Model']}\n")
                f.write(f"- **Engine**: {fastest['Engine']}\n")
                f.write(f"- **Latency (mean)**: {fastest['Latency Mean (ms)']:.2f} ms\n")
                f.write(f"- **Throughput**: {fastest['Throughput (samples/sec)']:.2f} samples/sec\n\n")

                # Highest throughput
                highest_throughput = summary_df.loc[summary_df["Throughput (samples/sec)"].idxmax()]
                f.write("### Highest Throughput\n\n")
                f.write(f"- **Model**: {highest_throughput['Model']}\n")
                f.write(f"- **Engine**: {highest_throughput['Engine']}\n")
                f.write(f"- **Throughput**: {highest_throughput['Throughput (samples/sec)']:.2f} samples/sec\n")
                f.write(f"- **Latency (mean)**: {highest_throughput['Latency Mean (ms)']:.2f} ms\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("### For Production Deployment\n\n")

            if not bert_df.empty:
                # Find best BERT configuration
                best_bert = bert_df.loc[bert_df["Latency Mean (ms)"].idxmin()]

                f.write(f"**BERT Models**:\n")
                f.write(f"- Recommended configuration: {best_bert['Configuration']}\n")
                f.write(f"- Expected latency: {best_bert['Latency Mean (ms)']:.2f} ms\n")
                f.write(f"- Expected throughput: {best_bert['Throughput (samples/sec)']:.2f} samples/sec\n\n")

            if not summary_df.empty:
                # Image models recommendation
                image_results = summary_df[summary_df["Model"].str.contains("mobilenet|resnet|efficientnet", case=False, na=False)]

                if not image_results.empty:
                    best_image = image_results.loc[image_results["Latency Mean (ms)"].idxmin()]

                    f.write(f"**Image Models**:\n")
                    f.write(f"- Recommended configuration: {best_image['Model']} + {best_image['Engine']}\n")
                    f.write(f"- Expected latency: {best_image['Latency Mean (ms)']:.2f} ms\n")
                    f.write(f"- Expected throughput: {best_image['Throughput (samples/sec)']:.2f} samples/sec\n\n")

        print(f"✓ Markdown report saved to: {report_file}")

    def generate_csv_export(self):
        """Export all results to CSV."""
        csv_file = self.output_dir / "all_results.csv"

        summary_df = self.generate_summary_table()

        if not summary_df.empty:
            summary_df.to_csv(csv_file, index=False)
            print(f"✓ CSV export saved to: {csv_file}")

    def generate_bert_csv_export(self):
        """Export BERT comparison to CSV."""
        csv_file = self.output_dir / "bert_comparison.csv"

        bert_df = self.generate_bert_comparison_table()

        if not bert_df.empty:
            bert_df.to_csv(csv_file, index=False)
            print(f"✓ BERT comparison CSV saved to: {csv_file}")

    def run(self):
        """Run the complete report generation."""
        print("\n" + "=" * 70)
        print("Consolidated Report Generator")
        print("=" * 70)

        self.collect_results()

        if not self.all_results:
            print("\n✗ No results found to generate report")
            return

        print("\nGenerating reports...")

        self.generate_markdown_report()
        self.generate_csv_export()
        self.generate_bert_csv_export()

        print("\n" + "=" * 70)
        print("✓ Report Generation Complete!")
        print("=" * 70)
        print(f"\nReports saved to: {self.output_dir}")
        print(f"  - Markdown report: {self.output_dir / 'consolidated_report.md'}")
        print(f"  - CSV export: {self.output_dir / 'all_results.csv'}")
        print(f"  - BERT comparison: {self.output_dir / 'bert_comparison.csv'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate consolidated benchmark report"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/consolidated_report",
        help="Output directory for consolidated report",
    )

    args = parser.parse_args()

    # Generate report
    generator = ConsolidatedReportGenerator(
        input_dir=args.input_dir,
        output_dir=args.output,
    )

    generator.run()


if __name__ == "__main__":
    main()
