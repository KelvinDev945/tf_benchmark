"""
Report Generator for TensorFlow Benchmark.

This module generates HTML and Markdown reports from benchmark results.
Simplified version for Phase 5.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


class ReportGenerator:
    """
    Benchmark report generator.

    Creates HTML and Markdown reports from benchmark data.
    """

    def __init__(self):
        """Initialize ReportGenerator."""
        pass

    def generate_html_report(
        self, data: pd.DataFrame, plots: List[Path], output_path: Path
    ) -> None:
        """
        Generate HTML report.

        Args:
            data: Benchmark results DataFrame
            plots: List of plot file paths
            output_path: Path to save HTML report
        """
        # Simple HTML template
        html = self._create_html_template(data, plots)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(html)

        print(f"✓ HTML report saved to {output_path}")

    def generate_markdown_report(
        self, data: pd.DataFrame, plots: List[Path], output_path: Path
    ) -> None:
        """
        Generate Markdown report.

        Args:
            data: Benchmark results DataFrame
            plots: List of plot file paths
            output_path: Path to save Markdown report
        """
        md = self._create_markdown_template(data, plots)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(md)

        print(f"✓ Markdown report saved to {output_path}")

    def generate_recommendations(
        self, data: pd.DataFrame, output_path: Path
    ) -> None:
        """
        Generate configuration recommendations.

        Args:
            data: Benchmark results DataFrame
            output_path: Path to save recommendations
        """
        recommendations = self._generate_recommendations_text(data)

        output_path = Path(output_path)
        with open(output_path, "w") as f:
            f.write(recommendations)

        print(f"✓ Recommendations saved to {output_path}")

    def _create_html_template(
        self, data: pd.DataFrame, plots: List[Path]
    ) -> str:
        """Create HTML report template."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TensorFlow Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; border-radius: 4px; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>TensorFlow Multi-Engine CPU Benchmark Report</h1>
        <p>Generated: {timestamp}</p>

        <h2>Executive Summary</h2>
        <div class="metric">
            <div class="metric-value">{len(data)}</div>
            <div class="metric-label">Total Benchmarks</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(data['engine'].unique())}</div>
            <div class="metric-label">Engines Tested</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(data['model'].unique())}</div>
            <div class="metric-label">Models Tested</div>
        </div>

        <h2>Results Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Engine</th>
                <th>Config</th>
                <th>Throughput (samples/sec)</th>
                <th>Latency P50 (ms)</th>
            </tr>
"""

        for _, row in data.iterrows():
            stats = row.get("statistics", {})
            html += f"""            <tr>
                <td>{row.get('model', 'N/A')}</td>
                <td>{row.get('engine', 'N/A')}</td>
                <td>{row.get('config', 'N/A')}</td>
                <td>{stats.get('throughput_samples_per_sec', 0):.2f}</td>
                <td>{stats.get('latency_p50', 0):.2f}</td>
            </tr>
"""

        html += """        </table>

        <h2>Visualizations</h2>
"""

        for plot_path in plots:
            plot_name = plot_path.stem.replace("_", " ").title()
            rel_path = plot_path.name
            html += f"""        <h3>{plot_name}</h3>
        <img src="../plots/{rel_path}" alt="{plot_name}">
"""

        html += """    </div>
</body>
</html>
"""
        return html

    def _create_markdown_template(
        self, data: pd.DataFrame, plots: List[Path]
    ) -> str:
        """Create Markdown report template."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = f"""# TensorFlow Multi-Engine CPU Benchmark Report

**Generated**: {timestamp}

## Executive Summary

- **Total Benchmarks**: {len(data)}
- **Engines Tested**: {len(data['engine'].unique())}
- **Models Tested**: {len(data['model'].unique())}

## Results Summary

| Model | Engine | Config | Throughput (samples/sec) | Latency P50 (ms) |
|-------|--------|--------|--------------------------|------------------|
"""

        for _, row in data.iterrows():
            stats = row.get("statistics", {})
            md += f"| {row.get('model', 'N/A')} | {row.get('engine', 'N/A')} | {row.get('config', 'N/A')} | {stats.get('throughput_samples_per_sec', 0):.2f} | {stats.get('latency_p50', 0):.2f} |\n"

        md += "\n## Visualizations\n\n"

        for plot_path in plots:
            plot_name = plot_path.stem.replace("_", " ").title()
            md += f"### {plot_name}\n\n![{plot_name}](plots/{plot_path.name})\n\n"

        md += """
---

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)
"""
        return md

    def _generate_recommendations_text(self, data: pd.DataFrame) -> str:
        """Generate recommendations text."""
        # Find best configurations
        best_throughput = data.loc[
            data["statistics"].apply(lambda x: x.get("throughput_samples_per_sec", 0)).idxmax()
        ]
        best_latency = data.loc[
            data["statistics"].apply(lambda x: x.get("latency_p50", float("inf"))).idxmin()
        ]

        text = f"""TensorFlow Multi-Engine Benchmark - Configuration Recommendations
{'=' * 70}

Based on the benchmark results, here are the recommended configurations
for different use cases:

1. BEST THROUGHPUT (High-volume processing)
   Engine: {best_throughput.get('engine')}
   Config: {best_throughput.get('config')}
   Throughput: {best_throughput.get('statistics', {}).get('throughput_samples_per_sec', 0):.2f} samples/sec

2. LOWEST LATENCY (Real-time inference)
   Engine: {best_latency.get('engine')}
   Config: {best_latency.get('config')}
   Latency P50: {best_latency.get('statistics', {}).get('latency_p50', 0):.2f} ms

3. BALANCED (General purpose)
   Consider engines with good throughput-to-latency ratio
   Review the full report for detailed comparisons

{'=' * 70}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return text

    def __repr__(self) -> str:
        """String representation."""
        return "ReportGenerator()"
