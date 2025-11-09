#!/usr/bin/env python3
"""
Standalone Report Generation Script.

This script generates benchmark reports from existing results.
Can be run independently without re-running benchmarks.
"""

import sys
from pathlib import Path

import click

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reporting import DataProcessor, ReportGenerator


@click.command()
@click.option(
    "--results-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing benchmark results",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./reports",
    help="Output directory for reports",
)
@click.option(
    "--format",
    type=click.Choice(["html", "markdown", "both"]),
    default="both",
    help="Report format to generate",
)
def main(results_dir, output_dir, format):
    """Generate benchmark report from existing results."""

    print("\n" + "=" * 70)
    print("TensorFlow Benchmark - Report Generator")
    print("=" * 70)

    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    print(f"\n📊 Loading results from: {results_dir}")
    processor = DataProcessor()

    try:
        data = processor.load_results(results_dir)
    except Exception as e:
        print(f"✗ Error loading results: {e}")
        return 1

    # Step 2: Generate reports
    print(f"\n📝 Generating reports...")
    generator = ReportGenerator()

    try:
        if format in ["html", "both"]:
            html_path = output_dir / "report.html"
            generator.generate_html_report(data, [], html_path)

        if format in ["markdown", "both"]:
            md_path = output_dir / "report.md"
            generator.generate_markdown_report(data, [], md_path)

        # Always generate recommendations
        rec_path = output_dir / "recommendations.txt"
        generator.generate_recommendations(data, rec_path)

    except Exception as e:
        print(f"✗ Error generating reports: {e}")
        return 1

    # Step 3: Export summary CSV
    print(f"\n💾 Exporting summary...")
    try:
        csv_path = output_dir / "summary.csv"
        processor.export_summary_csv(csv_path)
    except Exception as e:
        print(f"⚠️  Warning: Could not export CSV: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("✓ Report Generation Complete!")
    print("=" * 70)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   - HTML report: report.html")
    print(f"   - Markdown report: report.md")
    print(f"   - Recommendations: recommendations.txt")
    print(f"   - Summary CSV: summary.csv")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
