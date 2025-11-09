"""
TensorFlow Multi-Engine CPU Inference Benchmark - Main Entry Point

Command-line interface for running benchmarks.
"""

import sys
from pathlib import Path

import click

if __package__ is None or __package__ == "":
    # Allow running as a script via `python src/main.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmark import BenchmarkRunner
    from config import ConfigLoader
else:
    from .benchmark import BenchmarkRunner
    from .config import ConfigLoader


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    default="configs/benchmark_config.yaml",
    help="Path to configuration file",
)
@click.option(
    "--mode",
    type=click.Choice(["quick", "standard", "full"]),
    default="standard",
    help="Benchmark mode",
)
@click.option(
    "--engines",
    type=str,
    help="Comma-separated engines (e.g., 'tensorflow,tflite')",
)
@click.option(
    "--models",
    type=str,
    help="Comma-separated models (e.g., 'mobilenet_v2')",
)
@click.option(
    "--output",
    type=click.Path(),
    default="./results",
    help="Output directory",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Verbose output",
)
def main(config, mode, engines, models, output, verbose):
    """TensorFlow Multi-Engine CPU Benchmark."""
    print("\n" + "=" * 70)
    print("TensorFlow Multi-Engine CPU Inference Benchmark")
    print("=" * 70)

    # Load config
    print(f"\n📋 Loading: {config}")
    try:
        loader = ConfigLoader(config)
        cfg = loader.get_mode_config(mode) if mode else loader.to_dict()
        cfg.setdefault("output", {})["results_dir"] = output
        print("✓ Config loaded")
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

    # Run benchmark
    print("\n🏃 Running benchmark...")
    try:
        runner = BenchmarkRunner(cfg)
        runner.run_all()
        runner.save_results(Path(output) / "results.json")
        print("\n✓ Completed!")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
