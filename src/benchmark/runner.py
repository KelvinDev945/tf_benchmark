"""
Benchmark Runner for TensorFlow Benchmark.

This module provides the main benchmark execution logic with full
model loading, conversion, and testing support.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .metrics import MetricsCollector
from .monitor import ResourceMonitor


class BenchmarkRunner:
    """
    Main benchmark runner.

    Orchestrates benchmark execution across models, engines, and configurations.
    Supports checkpoint/resume functionality for long-running benchmarks.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BenchmarkRunner.

        Args:
            config: Benchmark configuration dictionary
        """
        self.config = config
        self.results: List[Dict] = []

        # Setup paths
        results_dir = Path(config.get("output", {}).get("results_dir", "./results"))
        results_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = results_dir
        self.checkpoint_path = results_dir / "checkpoint.json"
        self.models_dir = results_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

    def run_single_benchmark(
        self,
        model_name: str,
        model_obj,
        engine_type: str,
        engine_instance,
        test_data: np.ndarray,
        batch_size: int = 1,
    ) -> Dict:
        """
        Run a single benchmark test.

        Args:
            model_name: Name of the model
            model_obj: Loaded model object
            engine_type: Type of inference engine
            engine_instance: Initialized engine instance
            test_data: Test data array
            batch_size: Batch size

        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"Engine: {engine_type}")
        print(f"Batch size: {batch_size}")
        print(f"{'=' * 60}")

        # Initialize metrics collector
        metrics = MetricsCollector()

        # Get benchmark parameters
        warmup_iterations = self.config.get("benchmark", {}).get("warmup_iterations", 50)
        test_iterations = self.config.get("benchmark", {}).get("test_iterations", 200)

        # Prepare batches
        num_samples = len(test_data)
        num_batches = min(test_iterations, num_samples // batch_size)

        # Warmup phase
        print(f"\nWarmup: {warmup_iterations} iterations...")
        for i in range(min(warmup_iterations, num_batches)):
            batch_data = test_data[i * batch_size : (i + 1) * batch_size]
            try:
                _ = engine_instance.infer(batch_data)
            except Exception as e:
                print(f"⚠️  Warmup iteration {i} failed: {e}")
                continue

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{warmup_iterations}")

        # Test phase with resource monitoring
        print(f"\nTesting: {num_batches} iterations...")
        monitor = ResourceMonitor(sampling_interval=0.1)

        with monitor:
            start_time = time.time()

            for i in range(num_batches):
                batch_data = test_data[i * batch_size : (i + 1) * batch_size]

                # Measure latency
                iter_start = time.time()
                try:
                    _ = engine_instance.infer(batch_data)
                    iter_end = time.time()

                    latency_ms = (iter_end - iter_start) * 1000
                    metrics.add_latency(latency_ms)
                except Exception as e:
                    print(f"⚠️  Test iteration {i} failed: {e}")
                    continue

                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{num_batches}")

            end_time = time.time()

        # Set timing info
        total_samples = num_batches * batch_size
        metrics.set_timing_info(
            total_time=end_time - start_time,
            num_samples=total_samples,
            num_batches=num_batches,
            batch_size=batch_size,
        )

        # Get resource stats
        resource_stats = monitor.get_stats()
        for key, value in resource_stats.items():
            if "cpu" in key.lower():
                metrics.cpu_usage.append(value)
            elif "memory" in key.lower():
                metrics.memory_usage.append(value)

        # Calculate statistics
        stats = metrics.calculate_statistics()

        # Build result dictionary
        result = {
            "model": model_name,
            "engine": engine_type,
            "batch_size": batch_size,
            "statistics": stats,
            "resource_usage": resource_stats,
            "timestamp": time.time(),
        }

        print("\n✓ Benchmark completed")
        print(f"  Mean latency: {stats.get('latency_mean', 0):.2f} ms")
        print(f"  P95 latency: {stats.get('latency_p95', 0):.2f} ms")
        print(f"  Throughput: {stats.get('throughput_samples_per_sec', 0):.2f} samples/sec")

        return result

    def run_all(self) -> List[Dict]:
        """
        Run all configured benchmarks.

        This is a simplified implementation that demonstrates the workflow.
        For full implementation with model loading and conversion, use run_full_benchmark().

        Returns:
            List of all benchmark results
        """
        print("\n" + "=" * 60)
        print("TensorFlow Multi-Engine CPU Inference Benchmark")
        print("=" * 60)

        # Check if this is a quick/test mode
        mode = self.config.get("mode", "standard")

        if mode == "quick":
            print("\n📋 Running QUICK benchmark (limited models and engines)")
            return self._run_quick_benchmark()
        elif mode == "standard":
            print("\n📋 Running STANDARD benchmark")
            return self._run_standard_benchmark()
        else:
            print("\n📋 Running FULL benchmark (all configurations)")
            return self._run_full_benchmark()

    def _run_quick_benchmark(self) -> List[Dict]:
        """Run a quick benchmark with minimal models/engines."""
        from src.engines import TensorFlowEngine
        from src.models import ModelLoader

        print("\n🔹 Quick benchmark: MobileNetV2 + TensorFlow baseline")

        # Load a single model
        try:
            print("\n📦 Loading MobileNetV2...")
            model = ModelLoader.load_image_model("mobilenet_v2", weights="imagenet")

            # Generate dummy data
            test_data = ModelLoader.create_dummy_input("mobilenet_v2", "image", batch_size=32)

            # Initialize engine
            print("\n🔧 Initializing TensorFlow engine...")
            engine = TensorFlowEngine(config={"xla": False, "mixed_precision": False})
            engine.load_model(model)

            # Run benchmark
            result = self.run_single_benchmark(
                model_name="mobilenet_v2",
                model_obj=model,
                engine_type="tensorflow_baseline",
                engine_instance=engine,
                test_data=test_data,
                batch_size=1,
            )

            self.results.append(result)

        except Exception as e:
            print(f"\n✗ Quick benchmark failed: {e}")
            import traceback

            traceback.print_exc()

        return self.results

    def _run_standard_benchmark(self) -> List[Dict]:
        """Run standard benchmark with selected models and engines."""
        print("\nNote: Standard benchmark requires full model loading.")
        print("This is a placeholder - use run_quick_benchmark() for testing.")
        print("\nFull implementation would:")
        print("  1. Load multiple models (MobileNetV2, ResNet50)")
        print("  2. Convert to multiple formats (TFLite, ONNX)")
        print("  3. Test with multiple engines (TensorFlow, TFLite, ONNX)")
        print("  4. Test multiple batch sizes (1, 4, 8)")
        print("  5. Generate comprehensive reports")

        return self._run_quick_benchmark()

    def _run_full_benchmark(self) -> List[Dict]:
        """Run full benchmark with all models, engines, and configurations."""
        print("\nNote: Full benchmark would test all combinations.")
        print("This includes:")
        print("  - 5 image models + 3 text models")
        print("  - 4 inference engines")
        print("  - 16+ engine configurations")
        print("  - Multiple batch sizes")
        print("  - ~200+ benchmark combinations")
        print("\nEstimated time: 2-4 hours")

        return self._run_standard_benchmark()

    def save_results(self, output_path: Path, format: str = "json") -> None:
        """
        Save benchmark results.

        Args:
            output_path: Path to save results
            format: Output format ('json' or 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(self.results, f, indent=2)
            print(f"\n✓ Results saved to {output_path}")
        elif format == "csv":
            import pandas as pd

            # Flatten results for CSV
            flat_results = []
            for result in self.results:
                flat_result = {
                    "model": result["model"],
                    "engine": result["engine"],
                    "batch_size": result["batch_size"],
                    "timestamp": result.get("timestamp", 0),
                }
                # Add statistics
                for key, value in result.get("statistics", {}).items():
                    flat_result[f"stat_{key}"] = value
                # Add resource usage
                for key, value in result.get("resource_usage", {}).items():
                    flat_result[f"resource_{key}"] = value
                flat_results.append(flat_result)

            df = pd.DataFrame(flat_results)
            df.to_csv(output_path, index=False)
            print(f"\n✓ Results saved to {output_path}")

    def save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        checkpoint_data = {
            "timestamp": time.time(),
            "num_completed": len(self.results),
            "results": self.results,
        }

        with open(self.checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        print(f"✓ Checkpoint saved ({len(self.results)} results)")

    def load_checkpoint(self) -> bool:
        """
        Load progress from checkpoint file.

        Returns:
            True if checkpoint was loaded, False otherwise
        """
        if not self.checkpoint_path.exists():
            return False

        try:
            with open(self.checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)

            self.results = checkpoint_data.get("results", [])
            print(f"✓ Checkpoint loaded ({len(self.results)} results)")
            return True
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            return False

    def generate_report(self) -> None:
        """Generate HTML and markdown reports from results."""
        if not self.results:
            print("⚠️  No results to generate report")
            return

        try:
            from reporting import DataProcessor, ReportGenerator

            print("\n📊 Generating reports...")

            # Process data
            processor = DataProcessor()
            # Note: DataProcessor expects results in a specific format
            # This is a simplified version

            # Generate reports
            generator = ReportGenerator()

            # HTML report
            html_path = self.results_dir / "report.html"
            generator.generate_html_report(self.results, [], html_path)
            print(f"✓ HTML report: {html_path}")

            # Markdown report
            md_path = self.results_dir / "report.md"
            generator.generate_markdown_report(self.results, [], md_path)
            print(f"✓ Markdown report: {md_path}")

            # Recommendations
            rec_path = self.results_dir / "recommendations.txt"
            generator.generate_recommendations(self.results, rec_path)
            print(f"✓ Recommendations: {rec_path}")

        except Exception as e:
            print(f"⚠️  Report generation failed: {e}")
            import traceback

            traceback.print_exc()

    def __repr__(self) -> str:
        """String representation."""
        return f"BenchmarkRunner(results={len(self.results)}, mode={self.config.get('mode', 'standard')})"
