#!/usr/bin/env python3
"""
BERT Model Inference Speed Comparison Script

This script compares inference speed across:
1. Base TensorFlow model (baseline)
2. Quantized TFLite models (INT8, Float16)
3. ONNX Runtime models (default, optimized, quantized)

Usage:
    python scripts/benchmark_bert_comparison.py --mode quick
    python scripts/benchmark_bert_comparison.py --mode standard
    python scripts/benchmark_bert_comparison.py --model bert-base-uncased
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import ConfigLoader
from dataset.text_dataset import TextDatasetLoader
from engines.onnx_engine import ONNXEngine
from engines.tensorflow_engine import TensorFlowEngine
from engines.tflite_engine import TFLiteEngine
from models.model_converter import ModelConverter


class BERTBenchmarkComparison:
    """
    BERT model inference speed comparison framework.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        mode: str = "quick",
        output_dir: str = "./results/bert_comparison",
    ):
        """
        Initialize BERT benchmark comparison.

        Args:
            model_name: Name of BERT model to test
            mode: Testing mode (quick, standard, full)
            output_dir: Output directory for results
        """
        self.model_name = model_name
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        config_path = Path(__file__).parent.parent / "configs" / "benchmark_config.yaml"
        loader = ConfigLoader(str(config_path))
        self.config = loader.get_mode_config(mode)

        # Results storage
        self.results: List[Dict] = []

        # Model directory
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        print(f"\n{'='*70}")
        print("BERT Model Inference Speed Comparison")
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"Mode: {mode}")
        print(f"Output: {output_dir}")
        print(f"{'='*70}\n")

    def load_dataset(self, num_samples: int = 100) -> np.ndarray:
        """
        Load text dataset for testing.

        Args:
            num_samples: Number of samples to load

        Returns:
            Test dataset
        """
        print("\n📊 Loading dataset...")

        dataset_config = self.config.get("dataset", {}).get("text", {})
        dataset_loader = TextDatasetLoader(
            dataset_name=dataset_config.get("name", "glue"),
            subset=dataset_config.get("subset", "sst2"),
            split=dataset_config.get("split", "validation"),
            tokenizer=self.model_name,
            max_length=128,  # Use fixed length for comparison
            num_samples=num_samples,
        )

        # Load and tokenize data
        dataset_loader.load()
        test_data = dataset_loader.get_numpy_batches(
            batch_size=1,
            max_length=128,
            max_batches=num_samples,
        )

        print(f"✓ Loaded {len(test_data['input_ids'])} samples")
        return test_data

    def benchmark_tensorflow_baseline(
        self, test_data: Dict[str, np.ndarray], batch_size: int = 1
    ) -> Dict:
        """
        Benchmark base TensorFlow model (no optimizations).

        Args:
            test_data: Test dataset
            batch_size: Batch size for inference

        Returns:
            Benchmark results
        """
        print(f"\n{'='*70}")
        print("1. TensorFlow Baseline (Base Model)")
        print(f"{'='*70}")

        # Load base TensorFlow model (h5 format, no PyTorch needed)
        print("Loading TensorFlow model (h5 format)...")
        from transformers import TFBertForSequenceClassification

        model = TFBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            from_pt=False,
            use_safetensors=False,  # 使用h5格式的TensorFlow权重
        )
        print(f"✓ Loaded model with {model.count_params():,} parameters")

        # Initialize TensorFlow engine with baseline config
        engine = TensorFlowEngine(config={"xla": False, "mixed_precision": False})
        engine.load_model(model)

        # Run benchmark
        results = self._run_benchmark(
            engine_name="tensorflow_baseline",
            engine=engine,
            test_data=test_data,
            batch_size=batch_size,
        )

        return results

    def benchmark_tflite_quantized(
        self, test_data: Dict[str, np.ndarray], batch_size: int = 1
    ) -> List[Dict]:
        """
        Benchmark quantized TFLite models.

        Args:
            test_data: Test dataset
            batch_size: Batch size for inference

        Returns:
            List of benchmark results for each quantization mode
        """
        results = []
        quantization_modes = ["int8", "float16", "dynamic_range"]

        # Load base TensorFlow model (h5 format, no PyTorch needed)
        print("\nLoading TensorFlow model for conversion (h5 format)...")
        from transformers import TFBertForSequenceClassification

        base_model = TFBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            from_pt=False,
            use_safetensors=False,  # 使用h5格式的TensorFlow权重
        )
        print(f"✓ Loaded model with {base_model.count_params():,} parameters")

        for quant_mode in quantization_modes:
            print(f"\n{'='*70}")
            print(f"2. TFLite {quant_mode.upper()} (Quantized Model)")
            print(f"{'='*70}")

            # Convert to TFLite
            tflite_path = self.models_dir / f"{self.model_name}_{quant_mode}.tflite"

            if not tflite_path.exists():
                print(f"Converting to TFLite with {quant_mode} quantization...")

                # Prepare calibration data for INT8
                calibration_data = None
                if quant_mode == "int8":
                    # Use subset of test data for calibration
                    calibration_data = self._prepare_calibration_data(test_data, num_samples=100)

                tflite_model, metadata = ModelConverter.to_tflite(
                    model=base_model,
                    optimization=quant_mode.upper(),
                    output_path=str(tflite_path),
                    calibration_data=calibration_data,
                )
                print(f"✓ Converted to TFLite ({quant_mode})")
                print(f"  Model size: {metadata.get('model_size_mb', 0):.2f} MB")
            else:
                print(f"✓ Using cached TFLite model: {tflite_path}")

            # Initialize TFLite engine
            engine = TFLiteEngine(config={"num_threads": 4})
            engine.load_model(str(tflite_path))

            # Run benchmark
            result = self._run_benchmark(
                engine_name=f"tflite_{quant_mode}",
                engine=engine,
                test_data=test_data,
                batch_size=batch_size,
            )

            results.append(result)

        return results

    def benchmark_onnx_runtime(
        self, test_data: Dict[str, np.ndarray], batch_size: int = 1
    ) -> List[Dict]:
        """
        Benchmark ONNX Runtime models.

        Args:
            test_data: Test dataset
            batch_size: Batch size for inference

        Returns:
            List of benchmark results for each ONNX configuration
        """
        results = []
        onnx_configs = [
            {"name": "default", "graph_optimization_level": "ENABLE_BASIC"},
            {"name": "optimized", "graph_optimization_level": "ENABLE_ALL"},
        ]

        # Load base TensorFlow model (h5 format, no PyTorch needed)
        print("\nLoading TensorFlow model for conversion (h5 format)...")
        from transformers import TFBertForSequenceClassification

        base_model = TFBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            from_pt=False,
            use_safetensors=False,  # 使用h5格式的TensorFlow权重
        )
        print(f"✓ Loaded model with {base_model.count_params():,} parameters")

        # Convert to ONNX
        onnx_path = self.models_dir / f"{self.model_name}.onnx"

        if not onnx_path.exists():
            print("Converting to ONNX...")
            metadata = ModelConverter.to_onnx(
                model=base_model,
                output_path=str(onnx_path),
                opset=13,
            )
            print("✓ Converted to ONNX")
            print(f"  Model size: {metadata.get('model_size_mb', 0):.2f} MB")
        else:
            print(f"✓ Using cached ONNX model: {onnx_path}")

        for onnx_config in onnx_configs:
            config_name = onnx_config["name"]

            print(f"\n{'='*70}")
            print(f"3. ONNX Runtime {config_name.upper()} (ONNX Model)")
            print(f"{'='*70}")

            # Initialize ONNX engine
            engine = ONNXEngine(
                config={
                    "graph_optimization_level": onnx_config["graph_optimization_level"],
                    "inter_op_num_threads": 4,
                    "intra_op_num_threads": 8,
                }
            )
            engine.load_model(str(onnx_path))

            # Run benchmark
            result = self._run_benchmark(
                engine_name=f"onnx_{config_name}",
                engine=engine,
                test_data=test_data,
                batch_size=batch_size,
            )

            results.append(result)

        return results

    def _prepare_calibration_data(self, test_data: Dict[str, np.ndarray], num_samples: int = 100):
        """
        Prepare calibration data for quantization.

        Args:
            test_data: Full test dataset
            num_samples: Number of calibration samples

        Returns:
            Calibration data generator
        """

        def calibration_data_gen():
            for i in range(min(num_samples, len(test_data["input_ids"]))):
                # Create batch with required inputs
                yield [
                    test_data["input_ids"][i : i + 1],
                    test_data["attention_mask"][i : i + 1],
                    test_data.get("token_type_ids", np.zeros_like(test_data["input_ids"]))[
                        i : i + 1
                    ],
                ]

        return calibration_data_gen

    def _run_benchmark(
        self,
        engine_name: str,
        engine,
        test_data: Dict[str, np.ndarray],
        batch_size: int = 1,
    ) -> Dict:
        """
        Run benchmark for a specific engine configuration.

        Args:
            engine_name: Name of the engine configuration
            engine: Initialized engine instance
            test_data: Test dataset
            batch_size: Batch size for inference

        Returns:
            Benchmark results dictionary
        """
        warmup_iterations = self.config.get("benchmark", {}).get("warmup_iterations", 10)
        test_iterations = self.config.get("benchmark", {}).get("test_iterations", 50)

        num_samples = len(test_data["input_ids"])
        num_batches = min(test_iterations, num_samples // batch_size)

        print(f"\nWarmup: {warmup_iterations} iterations...")
        for i in range(min(warmup_iterations, num_batches)):
            batch_data = {
                key: value[i * batch_size : (i + 1) * batch_size]
                for key, value in test_data.items()
            }
            try:
                _ = engine.infer(batch_data)
            except Exception as e:
                print(f"⚠️  Warmup iteration {i} failed: {e}")
                continue

        # Test phase
        print(f"Testing: {num_batches} iterations...")
        latencies = []
        start_time = time.time()

        for i in range(num_batches):
            batch_data = {
                key: value[i * batch_size : (i + 1) * batch_size]
                for key, value in test_data.items()
            }

            iter_start = time.time()
            try:
                _ = engine.infer(batch_data)
                iter_end = time.time()
                latency_ms = (iter_end - iter_start) * 1000
                latencies.append(latency_ms)
            except Exception as e:
                print(f"⚠️  Test iteration {i} failed: {e}")
                continue

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{num_batches}")

        end_time = time.time()

        # Calculate statistics
        latencies_np = np.array(latencies)
        total_time = end_time - start_time
        total_samples = num_batches * batch_size

        results = {
            "model": self.model_name,
            "engine": engine_name,
            "batch_size": batch_size,
            "statistics": {
                "latency_mean": float(np.mean(latencies_np)),
                "latency_median": float(np.median(latencies_np)),
                "latency_std": float(np.std(latencies_np)),
                "latency_min": float(np.min(latencies_np)),
                "latency_max": float(np.max(latencies_np)),
                "latency_p50": float(np.percentile(latencies_np, 50)),
                "latency_p95": float(np.percentile(latencies_np, 95)),
                "latency_p99": float(np.percentile(latencies_np, 99)),
                "throughput_samples_per_sec": total_samples / total_time,
                "throughput_batches_per_sec": num_batches / total_time,
                "num_iterations": len(latencies),
                "total_time_sec": total_time,
            },
            "timestamp": time.time(),
        }

        print("\n✓ Results:")
        print(f"  Latency (mean): {results['statistics']['latency_mean']:.2f} ms")
        print(f"  Latency (p50): {results['statistics']['latency_p50']:.2f} ms")
        print(f"  Latency (p95): {results['statistics']['latency_p95']:.2f} ms")
        print(
            f"  Throughput: {results['statistics']['throughput_samples_per_sec']:.2f} samples/sec"
        )

        return results

    def run_all_benchmarks(self, batch_size: int = 1) -> List[Dict]:
        """
        Run all benchmark tests.

        Args:
            batch_size: Batch size for inference

        Returns:
            List of all benchmark results
        """
        # Load dataset
        num_samples = self.config.get("dataset", {}).get("text", {}).get("num_samples", 100)
        test_data = self.load_dataset(num_samples=num_samples)

        all_results = []

        # 1. TensorFlow Baseline
        try:
            result = self.benchmark_tensorflow_baseline(test_data, batch_size)
            all_results.append(result)
        except Exception as e:
            print(f"✗ TensorFlow baseline benchmark failed: {e}")

        # 2. TFLite Quantized
        try:
            results = self.benchmark_tflite_quantized(test_data, batch_size)
            all_results.extend(results)
        except Exception as e:
            print(f"✗ TFLite quantized benchmark failed: {e}")

        # 3. ONNX Runtime
        try:
            results = self.benchmark_onnx_runtime(test_data, batch_size)
            all_results.extend(results)
        except Exception as e:
            print(f"✗ ONNX Runtime benchmark failed: {e}")

        self.results = all_results
        return all_results

    def save_results(self):
        """Save benchmark results to JSON file."""
        output_file = self.output_dir / "bert_comparison_results.json"

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

    def generate_comparison_report(self):
        """Generate a comparison report."""
        report_file = self.output_dir / "bert_comparison_report.md"

        with open(report_file, "w") as f:
            f.write("# BERT Model Inference Speed Comparison\n\n")
            f.write(f"**Model**: {self.model_name}\n")
            f.write(f"**Mode**: {self.mode}\n")
            f.write(f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if not self.results:
                f.write("## No Results\n\n")
                f.write("⚠️ All benchmarks failed. Please check the error messages above.\n")
                print(f"⚠️ Report saved (with no results): {report_file}")
                return

            f.write("## Results Summary\n\n")
            f.write("| Engine | Config | Latency (mean) | Latency (p95) | Throughput |\n")
            f.write("|--------|--------|----------------|---------------|------------|\n")

            for result in self.results:
                engine = result["engine"]
                stats = result["statistics"]
                f.write(
                    f"| {result['model']} | {engine} | "
                    f"{stats['latency_mean']:.2f} ms | "
                    f"{stats['latency_p95']:.2f} ms | "
                    f"{stats['throughput_samples_per_sec']:.2f} samples/sec |\n"
                )

            f.write("\n## Comparison Analysis\n\n")

            # Find fastest configuration
            fastest = min(self.results, key=lambda x: x["statistics"]["latency_mean"])
            f.write(f"**Fastest Configuration**: {fastest['engine']}\n")
            f.write(f"- Latency: {fastest['statistics']['latency_mean']:.2f} ms\n")
            f.write(
                f"- Throughput: {fastest['statistics']['throughput_samples_per_sec']:.2f} samples/sec\n\n"
            )

            # Calculate speedups
            baseline = next((r for r in self.results if "baseline" in r["engine"]), None)
            if baseline:
                f.write("### Speedup vs Baseline\n\n")
                baseline_latency = baseline["statistics"]["latency_mean"]

                for result in self.results:
                    if result != baseline:
                        speedup = baseline_latency / result["statistics"]["latency_mean"]
                        f.write(f"- {result['engine']}: {speedup:.2f}x\n")

        print(f"✓ Report saved to: {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BERT Model Inference Speed Comparison")
    parser.add_argument(
        "--model",
        type=str,
        default="google-bert/bert-base-uncased",
        choices=["bert-base-uncased", "google-bert/bert-base-uncased", "distilbert-base-uncased"],
        help="BERT model to benchmark",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "standard", "full"],
        help="Testing mode",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/bert_comparison",
        help="Output directory",
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = BERTBenchmarkComparison(
        model_name=args.model,
        mode=args.mode,
        output_dir=args.output,
    )

    results = benchmark.run_all_benchmarks(batch_size=args.batch_size)

    # Save results and generate report
    benchmark.save_results()
    benchmark.generate_comparison_report()

    print(f"\n{'='*70}")
    print("✓ BERT Benchmark Comparison Complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
