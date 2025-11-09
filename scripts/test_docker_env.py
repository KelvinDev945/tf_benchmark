#!/usr/bin/env python3
"""
Test Docker Environment with uv package manager
验证使用uv优化后的Docker环境
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

print("=" * 70)
print("Docker Environment Test - uv Package Manager")
print("=" * 70)
print()

# System info
print("📊 System Information:")
print(f"  Platform: {platform.platform()}")
print(f"  Architecture: {platform.machine()}")
print(f"  Python: {sys.version.split()[0]}")
print()

# Package versions
print("📦 Package Versions:")
print(f"  TensorFlow: {tf.__version__}")
print(f"  NumPy: {np.__version__}")

try:
    import onnxruntime as ort

    print(f"  ONNX Runtime: {ort.__version__}")
except ImportError:
    print("  ONNX Runtime: Not installed")

try:
    import tensorflow_hub as hub

    print(f"  TensorFlow Hub: {hub.__version__}")
except ImportError:
    print("  TensorFlow Hub: Not installed")

try:
    from openvino.runtime import Core

    _ = Core
    print("  OpenVINO: Available")
except ImportError:
    print("  OpenVINO: Not available")

print()

# Quick MobileNetV2 inference test
print("=" * 70)
print("Quick MobileNetV2 Inference Test")
print("=" * 70)
print()

print("Loading MobileNetV2 model...")
model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=True, weights="imagenet"
)
print("✓ Model loaded successfully")
print(f"  Parameters: {model.count_params():,}")
print()

# Create test data
batch_size = 1
test_samples = 20
test_data = np.random.rand(test_samples, 224, 224, 3).astype(np.float32)

print("Test Configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Test samples: {test_samples}")
print(f"  Input shape: {test_data.shape}")
print()

# Warmup
print("Warmup (5 iterations)...")
for i in range(5):
    _ = model.predict(test_data[i : i + 1], verbose=0)
print("✓ Warmup complete")
print()

# Benchmark
print(f"Running benchmark ({test_samples} iterations)...")
latencies = []

for i in range(test_samples):
    start = time.perf_counter()
    _ = model.predict(test_data[i : i + 1], verbose=0)
    end = time.perf_counter()
    latency_ms = (end - start) * 1000
    latencies.append(latency_ms)
    if (i + 1) % 5 == 0:
        print(f"  Progress: {i+1}/{test_samples}")

print("✓ Benchmark complete")
print()

# Calculate statistics
latencies_np = np.array(latencies)
results = {
    "model": "MobileNetV2",
    "engine": "TensorFlow (uv Docker)",
    "batch_size": batch_size,
    "test_samples": test_samples,
    "tensorflow_version": tf.__version__,
    "numpy_version": np.__version__,
    "python_version": sys.version.split()[0],
    "statistics": {
        "latency_mean_ms": float(np.mean(latencies_np)),
        "latency_median_ms": float(np.median(latencies_np)),
        "latency_std_ms": float(np.std(latencies_np)),
        "latency_min_ms": float(np.min(latencies_np)),
        "latency_max_ms": float(np.max(latencies_np)),
        "latency_p50_ms": float(np.percentile(latencies_np, 50)),
        "latency_p95_ms": float(np.percentile(latencies_np, 95)),
        "latency_p99_ms": float(np.percentile(latencies_np, 99)),
        "throughput_samples_per_sec": test_samples / (np.sum(latencies_np) / 1000),
    },
}

# Print results
print("=" * 70)
print("Benchmark Results")
print("=" * 70)
print()
print("Latency Statistics:")
print(f"  Mean:   {results['statistics']['latency_mean_ms']:.2f} ms")
print(f"  Median: {results['statistics']['latency_median_ms']:.2f} ms")
print(f"  Std:    {results['statistics']['latency_std_ms']:.2f} ms")
print(f"  Min:    {results['statistics']['latency_min_ms']:.2f} ms")
print(f"  Max:    {results['statistics']['latency_max_ms']:.2f} ms")
print(f"  P50:    {results['statistics']['latency_p50_ms']:.2f} ms")
print(f"  P95:    {results['statistics']['latency_p95_ms']:.2f} ms")
print(f"  P99:    {results['statistics']['latency_p99_ms']:.2f} ms")
print()
print(f"Throughput: {results['statistics']['throughput_samples_per_sec']:.2f} samples/sec")
print()

# Save results
output_dir = Path("/app/results/docker_uv_test")
output_dir.mkdir(parents=True, exist_ok=True)

results_file = output_dir / "mobilenet_v2_results.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to: {results_file}")
print()

print("=" * 70)
print("✓ Docker Environment Test Complete!")
print("=" * 70)
print()
print("Summary:")
print("  - uv package manager: Working ✓")
print("  - TensorFlow: Working ✓")
print("  - Model loading: Working ✓")
print("  - Inference: Working ✓")
print("  - Performance measurement: Working ✓")
