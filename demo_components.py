#!/usr/bin/env python3
"""
Demo script to showcase the existing benchmark components.

This demonstrates:
1. Model loading and metadata
2. Dataset preparation
3. Engine initialization
4. Single inference execution
"""

import random
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from benchmark.metrics import MetricsCollector
from benchmark.monitor import ResourceMonitor
from dataset import ImageDatasetLoader, TextDatasetLoader
from engines import TensorFlowEngine
from models import ModelLoader

print("=" * 70)
print("TensorFlow Multi-Engine Benchmark - Component Demo")
print("=" * 70)

# ============================================================================
# 1. Model Loading Demo
# ============================================================================
print("\n" + "=" * 70)
print("1. MODEL LOADING DEMONSTRATION")
print("=" * 70)

print("\n📦 Available Image Models:")
for model_name, model_info in ModelLoader.IMAGE_MODELS.items():
    print(f"  - {model_name}")
    print(f"    Input shape: {model_info['input_shape']}")
    print(f"    Has preprocessing: {model_info.get('preprocessing') is not None}")

print("\n📦 Available Text Models:")
for model_name, model_info in ModelLoader.TEXT_MODELS.items():
    print(f"  - {model_name}")
    print(f"    Max length: {model_info['max_length']}")
    print(f"    Hub URL: {model_info['hub_url']}")

# ============================================================================
# 2. Dataset Preparation Demo
# ============================================================================
print("\n" + "=" * 70)
print("2. DATASET PREPARATION DEMONSTRATION")
print("=" * 70)

print("\n📊 Image Dataset Loader:")
image_loader = ImageDatasetLoader(
    dataset_name="imagenet-1k", split="validation", num_samples=10, target_size=(224, 224)
)
print(f"  {image_loader}")
print(f"  Target size: {image_loader.target_size}")
print(f"  Num samples: {image_loader.num_samples}")

print("\n📊 Text Dataset Loader:")
text_loader = TextDatasetLoader(num_samples=4, max_length=32)
text_loader.load()
print(f"  {text_loader}")
text_tokens = text_loader.tokenize(["Benchmarking is fun"])
print(f"  Tokenized input_ids shape: {text_tokens['input_ids'].shape if text_tokens['input_ids'].ndim > 1 else text_tokens['input_ids'].shape}")

# ============================================================================
# 3. Dummy Input Generation Demo
# ============================================================================
print("\n" + "=" * 70)
print("3. DUMMY INPUT GENERATION DEMONSTRATION")
print("=" * 70)

print("\n🎲 Creating dummy image input:")
dummy_image = ModelLoader.create_dummy_input("mobilenet_v2", "image", batch_size=4)
print(f"  Shape: {dummy_image.shape}")
print(f"  Dtype: {dummy_image.dtype}")
print(f"  Range: [{dummy_image.min():.3f}, {dummy_image.max():.3f}]")

print("\n🎲 Creating dummy text input:")
dummy_text = ModelLoader.create_dummy_input("bert-base-uncased", "text", batch_size=4)
for key, value in dummy_text.items():
    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

# ============================================================================
# 4. Metrics Collection Demo
# ============================================================================
print("\n" + "=" * 70)
print("4. METRICS COLLECTION DEMONSTRATION")
print("=" * 70)

print("\n📈 Simulating inference metrics:")
metrics = MetricsCollector()

# Simulate some latency measurements
for i in range(100):
    latency_ms = random.uniform(10, 30)
    metrics.add_latency(latency_ms)

metrics.set_timing_info(total_time=2.5, num_samples=100, num_batches=100, batch_size=1)

stats = metrics.calculate_statistics()
print(f"  Mean latency: {stats['latency_mean']:.2f} ms")
print(f"  P50 latency: {stats['latency_p50']:.2f} ms")
print(f"  P95 latency: {stats['latency_p95']:.2f} ms")
print(f"  P99 latency: {stats['latency_p99']:.2f} ms")
print(f"  Throughput: {stats['throughput_samples_per_sec']:.2f} samples/sec")

# ============================================================================
# 5. Resource Monitoring Demo
# ============================================================================
print("\n" + "=" * 70)
print("5. RESOURCE MONITORING DEMONSTRATION")
print("=" * 70)

print("\n🔍 Monitoring system resources:")
monitor = ResourceMonitor(sampling_interval=0.1)

with monitor:
    # Simulate some work
    for i in range(10):
        _ = np.random.rand(1000, 1000) @ np.random.rand(1000, 1000)
        time.sleep(0.1)

resource_stats = monitor.get_stats()
print(f"  Average CPU usage: {resource_stats.get('cpu_percent_avg', 0):.1f}%")
print(f"  Peak CPU usage: {resource_stats.get('cpu_percent_max', 0):.1f}%")
print(f"  Average memory: {resource_stats.get('memory_mb_avg', 0):.1f} MB")
print(f"  Peak memory: {resource_stats.get('memory_mb_max', 0):.1f} MB")

# ============================================================================
# 6. Engine Features Demo
# ============================================================================
print("\n" + "=" * 70)
print("6. ENGINE FEATURES DEMONSTRATION")
print("=" * 70)

print("\n🔧 TensorFlow Engine:")
print("  Supports: XLA compilation, mixed precision, thread optimization")
tf_engine = TensorFlowEngine(config={"xla": False, "mixed_precision": False})
print(f"  Engine name: {tf_engine.engine_name}")
print(f"  Initialized: {tf_engine.model is None}")

print("\n🔧 ONNX Engine:")
print("  Supports: Multiple execution providers, graph optimization")

print("\n🔧 Available engine types:")
print("  - TensorFlow (baseline, XLA, mixed precision)")
print("  - TFLite (dynamic, int8, float16, full_int8)")
print("  - ONNX Runtime (baseline, O1, O2)")
print("  - OpenVINO (FP32, FP16, INT8, AUTO - x86_64 only)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("DEMO COMPLETE")
print("=" * 70)
print("\n✅ Successfully demonstrated:")
print("  ✓ Model metadata and registry")
print("  ✓ Dataset loader initialization")
print("  ✓ Dummy input generation")
print("  ✓ Metrics collection and statistics")
print("  ✓ Resource monitoring")
print("  ✓ Engine configurations")
print("\n💡 Note: Full benchmark execution requires model loading,")
print("   conversion, and end-to-end integration (Phase 6)")
print()
