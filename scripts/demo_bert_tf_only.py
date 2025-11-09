#!/usr/bin/env python3
"""
BERT TensorFlow-Only Demo Script

演示如何使用纯TensorFlow BERT模型进行benchmark测试
不需要PyTorch，使用TensorFlow Hub的预训练模型

Usage:
    python3 scripts/demo_bert_tf_only.py
"""

import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

print("=" * 70)
print("BERT TensorFlow-Only Benchmark Demo")
print("=" * 70)
print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Hub version: {hub.__version__}")
print()

# 配置
BATCH_SIZE = 1
SEQ_LENGTH = 128
NUM_WARMUP = 5
NUM_TEST = 20

# 创建结果目录
output_dir = Path("./results/bert_tf_demo")
output_dir.mkdir(parents=True, exist_ok=True)

# 配置 TF Hub 缓存目录
tfhub_cache_dir = Path.home() / ".cache" / "tfhub"
os.environ.setdefault("TFHUB_CACHE_DIR", str(tfhub_cache_dir))
tfhub_cache_dir.mkdir(parents=True, exist_ok=True)

print("加载TensorFlow BERT模型（从TensorFlow Hub）...")
print("模型: bert_en_uncased_L-12_H-768_A-12")

# 使用TensorFlow Hub的BERT模型
# 这是纯TensorFlow版本，不需要PyTorch
bert_model_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

try:
    # 加载预训练的BERT模型
    bert_module = hub.load(bert_model_url)
    signature = bert_module.signatures["serving_default"]
    print("✓ BERT模型加载成功")

    # 创建一个简单的分类模型
    input_word_ids = tf.keras.layers.Input(
        shape=(SEQ_LENGTH,), dtype=tf.int32, name="input_word_ids"
    )
    input_mask = tf.keras.layers.Input(shape=(SEQ_LENGTH,), dtype=tf.int32, name="input_mask")
    input_type_ids = tf.keras.layers.Input(
        shape=(SEQ_LENGTH,), dtype=tf.int32, name="input_type_ids"
    )

    # BERT编码
    def bert_forward(inputs):
        word_ids, mask, type_ids = inputs
        outputs = signature(
            input_word_ids=tf.cast(word_ids, tf.int32),
            input_mask=tf.cast(mask, tf.int32),
            input_type_ids=tf.cast(type_ids, tf.int32),
        )
        return outputs["bert_encoder"]

    pooled_output = tf.keras.layers.Lambda(bert_forward)([input_word_ids, input_mask, input_type_ids])

    # 添加分类层
    output = tf.keras.layers.Dense(2, activation="softmax")(pooled_output)

    model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)

    print("✓ 模型构建完成")
    print(f"  参数总数: {model.count_params():,}")

    # 创建模拟测试数据
    print(f"\n创建测试数据 (batch_size={BATCH_SIZE}, seq_length={SEQ_LENGTH})...")

    test_data = {
        "input_word_ids": np.random.randint(0, 30000, size=(NUM_TEST, SEQ_LENGTH), dtype=np.int32),
        "input_mask": np.ones((NUM_TEST, SEQ_LENGTH), dtype=np.int32),
        "input_type_ids": np.zeros((NUM_TEST, SEQ_LENGTH), dtype=np.int32),
    }

    print(f"✓ 测试数据准备完成 ({NUM_TEST} 样本)")

    # Benchmark: TensorFlow Baseline
    print(f"\n{'='*70}")
    print("1. TensorFlow Baseline测试")
    print(f"{'='*70}")

    print(f"\nWarmup: {NUM_WARMUP} iterations...")
    for i in range(NUM_WARMUP):
        batch_data = [
            test_data["input_word_ids"][i : i + 1],
            test_data["input_mask"][i : i + 1],
            test_data["input_type_ids"][i : i + 1],
        ]
        _ = model(batch_data, training=False)

    print(f"测试: {NUM_TEST} iterations...")
    latencies = []

    for i in range(NUM_TEST):
        batch_data = [
            test_data["input_word_ids"][i : i + 1],
            test_data["input_mask"][i : i + 1],
            test_data["input_type_ids"][i : i + 1],
        ]

        start = time.time()
        _ = model(batch_data, training=False)
        end = time.time()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

    # 计算统计信息
    latencies_np = np.array(latencies)

    results = {
        "model": "bert_en_uncased_L-12_H-768_A-12",
        "engine": "tensorflow_baseline",
        "batch_size": BATCH_SIZE,
        "seq_length": SEQ_LENGTH,
        "latency_mean": float(np.mean(latencies_np)),
        "latency_median": float(np.median(latencies_np)),
        "latency_std": float(np.std(latencies_np)),
        "latency_min": float(np.min(latencies_np)),
        "latency_max": float(np.max(latencies_np)),
        "latency_p50": float(np.percentile(latencies_np, 50)),
        "latency_p95": float(np.percentile(latencies_np, 95)),
        "latency_p99": float(np.percentile(latencies_np, 99)),
        "throughput_samples_per_sec": NUM_TEST / np.sum(latencies_np) * 1000,
    }

    print("\n✓ 测试完成!")
    print("\n结果:")
    print(f"  延迟 (mean):   {results['latency_mean']:.2f} ms")
    print(f"  延迟 (median): {results['latency_median']:.2f} ms")
    print(f"  延迟 (p95):    {results['latency_p95']:.2f} ms")
    print(f"  延迟 (p99):    {results['latency_p99']:.2f} ms")
    print(f"  吞吐量:        {results['throughput_samples_per_sec']:.2f} samples/sec")

    # 保存结果
    import json

    result_file = output_dir / "tf_bert_demo_results.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ 结果已保存到: {result_file}")

    # 生成报告
    report_file = output_dir / "tf_bert_demo_report.md"
    with open(report_file, "w") as f:
        f.write("# BERT TensorFlow Benchmark Demo\n\n")
        f.write("## 配置\n\n")
        f.write(f"- **模型**: {results['model']}\n")
        f.write("- **引擎**: TensorFlow (纯TF，无PyTorch)\n")
        f.write(f"- **Batch Size**: {BATCH_SIZE}\n")
        f.write(f"- **序列长度**: {SEQ_LENGTH}\n")
        f.write(f"- **测试迭代**: {NUM_TEST}\n\n")

        f.write("## 性能结果\n\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|----|\n")
        f.write(f"| 延迟 (mean) | {results['latency_mean']:.2f} ms |\n")
        f.write(f"| 延迟 (median) | {results['latency_median']:.2f} ms |\n")
        f.write(f"| 延迟 (p95) | {results['latency_p95']:.2f} ms |\n")
        f.write(f"| 延迟 (p99) | {results['latency_p99']:.2f} ms |\n")
        f.write(f"| 吞吐量 | {results['throughput_samples_per_sec']:.2f} samples/sec |\n\n")

        f.write("## 说明\n\n")
        f.write("本演示使用纯TensorFlow BERT模型（从TensorFlow Hub加载），\n")
        f.write("不需要PyTorch依赖。这证明了benchmark框架可以完全在TensorFlow环境下工作。\n\n")
        f.write("**下一步**: 使用TFLite和ONNX Runtime进行量化模型对比测试。\n")

    print(f"✓ 报告已保存到: {report_file}")

    print(f"\n{'='*70}")
    print("✓ BERT TensorFlow演示完成!")
    print(f"{'='*70}")

except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback

    traceback.print_exc()

print("\n说明: TensorFlow Hub模型需要网络下载。")
print("如果下载失败，请检查网络连接或使用代理。")
