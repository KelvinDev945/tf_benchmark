#!/usr/bin/env python3
"""
BERT TensorFlow vs ONNX Runtime Performance Comparison

比较TensorFlow原生BERT和ONNX Runtime BERT的CPU推理性能
不需要HuggingFace Transformers库，使用TensorFlow Hub

Usage:
    python3 scripts/bert_tf_vs_onnx.py
    python3 scripts/bert_tf_vs_onnx.py --num-test 50
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# 设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*70)
print("BERT CPU推理性能对比: TensorFlow vs ONNX Runtime")
print("="*70)
print(f"TensorFlow 版本: {tf.__version__}")
print(f"NumPy 版本: {np.__version__}")
print()


def parse_args():
    parser = argparse.ArgumentParser(description="BERT TensorFlow vs ONNX Benchmark")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--num-warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--num-test", type=int, default=50, help="Test iterations")
    parser.add_argument("--output", type=str, default="./results/bert_tf_vs_onnx", help="Output directory")
    return parser.parse_args()


def create_test_data(num_samples, seq_length, batch_size=1):
    """创建模拟的测试数据"""
    print(f"\n创建测试数据...")
    print(f"  样本数: {num_samples}")
    print(f"  序列长度: {seq_length}")
    print(f"  Batch size: {batch_size}")

    # 生成随机token IDs (BERT vocab size约为30000)
    input_word_ids = np.random.randint(0, 30000, size=(num_samples, seq_length), dtype=np.int32)
    input_mask = np.ones((num_samples, seq_length), dtype=np.int32)
    input_type_ids = np.zeros((num_samples, seq_length), dtype=np.int32)

    print(f"✓ 测试数据准备完成")

    return {
        "input_word_ids": input_word_ids,
        "input_mask": input_mask,
        "input_type_ids": input_type_ids,
    }


def benchmark_tensorflow(model, test_data, num_warmup, num_test, batch_size):
    """测试TensorFlow推理性能"""
    print(f"\n{'='*70}")
    print("1. TensorFlow 原生推理测试")
    print(f"{'='*70}")

    # Warmup
    print(f"\n热身运行: {num_warmup} iterations...")
    for i in range(num_warmup):
        batch_data = [
            test_data["input_word_ids"][i:i+batch_size],
            test_data["input_mask"][i:i+batch_size],
            test_data["input_type_ids"][i:i+batch_size],
        ]
        _ = model(batch_data, training=False)
        if (i + 1) % 5 == 0:
            print(f"  Warmup: {i+1}/{num_warmup}")

    print(f"✓ 热身完成")

    # 性能测试
    print(f"\n性能测试: {num_test} iterations...")
    latencies = []

    for i in range(num_test):
        batch_data = [
            test_data["input_word_ids"][i:i+batch_size],
            test_data["input_mask"][i:i+batch_size],
            test_data["input_type_ids"][i:i+batch_size],
        ]

        start = time.perf_counter()
        _ = model(batch_data, training=False)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        if (i + 1) % 10 == 0:
            print(f"  测试: {i+1}/{num_test}")

    # 计算统计信息
    latencies_np = np.array(latencies)

    results = {
        "engine": "TensorFlow",
        "latency_mean_ms": float(np.mean(latencies_np)),
        "latency_median_ms": float(np.median(latencies_np)),
        "latency_std_ms": float(np.std(latencies_np)),
        "latency_min_ms": float(np.min(latencies_np)),
        "latency_max_ms": float(np.max(latencies_np)),
        "latency_p50_ms": float(np.percentile(latencies_np, 50)),
        "latency_p95_ms": float(np.percentile(latencies_np, 95)),
        "latency_p99_ms": float(np.percentile(latencies_np, 99)),
        "throughput_samples_per_sec": batch_size * num_test / (np.sum(latencies_np) / 1000),
    }

    print(f"\n✓ TensorFlow 测试完成!")
    print(f"\n结果:")
    print(f"  延迟 (mean):   {results['latency_mean_ms']:.2f} ms")
    print(f"  延迟 (median): {results['latency_median_ms']:.2f} ms")
    print(f"  延迟 (p95):    {results['latency_p95_ms']:.2f} ms")
    print(f"  延迟 (p99):    {results['latency_p99_ms']:.2f} ms")
    print(f"  吞吐量:        {results['throughput_samples_per_sec']:.2f} samples/sec")

    return results


def benchmark_onnx(onnx_model_path, test_data, num_warmup, num_test, batch_size):
    """测试ONNX Runtime推理性能"""
    print(f"\n{'='*70}")
    print("2. ONNX Runtime 推理测试")
    print(f"{'='*70}")

    try:
        import onnxruntime as ort
        print(f"ONNX Runtime 版本: {ort.__version__}")
    except ImportError:
        print("✗ ONNX Runtime 未安装")
        return None

    # 检查ONNX模型是否存在
    if not onnx_model_path.exists():
        print(f"✗ ONNX 模型文件不存在: {onnx_model_path}")
        print(f"  请先将TensorFlow模型转换为ONNX格式")
        return None

    # 加载ONNX模型
    print(f"\n加载ONNX模型: {onnx_model_path}")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(str(onnx_model_path), sess_options=session_options, providers=providers)

    print(f"✓ ONNX 模型加载成功")

    # 获取输入输出名称
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    print(f"  输入: {input_names}")
    print(f"  输出: {output_names}")

    # Warmup
    print(f"\n热身运行: {num_warmup} iterations...")
    for i in range(num_warmup):
        inputs = {
            input_names[0]: test_data["input_word_ids"][i:i+batch_size],
            input_names[1]: test_data["input_mask"][i:i+batch_size],
            input_names[2]: test_data["input_type_ids"][i:i+batch_size],
        }
        _ = session.run(output_names, inputs)
        if (i + 1) % 5 == 0:
            print(f"  Warmup: {i+1}/{num_warmup}")

    print(f"✓ 热身完成")

    # 性能测试
    print(f"\n性能测试: {num_test} iterations...")
    latencies = []

    for i in range(num_test):
        inputs = {
            input_names[0]: test_data["input_word_ids"][i:i+batch_size],
            input_names[1]: test_data["input_mask"][i:i+batch_size],
            input_names[2]: test_data["input_type_ids"][i:i+batch_size],
        }

        start = time.perf_counter()
        _ = session.run(output_names, inputs)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        if (i + 1) % 10 == 0:
            print(f"  测试: {i+1}/{num_test}")

    # 计算统计信息
    latencies_np = np.array(latencies)

    results = {
        "engine": "ONNX Runtime",
        "latency_mean_ms": float(np.mean(latencies_np)),
        "latency_median_ms": float(np.median(latencies_np)),
        "latency_std_ms": float(np.std(latencies_np)),
        "latency_min_ms": float(np.min(latencies_np)),
        "latency_max_ms": float(np.max(latencies_np)),
        "latency_p50_ms": float(np.percentile(latencies_np, 50)),
        "latency_p95_ms": float(np.percentile(latencies_np, 95)),
        "latency_p99_ms": float(np.percentile(latencies_np, 99)),
        "throughput_samples_per_sec": batch_size * num_test / (np.sum(latencies_np) / 1000),
    }

    print(f"\n✓ ONNX Runtime 测试完成!")
    print(f"\n结果:")
    print(f"  延迟 (mean):   {results['latency_mean_ms']:.2f} ms")
    print(f"  延迟 (median): {results['latency_median_ms']:.2f} ms")
    print(f"  延迟 (p95):    {results['latency_p95_ms']:.2f} ms")
    print(f"  延迟 (p99):    {results['latency_p99_ms']:.2f} ms")
    print(f"  吞吐量:        {results['throughput_samples_per_sec']:.2f} samples/sec")

    return results


def generate_comparison_report(tf_results, onnx_results, output_dir, config):
    """生成对比报告"""
    print(f"\n{'='*70}")
    print("生成对比报告")
    print(f"{'='*70}")

    report_file = output_dir / "comparison_report.md"

    with open(report_file, "w") as f:
        f.write("# BERT CPU推理性能对比: TensorFlow vs ONNX Runtime\n\n")

        f.write("## 测试配置\n\n")
        f.write(f"- **模型**: BERT-base (TensorFlow Hub)\n")
        f.write(f"- **Batch Size**: {config['batch_size']}\n")
        f.write(f"- **序列长度**: {config['seq_length']}\n")
        f.write(f"- **热身迭代**: {config['num_warmup']}\n")
        f.write(f"- **测试迭代**: {config['num_test']}\n")
        f.write(f"- **TensorFlow 版本**: {tf.__version__}\n\n")

        f.write("## 性能对比\n\n")
        f.write("| 指标 | TensorFlow | ONNX Runtime | 提升 |\n")
        f.write("|------|------------|--------------|------|\n")

        if onnx_results:
            speedup_mean = tf_results['latency_mean_ms'] / onnx_results['latency_mean_ms']
            speedup_p95 = tf_results['latency_p95_ms'] / onnx_results['latency_p95_ms']
            speedup_throughput = onnx_results['throughput_samples_per_sec'] / tf_results['throughput_samples_per_sec']

            f.write(f"| 延迟 (mean) | {tf_results['latency_mean_ms']:.2f} ms | {onnx_results['latency_mean_ms']:.2f} ms | {speedup_mean:.2f}x |\n")
            f.write(f"| 延迟 (median) | {tf_results['latency_median_ms']:.2f} ms | {onnx_results['latency_median_ms']:.2f} ms | {tf_results['latency_median_ms']/onnx_results['latency_median_ms']:.2f}x |\n")
            f.write(f"| 延迟 (p95) | {tf_results['latency_p95_ms']:.2f} ms | {onnx_results['latency_p95_ms']:.2f} ms | {speedup_p95:.2f}x |\n")
            f.write(f"| 延迟 (p99) | {tf_results['latency_p99_ms']:.2f} ms | {onnx_results['latency_p99_ms']:.2f} ms | {tf_results['latency_p99_ms']/onnx_results['latency_p99_ms']:.2f}x |\n")
            f.write(f"| 吞吐量 | {tf_results['throughput_samples_per_sec']:.2f} samples/s | {onnx_results['throughput_samples_per_sec']:.2f} samples/s | {speedup_throughput:.2f}x |\n\n")

            f.write("## 总结\n\n")
            if speedup_mean > 1.0:
                f.write(f"✅ **ONNX Runtime 比 TensorFlow 快 {speedup_mean:.2f}x**\n\n")
            else:
                f.write(f"✅ **TensorFlow 比 ONNX Runtime 快 {1/speedup_mean:.2f}x**\n\n")

            f.write(f"- 平均延迟提升: {speedup_mean:.2f}x\n")
            f.write(f"- P95延迟提升: {speedup_p95:.2f}x\n")
            f.write(f"- 吞吐量提升: {speedup_throughput:.2f}x\n\n")
        else:
            f.write(f"| 延迟 (mean) | {tf_results['latency_mean_ms']:.2f} ms | N/A | N/A |\n")
            f.write(f"| 延迟 (median) | {tf_results['latency_median_ms']:.2f} ms | N/A | N/A |\n")
            f.write(f"| 延迟 (p95) | {tf_results['latency_p95_ms']:.2f} ms | N/A | N/A |\n")
            f.write(f"| 延迟 (p99) | {tf_results['latency_p99_ms']:.2f} ms | N/A | N/A |\n")
            f.write(f"| 吞吐量 | {tf_results['throughput_samples_per_sec']:.2f} samples/s | N/A | N/A |\n\n")

            f.write("## 说明\n\n")
            f.write("ONNX模型未找到或测试失败。请先将TensorFlow模型转换为ONNX格式。\n\n")

        f.write("## 测试方法\n\n")
        f.write("1. 使用TensorFlow Hub加载BERT-base模型\n")
        f.write("2. 生成随机测试数据 (模拟tokenized输入)\n")
        f.write(f"3. 热身{config['num_warmup']}次迭代\n")
        f.write(f"4. 测试{config['num_test']}次迭代并记录延迟\n")
        f.write("5. 计算统计指标 (mean, median, p95, p99)\n\n")

        f.write("## 环境信息\n\n")
        f.write(f"- Python: {'.'.join(map(str, __import__('sys').version_info[:3]))}\n")
        f.write(f"- TensorFlow: {tf.__version__}\n")
        f.write(f"- NumPy: {np.__version__}\n")

        if onnx_results:
            import onnxruntime as ort
            f.write(f"- ONNX Runtime: {ort.__version__}\n")

    print(f"✓ 报告已保存到: {report_file}")

    return report_file


def main():
    args = parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"输出目录: {output_dir}")

    # 创建测试数据
    test_data = create_test_data(
        num_samples=args.num_test,
        seq_length=args.seq_length,
        batch_size=args.batch_size
    )

    # 加载或创建TensorFlow BERT模型
    print(f"\n{'='*70}")
    print("加载 TensorFlow BERT 模型")
    print(f"{'='*70}")

    tf_model_path = models_dir / "bert_tf_model"

    if tf_model_path.exists():
        print(f"从缓存加载模型: {tf_model_path}")
        try:
            model = tf.keras.models.load_model(tf_model_path)
            print("✓ 模型加载成功")
        except Exception as e:
            print(f"✗ 加载失败: {e}")
            print("将尝试重新下载...")
            tf_model_path = None

    if not tf_model_path or not tf_model_path.exists():
        print("从 TensorFlow Hub 下载 BERT 模型...")
        print("(首次运行需要下载，约440MB，可能需要几分钟...)")

        try:
            import tensorflow_hub as hub

            bert_model_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
            bert_layer = hub.KerasLayer(bert_model_url, trainable=False)

            # 构建模型
            input_word_ids = tf.keras.layers.Input(shape=(args.seq_length,), dtype=tf.int32, name="input_word_ids")
            input_mask = tf.keras.layers.Input(shape=(args.seq_length,), dtype=tf.int32, name="input_mask")
            input_type_ids = tf.keras.layers.Input(shape=(args.seq_length,), dtype=tf.int32, name="input_type_ids")

            bert_inputs = {
                "input_word_ids": input_word_ids,
                "input_mask": input_mask,
                "input_type_ids": input_type_ids
            }

            bert_outputs = bert_layer(bert_inputs)
            pooled_output = bert_outputs["pooled_output"]
            output = tf.keras.layers.Dense(2, activation='softmax')(pooled_output)

            model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)

            print("✓ BERT 模型构建完成")
            print(f"  参数总数: {model.count_params():,}")

            # 保存模型以便下次使用
            print(f"\n保存模型到: {tf_model_path}")
            model.save(tf_model_path)
            print("✓ 模型已保存")

        except Exception as e:
            print(f"\n✗ 错误: {e}")
            print("\n可能的原因:")
            print("  1. 网络连接问题 - 无法从 TensorFlow Hub 下载模型")
            print("  2. tensorflow-hub 库未安装 - 运行: pip install tensorflow-hub")
            print("\n解决方案:")
            print("  - 检查网络连接")
            print("  - 使用代理: export HTTPS_PROXY=http://proxy:port")
            print("  - 或手动下载模型")
            return

    # 测试 TensorFlow
    tf_results = benchmark_tensorflow(
        model=model,
        test_data=test_data,
        num_warmup=args.num_warmup,
        num_test=args.num_test,
        batch_size=args.batch_size
    )

    # 保存TensorFlow结果
    tf_result_file = output_dir / "tensorflow_results.json"
    with open(tf_result_file, "w") as f:
        json.dump(tf_results, f, indent=2)
    print(f"\n✓ TensorFlow 结果已保存到: {tf_result_file}")

    # 测试 ONNX Runtime (如果模型存在)
    onnx_model_path = models_dir / "bert_model.onnx"
    onnx_results = benchmark_onnx(
        onnx_model_path=onnx_model_path,
        test_data=test_data,
        num_warmup=args.num_warmup,
        num_test=args.num_test,
        batch_size=args.batch_size
    )

    if onnx_results:
        # 保存ONNX结果
        onnx_result_file = output_dir / "onnx_results.json"
        with open(onnx_result_file, "w") as f:
            json.dump(onnx_results, f, indent=2)
        print(f"\n✓ ONNX Runtime 结果已保存到: {onnx_result_file}")

    # 生成对比报告
    config = {
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "num_warmup": args.num_warmup,
        "num_test": args.num_test,
    }

    report_file = generate_comparison_report(tf_results, onnx_results, output_dir, config)

    # 最终总结
    print(f"\n{'='*70}")
    print("✓ 性能对比测试完成!")
    print(f"{'='*70}")
    print(f"\n结果文件:")
    print(f"  - TensorFlow: {output_dir}/tensorflow_results.json")
    if onnx_results:
        print(f"  - ONNX Runtime: {output_dir}/onnx_results.json")
    print(f"  - 对比报告: {report_file}")

    if not onnx_results:
        print(f"\n说明: ONNX 模型未找到")
        print(f"如需对比ONNX Runtime性能，请先将TensorFlow模型转换为ONNX格式:")
        print(f"  python -m tf2onnx.convert --saved-model {tf_model_path} --output {onnx_model_path}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
