#!/usr/bin/env python3
"""
BERT TensorFlow vs ONNX Runtime Performance Comparison (Fixed Version)

修复版本 - 使用SavedModel直接加载，避免TF Hub KerasLayer问题
支持ONNX Runtime对比测试
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*70)
print("BERT CPU推理性能对比: TensorFlow vs ONNX Runtime (Fixed)")
print("="*70)
print(f"TensorFlow 版本: {tf.__version__}")
print(f"NumPy 版本: {np.__version__}")

try:
    import onnxruntime as ort
    print(f"ONNX Runtime 版本: {ort.__version__}")
    ONNX_AVAILABLE = True
except ImportError:
    print("ONNX Runtime: Not installed")
    ONNX_AVAILABLE = False

print()


def parse_args():
    parser = argparse.ArgumentParser(description="BERT TensorFlow vs ONNX Benchmark (Fixed)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--num-warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--num-test", type=int, default=50, help="Test iterations")
    parser.add_argument("--output", type=str, default="./results/bert_tf_vs_onnx_fixed", help="Output directory")
    parser.add_argument("--use-saved-model", action="store_true", help="Use SavedModel instead of KerasLayer")
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


def load_bert_with_savedmodel(model_cache_dir, seq_length):
    """
    方案4: 使用SavedModel直接加载BERT
    避免KerasLayer的KerasTensor问题
    """
    print("使用SavedModel方式加载BERT...")

    try:
        import tensorflow_hub as hub

        model_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

        # 下载模型到本地
        print(f"下载模型: {model_url}")
        print("(首次运行需要下载，约440MB，可能需要几分钟...)")

        # 使用hub.resolve()获取本地路径
        model_path = hub.resolve(model_url)
        print(f"✓ 模型已下载到: {model_path}")

        # 直接加载SavedModel
        bert_model = tf.saved_model.load(model_path)
        print("✓ BERT SavedModel 加载成功")

        # 获取签名
        serving_fn = bert_model.signatures['serving_default']
        print(f"✓ 获取serving签名")
        print(f"  输入: {list(serving_fn.structured_input_signature[1].keys())}")
        print(f"  输出: {list(serving_fn.structured_outputs.keys())}")

        return bert_model, serving_fn

    except Exception as e:
        print(f"\n✗ SavedModel加载失败: {e}")
        return None, None


def benchmark_tensorflow_savedmodel(serving_fn, test_data, num_warmup, num_test, batch_size):
    """测试TensorFlow SavedModel推理性能"""
    print(f"\n{'='*70}")
    print("1. TensorFlow SavedModel 推理测试")
    print(f"{'='*70}")

    # Warmup
    print(f"\n热身运行: {num_warmup} iterations...")
    for i in range(num_warmup):
        inputs = {
            'input_word_ids': tf.constant(test_data["input_word_ids"][i:i+batch_size]),
            'input_mask': tf.constant(test_data["input_mask"][i:i+batch_size]),
            'input_type_ids': tf.constant(test_data["input_type_ids"][i:i+batch_size]),
        }
        _ = serving_fn(**inputs)
        if (i + 1) % 5 == 0:
            print(f"  Warmup: {i+1}/{num_warmup}")

    print(f"✓ 热身完成")

    # 性能测试
    print(f"\n性能测试: {num_test} iterations...")
    latencies = []

    for i in range(num_test):
        inputs = {
            'input_word_ids': tf.constant(test_data["input_word_ids"][i:i+batch_size]),
            'input_mask': tf.constant(test_data["input_mask"][i:i+batch_size]),
            'input_type_ids': tf.constant(test_data["input_type_ids"][i:i+batch_size]),
        }

        start = time.perf_counter()
        _ = serving_fn(**inputs)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        if (i + 1) % 10 == 0:
            print(f"  测试: {i+1}/{num_test}")

    # 计算统计信息
    latencies_np = np.array(latencies)

    results = {
        "engine": "TensorFlow SavedModel",
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

    print(f"\n✓ TensorFlow SavedModel 测试完成!")
    print(f"\n结果:")
    print(f"  延迟 (mean):   {results['latency_mean_ms']:.2f} ms")
    print(f"  延迟 (median): {results['latency_median_ms']:.2f} ms")
    print(f"  延迟 (p95):    {results['latency_p95_ms']:.2f} ms")
    print(f"  延迟 (p99):    {results['latency_p99_ms']:.2f} ms")
    print(f"  吞吐量:        {results['throughput_samples_per_sec']:.2f} samples/sec")

    return results


def convert_savedmodel_to_onnx(bert_model_path, output_path, seq_length):
    """将SavedModel转换为ONNX格式"""
    print(f"\n{'='*70}")
    print("转换 SavedModel 到 ONNX")
    print(f"{'='*70}")

    if not ONNX_AVAILABLE:
        print("\n✗ ONNX Runtime未安装，跳过转换")
        return None

    try:
        import tf2onnx

        print(f"SavedModel路径: {bert_model_path}")
        print(f"输出ONNX路径: {output_path}")

        # 使用tf2onnx转换
        print("\n开始转换...")
        print("(可能需要几分钟...)")

        model_proto, _ = tf2onnx.convert.from_saved_model(
            str(bert_model_path),
            input_names=['input_word_ids:0', 'input_mask:0', 'input_type_ids:0'],
            output_names=None,  # 自动检测
            opset=13,
            extra_opset=None,
        )

        # 保存ONNX模型
        with open(output_path, "wb") as f:
            f.write(model_proto.SerializeToString())

        print(f"✓ ONNX模型已保存到: {output_path}")
        print(f"  模型大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

        return output_path

    except Exception as e:
        print(f"\n✗ ONNX转换失败: {e}")
        print("  可能原因:")
        print("  1. tf2onnx未安装 (pip install tf2onnx)")
        print("  2. 模型结构不兼容")
        print("  3. ONNX opset版本问题")
        return None


def benchmark_onnx(onnx_model_path, test_data, num_warmup, num_test, batch_size):
    """测试ONNX Runtime推理性能"""
    print(f"\n{'='*70}")
    print("2. ONNX Runtime 推理测试")
    print(f"{'='*70}")

    if not ONNX_AVAILABLE:
        print("\n✗ ONNX Runtime未安装")
        return None

    if not onnx_model_path or not onnx_model_path.exists():
        print(f"\n✗ ONNX模型文件不存在: {onnx_model_path}")
        return None

    try:
        # 配置ONNX Runtime
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = os.cpu_count()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ['CPUExecutionProvider']

        print(f"\n加载ONNX模型: {onnx_model_path}")
        print(f"  线程数: {session_options.intra_op_num_threads}")
        print(f"  优化级别: ORT_ENABLE_ALL")
        print(f"  执行提供者: {providers}")

        session = ort.InferenceSession(
            str(onnx_model_path),
            sess_options=session_options,
            providers=providers
        )

        # 获取输入/输出信息
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]

        print(f"✓ ONNX Runtime 会话创建成功")
        print(f"  输入: {input_names}")
        print(f"  输出数量: {len(output_names)}")

        # Warmup
        print(f"\n热身运行: {num_warmup} iterations...")
        for i in range(num_warmup):
            inputs = {
                input_names[0]: test_data["input_word_ids"][i:i+batch_size],
                input_names[1]: test_data["input_mask"][i:i+batch_size],
                input_names[2]: test_data["input_type_ids"][i:i+batch_size],
            }
            _ = session.run(None, inputs)
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
            _ = session.run(None, inputs)
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

    except Exception as e:
        print(f"\n✗ ONNX Runtime测试失败: {e}")
        return None


def generate_comparison_report(tf_results, onnx_results, output_dir, config):
    """生成测试报告"""
    print(f"\n{'='*70}")
    print("生成对比报告")
    print(f"{'='*70}")

    report_file = output_dir / "comparison_report.md"

    with open(report_file, "w") as f:
        f.write("# BERT CPU推理性能对比: TensorFlow vs ONNX Runtime\n\n")
        f.write("**测试方法**: 使用SavedModel直接加载，避免KerasLayer问题\n\n")

        f.write("## 测试配置\n\n")
        f.write(f"- **模型**: BERT-base (TensorFlow Hub SavedModel)\n")
        f.write(f"- **Batch Size**: {config['batch_size']}\n")
        f.write(f"- **序列长度**: {config['seq_length']}\n")
        f.write(f"- **热身迭代**: {config['num_warmup']}\n")
        f.write(f"- **测试迭代**: {config['num_test']}\n")
        f.write(f"- **TensorFlow 版本**: {tf.__version__}\n")
        if ONNX_AVAILABLE:
            f.write(f"- **ONNX Runtime 版本**: {ort.__version__}\n")
        f.write("\n")

        f.write("## 性能对比\n\n")

        if onnx_results:
            # 计算加速比
            speedup_mean = tf_results['latency_mean_ms'] / onnx_results['latency_mean_ms']
            speedup_p95 = tf_results['latency_p95_ms'] / onnx_results['latency_p95_ms']
            speedup_throughput = onnx_results['throughput_samples_per_sec'] / tf_results['throughput_samples_per_sec']

            f.write("| 指标 | TensorFlow | ONNX Runtime | 加速比 |\n")
            f.write("|------|------------|--------------|--------|\n")
            f.write(f"| 延迟 (mean) | {tf_results['latency_mean_ms']:.2f} ms | {onnx_results['latency_mean_ms']:.2f} ms | {speedup_mean:.2f}x |\n")
            f.write(f"| 延迟 (median) | {tf_results['latency_median_ms']:.2f} ms | {onnx_results['latency_median_ms']:.2f} ms | {tf_results['latency_median_ms']/onnx_results['latency_median_ms']:.2f}x |\n")
            f.write(f"| 延迟 (std) | {tf_results['latency_std_ms']:.2f} ms | {onnx_results['latency_std_ms']:.2f} ms | - |\n")
            f.write(f"| 延迟 (min) | {tf_results['latency_min_ms']:.2f} ms | {onnx_results['latency_min_ms']:.2f} ms | - |\n")
            f.write(f"| 延迟 (max) | {tf_results['latency_max_ms']:.2f} ms | {onnx_results['latency_max_ms']:.2f} ms | - |\n")
            f.write(f"| 延迟 (p95) | {tf_results['latency_p95_ms']:.2f} ms | {onnx_results['latency_p95_ms']:.2f} ms | {speedup_p95:.2f}x |\n")
            f.write(f"| 延迟 (p99) | {tf_results['latency_p99_ms']:.2f} ms | {onnx_results['latency_p99_ms']:.2f} ms | {tf_results['latency_p99_ms']/onnx_results['latency_p99_ms']:.2f}x |\n")
            f.write(f"| 吞吐量 | {tf_results['throughput_samples_per_sec']:.2f} samples/s | {onnx_results['throughput_samples_per_sec']:.2f} samples/s | {speedup_throughput:.2f}x |\n\n")

            f.write("## 总结\n\n")
            if speedup_mean > 1.0:
                f.write(f"✅ **ONNX Runtime 比 TensorFlow 快 {speedup_mean:.2f}x**\n\n")
            else:
                f.write(f"✅ **TensorFlow 比 ONNX Runtime 快 {1/speedup_mean:.2f}x**\n\n")

            f.write(f"- **平均延迟提升**: {speedup_mean:.2f}x\n")
            f.write(f"- **P95延迟提升**: {speedup_p95:.2f}x\n")
            f.write(f"- **吞吐量提升**: {speedup_throughput:.2f}x\n\n")

            # 性能分析
            f.write("### 性能分析\n\n")
            if speedup_mean >= 1.5:
                f.write(f"🚀 ONNX Runtime显著优于TensorFlow，推荐用于生产环境部署。\n\n")
            elif speedup_mean >= 1.1:
                f.write(f"✅ ONNX Runtime性能优于TensorFlow，适合对延迟敏感的场景。\n\n")
            elif speedup_mean >= 0.9:
                f.write(f"⚖️ 两个引擎性能相当，可根据其他因素选择。\n\n")
            else:
                f.write(f"⚠️ TensorFlow在此配置下性能更好。\n\n")

        else:
            # 仅有TensorFlow结果
            f.write("| 指标 | TensorFlow | ONNX Runtime |\n")
            f.write("|------|------------|-------------|\n")
            f.write(f"| 延迟 (mean) | {tf_results['latency_mean_ms']:.2f} ms | N/A |\n")
            f.write(f"| 延迟 (median) | {tf_results['latency_median_ms']:.2f} ms | N/A |\n")
            f.write(f"| 延迟 (p95) | {tf_results['latency_p95_ms']:.2f} ms | N/A |\n")
            f.write(f"| 延迟 (p99) | {tf_results['latency_p99_ms']:.2f} ms | N/A |\n")
            f.write(f"| 吞吐量 | {tf_results['throughput_samples_per_sec']:.2f} samples/s | N/A |\n\n")

            f.write("## 说明\n\n")
            f.write("⚠️ ONNX模型转换或测试失败，仅显示TensorFlow结果。\n\n")

        f.write("## 技术说明\n\n")
        f.write("### SavedModel加载方式\n\n")
        f.write("本测试使用SavedModel方式直接加载BERT模型，成功避免了TensorFlow Hub ")
        f.write("KerasLayer在TF 2.20中的KerasTensor兼容性问题。\n\n")

        f.write("**对比**:\n")
        f.write("- ❌ **原始方法** (失败): `hub.KerasLayer()` → KerasTensor转换错误\n")
        f.write("- ✅ **新方法** (成功): `tf.saved_model.load()` → 直接使用serving signature\n\n")

        if onnx_results:
            f.write("### ONNX转换\n\n")
            f.write("- 使用 `tf2onnx` 将SavedModel转换为ONNX格式\n")
            f.write("- ONNX opset 版本: 13\n")
            f.write("- 保留完整的BERT模型结构\n\n")

        f.write("## 环境信息\n\n")
        f.write(f"- Python: {'.'.join(map(str, __import__('sys').version_info[:3]))}\n")
        f.write(f"- TensorFlow: {tf.__version__}\n")
        f.write(f"- NumPy: {np.__version__}\n")
        if ONNX_AVAILABLE:
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

    # 加载BERT模型 (SavedModel方式)
    print(f"\n{'='*70}")
    print("加载 BERT 模型 (SavedModel)")
    print(f"{'='*70}")

    bert_model, serving_fn = load_bert_with_savedmodel(models_dir, args.seq_length)

    if serving_fn is None:
        print("\n✗ BERT模型加载失败")
        print("可能的原因:")
        print("  1. 网络连接问题")
        print("  2. TensorFlow Hub下载失败")
        print("  3. SavedModel格式不兼容")
        return

    # 测试 TensorFlow SavedModel
    tf_results = benchmark_tensorflow_savedmodel(
        serving_fn=serving_fn,
        test_data=test_data,
        num_warmup=args.num_warmup,
        num_test=args.num_test,
        batch_size=args.batch_size
    )

    # 保存TensorFlow结果
    tf_result_file = output_dir / "tensorflow_savedmodel_results.json"
    with open(tf_result_file, "w") as f:
        json.dump(tf_results, f, indent=2)
    print(f"\n✓ TensorFlow结果已保存到: {tf_result_file}")

    # 转换并测试 ONNX
    onnx_results = None
    onnx_result_file = None

    if ONNX_AVAILABLE:
        # 获取SavedModel路径
        model_path = Path(hub.resolve(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
        ))

        # 转换为ONNX
        onnx_model_path = output_dir / "bert_model.onnx"
        if not onnx_model_path.exists():
            onnx_model_path = convert_savedmodel_to_onnx(
                model_path,
                onnx_model_path,
                args.seq_length
            )
        else:
            print(f"\n✓ ONNX模型已存在: {onnx_model_path}")

        # 测试ONNX Runtime
        if onnx_model_path and onnx_model_path.exists():
            onnx_results = benchmark_onnx(
                onnx_model_path=onnx_model_path,
                test_data=test_data,
                num_warmup=args.num_warmup,
                num_test=args.num_test,
                batch_size=args.batch_size
            )

            if onnx_results:
                onnx_result_file = output_dir / "onnx_runtime_results.json"
                with open(onnx_result_file, "w") as f:
                    json.dump(onnx_results, f, indent=2)
                print(f"\n✓ ONNX结果已保存到: {onnx_result_file}")
    else:
        print(f"\n⚠️ ONNX Runtime未安装，跳过ONNX测试")
        print(f"   安装方法: pip install onnxruntime tf2onnx")

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
    print("✓ BERT 性能对比测试完成!")
    print(f"{'='*70}")
    print(f"\n结果文件:")
    print(f"  - TensorFlow SavedModel: {tf_result_file}")
    if onnx_result_file:
        print(f"  - ONNX Runtime: {onnx_result_file}")
    print(f"  - 对比报告: {report_file}")
    print(f"\n说明:")
    print(f"  ✅ 成功使用SavedModel方式加载BERT")
    print(f"  ✅ 避免了KerasLayer的KerasTensor问题")
    if onnx_results:
        speedup = tf_results['latency_mean_ms'] / onnx_results['latency_mean_ms']
        if speedup > 1.0:
            print(f"  🚀 ONNX Runtime 比 TensorFlow 快 {speedup:.2f}x")
        else:
            print(f"  ℹ️ TensorFlow 比 ONNX Runtime 快 {1/speedup:.2f}x")
    else:
        print(f"  ⚠️ ONNX测试未运行")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
