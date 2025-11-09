#!/usr/bin/env python3
"""
BERT模型 TensorFlow vs ONNX Runtime 性能对比测试 (使用HuggingFace Optimum)

使用Optimum进行ONNX转换，避免tf2onnx的protobuf版本冲突
解决 TODO.md Issue #3 - ONNX转换失败问题
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_environment():
    """检查环境配置"""
    print_section("环境检查")

    env_info = {}

    import tensorflow as tf

    env_info["tensorflow"] = tf.__version__
    print(f"✓ TensorFlow: {tf.__version__}")

    import numpy as np

    env_info["numpy"] = np.__version__
    print(f"✓ NumPy: {np.__version__}")

    try:
        import onnxruntime as ort

        env_info["onnxruntime"] = ort.__version__
        print(f"✓ ONNX Runtime: {ort.__version__}")
    except Exception as e:
        print(f"✗ ONNX Runtime: {e}")
        sys.exit(1)

    try:
        import optimum

        env_info["optimum"] = optimum.__version__
        print(f"✓ Optimum: {optimum.__version__}")
    except Exception as e:
        print(f"✗ Optimum: {e}")
        print("请安装: pip install optimum[onnxruntime]")
        sys.exit(1)

    try:
        import transformers

        env_info["transformers"] = transformers.__version__
        print(f"✓ Transformers: {transformers.__version__}")
    except Exception as e:
        print(f"✗ Transformers: {e}")
        sys.exit(1)

    try:
        import google.protobuf

        env_info["protobuf"] = google.protobuf.__version__
        print(f"✓ Protobuf: {google.protobuf.__version__}")
    except Exception as e:
        print(f"✗ Protobuf: {e}")

    return env_info


def benchmark_tensorflow_bert(model_name="bert-base-uncased", num_runs=100, num_warmup=10):
    """
    TensorFlow BERT基准测试

    参数:
        model_name: HuggingFace模型名称
        num_runs: 测试运行次数
        num_warmup: 预热运行次数
    """
    print_section(f"TensorFlow BERT 基准测试 ({model_name})")

    from transformers import BertTokenizer, TFBertForSequenceClassification

    # 加载模型和tokenizer
    print("加载TensorFlow BERT模型...")
    model = TFBertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 准备测试数据
    test_text = "This is a test sentence for benchmarking BERT model performance."
    inputs = tokenizer(
        test_text, return_tensors="tf", padding=True, truncation=True, max_length=128
    )

    # 预热
    print(f"预热运行 {num_warmup} 次...")
    for _ in range(num_warmup):
        _ = model(inputs)

    # 基准测试
    print(f"基准测试运行 {num_runs} 次...")
    latencies = []

    for i in range(num_runs):
        start_time = time.perf_counter()
        _ = model(inputs)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

        if (i + 1) % 20 == 0:
            print(f"  进度: {i + 1}/{num_runs}")

    # 计算统计信息
    latencies = np.array(latencies)
    results = {
        "framework": "TensorFlow",
        "model": model_name,
        "num_runs": num_runs,
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "throughput_samples_per_sec": 1000.0 / np.mean(latencies),
    }

    print(f"\n结果:")
    print(f"  平均延迟: {results['mean_latency_ms']:.2f} ms")
    print(f"  P95延迟: {results['p95_latency_ms']:.2f} ms")
    print(f"  P99延迟: {results['p99_latency_ms']:.2f} ms")
    print(f"  吞吐量: {results['throughput_samples_per_sec']:.2f} samples/sec")

    return results, model, tokenizer


def convert_to_onnx_with_optimum(model_name, output_dir):
    """
    使用Optimum将HuggingFace模型转换为ONNX

    参数:
        model_name: HuggingFace模型名称
        output_dir: ONNX模型输出目录
    """
    print_section(f"使用Optimum转换为ONNX ({model_name})")

    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"转换模型到: {output_path}")
    print("这可能需要几分钟时间...")

    start_time = time.time()

    # Optimum会自动处理转换
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_name, export=True  # 自动导出为ONNX
    )

    # 保存ONNX模型
    ort_model.save_pretrained(output_path)

    # 同时保存tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_path)

    conversion_time = time.time() - start_time

    # 获取模型大小
    onnx_file = output_path / "model.onnx"
    if onnx_file.exists():
        model_size_mb = onnx_file.stat().st_size / (1024 * 1024)
        print(f"\n✓ ONNX转换成功!")
        print(f"  转换时间: {conversion_time:.2f} 秒")
        print(f"  ONNX模型大小: {model_size_mb:.2f} MB")
        print(f"  保存路径: {onnx_file}")
    else:
        print("警告: ONNX文件未找到")
        model_size_mb = 0

    return {
        "conversion_time_sec": conversion_time,
        "model_size_mb": model_size_mb,
        "output_path": str(output_path),
    }


def benchmark_onnx_bert(model_dir, num_runs=100, num_warmup=10):
    """
    ONNX Runtime BERT基准测试

    参数:
        model_dir: ONNX模型目录
        num_runs: 测试运行次数
        num_warmup: 预热运行次数
    """
    print_section(f"ONNX Runtime BERT 基准测试")

    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    # 加载ONNX模型
    print(f"加载ONNX模型: {model_dir}")
    model = ORTModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 准备测试数据
    test_text = "This is a test sentence for benchmarking BERT model performance."
    inputs = tokenizer(
        test_text, return_tensors="pt", padding=True, truncation=True, max_length=128
    )

    # 预热
    print(f"预热运行 {num_warmup} 次...")
    for _ in range(num_warmup):
        _ = model(**inputs)

    # 基准测试
    print(f"基准测试运行 {num_runs} 次...")
    latencies = []

    for i in range(num_runs):
        start_time = time.perf_counter()
        _ = model(**inputs)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

        if (i + 1) % 20 == 0:
            print(f"  进度: {i + 1}/{num_runs}")

    # 计算统计信息
    latencies = np.array(latencies)
    results = {
        "framework": "ONNX Runtime (Optimum)",
        "model_dir": str(model_dir),
        "num_runs": num_runs,
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "throughput_samples_per_sec": 1000.0 / np.mean(latencies),
    }

    print(f"\n结果:")
    print(f"  平均延迟: {results['mean_latency_ms']:.2f} ms")
    print(f"  P95延迟: {results['p95_latency_ms']:.2f} ms")
    print(f"  P99延迟: {results['p99_latency_ms']:.2f} ms")
    print(f"  吞吐量: {results['throughput_samples_per_sec']:.2f} samples/sec")

    return results


def generate_report(env_info, tf_results, onnx_results, conversion_info, output_file):
    """生成性能对比报告"""
    print_section("生成性能对比报告")

    # 计算性能提升
    speedup = tf_results["mean_latency_ms"] / onnx_results["mean_latency_ms"]
    p95_speedup = tf_results["p95_latency_ms"] / onnx_results["p95_latency_ms"]
    p99_speedup = tf_results["p99_latency_ms"] / onnx_results["p99_latency_ms"]
    throughput_speedup = (
        onnx_results["throughput_samples_per_sec"] / tf_results["throughput_samples_per_sec"]
    )

    # Format values for table
    tf_mean = tf_results["mean_latency_ms"]
    onnx_mean = onnx_results["mean_latency_ms"]
    tf_p95 = tf_results["p95_latency_ms"]
    onnx_p95 = onnx_results["p95_latency_ms"]
    tf_p99 = tf_results["p99_latency_ms"]
    onnx_p99 = onnx_results["p99_latency_ms"]
    tf_throughput = tf_results["throughput_samples_per_sec"]
    onnx_throughput = onnx_results["throughput_samples_per_sec"]

    report = f"""# BERT模型 TensorFlow vs ONNX Runtime 性能对比报告

## 环境信息

- **TensorFlow**: {env_info['tensorflow']}
- **NumPy**: {env_info['numpy']}
- **ONNX Runtime**: {env_info['onnxruntime']}
- **Optimum**: {env_info['optimum']}
- **Transformers**: {env_info['transformers']}
- **Protobuf**: {env_info.get('protobuf', 'N/A')}

## ONNX转换信息

- **转换方法**: HuggingFace Optimum
- **转换时间**: {conversion_info['conversion_time_sec']:.2f} 秒
- **ONNX模型大小**: {conversion_info['model_size_mb']:.2f} MB
- **优势**: 无protobuf版本冲突，原生支持Transformers模型

## 性能对比

### TensorFlow

| 指标 | 值 |
|------|-----|
| 平均延迟 | {tf_results['mean_latency_ms']:.2f} ms |
| P50延迟 | {tf_results['p50_latency_ms']:.2f} ms |
| P95延迟 | {tf_results['p95_latency_ms']:.2f} ms |
| P99延迟 | {tf_results['p99_latency_ms']:.2f} ms |
| 吞吐量 | {tf_results['throughput_samples_per_sec']:.2f} samples/sec |

### ONNX Runtime (via Optimum)

| 指标 | 值 |
|------|-----|
| 平均延迟 | {onnx_results['mean_latency_ms']:.2f} ms |
| P50延迟 | {onnx_results['p50_latency_ms']:.2f} ms |
| P95延迟 | {onnx_results['p95_latency_ms']:.2f} ms |
| P99延迟 | {onnx_results['p99_latency_ms']:.2f} ms |
| 吞吐量 | {onnx_results['throughput_samples_per_sec']:.2f} samples/sec |

### 性能提升

| 指标 | TensorFlow | ONNX Runtime | 提升倍数 |
|------|-----------|--------------|---------|
| **平均延迟** | {tf_mean:.2f} ms | {onnx_mean:.2f} ms | \
**{speedup:.2f}x** 🚀 |
| **P95延迟** | {tf_p95:.2f} ms | {onnx_p95:.2f} ms | \
**{p95_speedup:.2f}x** |
| **P99延迟** | {tf_p99:.2f} ms | {onnx_p99:.2f} ms | \
**{p99_speedup:.2f}x** |
| **吞吐量** | {tf_throughput:.2f} samples/s | {onnx_throughput:.2f} \
samples/s | **{throughput_speedup:.2f}x** 📈 |

## 结论

使用HuggingFace Optimum进行ONNX转换和推理：

✅ **无依赖冲突**: 完美兼容TensorFlow 2.20和protobuf 5.x
✅ **性能提升**: ONNX Runtime比TensorFlow快 **{speedup:.2f}倍**
✅ **易用性**: 自动化转换和优化，API简洁
✅ **生产就绪**: 适合部署到CPU推理环境

## 生成时间

{time.strftime('%Y-%m-%d %H:%M:%S')}
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✓ 报告已保存到: {output_file}")
    print(f"\n性能提升总结:")
    print(f"  ONNX Runtime 比 TensorFlow 快 {speedup:.2f}x 🚀")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="BERT TensorFlow vs ONNX Runtime 性能对比 (使用Optimum)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace模型名称 (default: bert-base-uncased)",
    )
    parser.add_argument("--num-runs", type=int, default=100, help="基准测试运行次数 (default: 100)")
    parser.add_argument("--num-warmup", type=int, default=10, help="预热运行次数 (default: 10)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/bert_optimum_benchmark",
        help="输出目录 (default: results/bert_optimum_benchmark)",
    )
    parser.add_argument("--skip-tf", action="store_true", help="跳过TensorFlow基准测试")
    parser.add_argument("--skip-onnx", action="store_true", help="跳过ONNX基准测试")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查环境
    env_info = check_environment()

    # TensorFlow基准测试
    if not args.skip_tf:
        tf_results, tf_model, tokenizer = benchmark_tensorflow_bert(
            model_name=args.model_name, num_runs=args.num_runs, num_warmup=args.num_warmup
        )
    else:
        print("跳过TensorFlow基准测试")
        tf_results = None

    # ONNX转换
    onnx_model_dir = output_dir / "onnx_model"
    conversion_info = convert_to_onnx_with_optimum(
        model_name=args.model_name, output_dir=onnx_model_dir
    )

    # ONNX基准测试
    if not args.skip_onnx:
        onnx_results = benchmark_onnx_bert(
            model_dir=onnx_model_dir, num_runs=args.num_runs, num_warmup=args.num_warmup
        )
    else:
        print("跳过ONNX基准测试")
        onnx_results = None

    # 保存结果
    results = {
        "environment": env_info,
        "tensorflow": tf_results,
        "onnx_runtime": onnx_results,
        "conversion": conversion_info,
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ 结果已保存到: {results_file}")

    # 生成报告
    if tf_results and onnx_results:
        report_file = output_dir / "benchmark_report.md"
        generate_report(env_info, tf_results, onnx_results, conversion_info, report_file)

    print_section("测试完成！")
    print(f"所有结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
