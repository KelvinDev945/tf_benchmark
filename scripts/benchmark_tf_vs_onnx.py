#!/usr/bin/env python3
"""
TensorFlow vs ONNX 性能对比测试

对比相同模型在TensorFlow和ONNX Runtime下的推理性能
解决 TODO.md Issue #3
"""

import os
import sys
import time
import json
import argparse
import subprocess
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

    try:
        import tensorflow as tf
        env_info['tensorflow'] = tf.__version__
        print(f"✓ TensorFlow: {tf.__version__}")
    except Exception as e:
        print(f"✗ TensorFlow: {e}")
        sys.exit(1)

    try:
        import numpy as np
        env_info['numpy'] = np.__version__
        print(f"✓ NumPy: {np.__version__}")
    except Exception as e:
        print(f"✗ NumPy: {e}")

    try:
        import tf2onnx
        env_info['tf2onnx'] = tf2onnx.__version__
        print(f"✓ tf2onnx: {tf2onnx.__version__}")
    except Exception as e:
        print(f"✗ tf2onnx: {e}")
        sys.exit(1)

    try:
        import onnxruntime as ort
        env_info['onnxruntime'] = ort.__version__
        print(f"✓ ONNXRuntime: {ort.__version__}")
    except Exception as e:
        print(f"✗ ONNXRuntime: {e}")
        sys.exit(1)

    try:
        import google.protobuf
        env_info['protobuf'] = google.protobuf.__version__
        print(f"✓ Protobuf: {google.protobuf.__version__}")
    except Exception as e:
        print(f"✗ Protobuf: {e}")

    return env_info


def create_test_model(model_type="cnn"):
    """创建测试模型"""
    import tensorflow as tf

    if model_type == "cnn":
        print("创建CNN模型...")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        input_shape = (1, 28, 28, 1)

    elif model_type == "dense":
        print("创建Dense模型...")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(784,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        input_shape = (1, 784)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"✓ 模型创建完成，参数总数: {model.count_params():,}")
    return model, input_shape


def benchmark_tensorflow(model, test_data, num_runs=100, num_warmup=10):
    """测试TensorFlow模型性能"""
    print_section("TensorFlow 性能测试")

    # 热身
    print(f"热身: {num_warmup} iterations...")
    for _ in range(num_warmup):
        _ = model(test_data, training=False)

    # 性能测试
    print(f"性能测试: {num_runs} iterations...")
    latencies = []

    for i in range(num_runs):
        start = time.perf_counter()
        _ = model(test_data, training=False)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)

        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{num_runs}")

    # 统计
    latencies = np.array(latencies)
    results = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_samples_per_sec": float(1000.0 / np.mean(latencies))
    }

    print("\n✓ TensorFlow 测试完成")
    print(f"  平均延迟: {results['mean_ms']:.2f} ms")
    print(f"  P95延迟: {results['p95_ms']:.2f} ms")
    print(f"  吞吐量: {results['throughput_samples_per_sec']:.2f} samples/sec")

    return results


def convert_to_onnx(model, output_path):
    """将TensorFlow模型转换为ONNX"""
    print_section("转换为ONNX")

    import tensorflow as tf

    # 先保存为SavedModel
    saved_model_path = Path(output_path).parent / "temp_savedmodel"
    model.export(saved_model_path)
    print(f"✓ SavedModel已保存: {saved_model_path}")

    # 转换为ONNX
    print("转换为ONNX...")
    start_time = time.time()

    cmd = [
        "python3", "-m", "tf2onnx.convert",
        "--saved-model", str(saved_model_path),
        "--output", str(output_path),
        "--opset", "13"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"✗ ONNX转换失败")
        print(f"错误: {result.stderr}")
        return None

    conversion_time = time.time() - start_time
    file_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"✓ ONNX转换成功")
    print(f"  转换时间: {conversion_time:.2f}s")
    print(f"  模型大小: {file_size:.2f} MB")
    print(f"  输出路径: {output_path}")

    return {
        "conversion_time": conversion_time,
        "model_size_mb": file_size
    }


def benchmark_onnx(onnx_path, test_data, num_runs=100, num_warmup=10):
    """测试ONNX模型性能"""
    print_section("ONNX Runtime 性能测试")

    import onnxruntime as ort

    # 创建会话
    sess = ort.InferenceSession(str(onnx_path))

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    print(f"  输入名称: {input_name}")
    print(f"  输出名称: {output_name}")

    # 热身
    print(f"\n热身: {num_warmup} iterations...")
    for _ in range(num_warmup):
        _ = sess.run([output_name], {input_name: test_data})

    # 性能测试
    print(f"性能测试: {num_runs} iterations...")
    latencies = []

    for i in range(num_runs):
        start = time.perf_counter()
        _ = sess.run([output_name], {input_name: test_data})
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)

        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{num_runs}")

    # 统计
    latencies = np.array(latencies)
    results = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_samples_per_sec": float(1000.0 / np.mean(latencies))
    }

    print("\n✓ ONNX Runtime 测试完成")
    print(f"  平均延迟: {results['mean_ms']:.2f} ms")
    print(f"  P95延迟: {results['p95_ms']:.2f} ms")
    print(f"  吞吐量: {results['throughput_samples_per_sec']:.2f} samples/sec")

    return results


def generate_report(env_info, tf_results, onnx_results, conversion_info, output_dir):
    """生成对比报告"""
    report_path = Path(output_dir) / "tf_vs_onnx_report.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# TensorFlow vs ONNX Runtime 性能对比\n\n")
        f.write(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 环境信息\n\n")
        for key, value in env_info.items():
            f.write(f"- {key}: {value}\n")

        f.write("\n## ONNX转换信息\n\n")
        if conversion_info:
            f.write(f"- 转换时间: {conversion_info['conversion_time']:.2f}s\n")
            f.write(f"- 模型大小: {conversion_info['model_size_mb']:.2f} MB\n")

        f.write("\n## 延迟对比\n\n")
        f.write("| 指标 | TensorFlow | ONNX Runtime | 比率 |\n")
        f.write("|------|-----------|--------------|------|\n")

        speedup_mean = tf_results['mean_ms'] / onnx_results['mean_ms']
        speedup_p95 = tf_results['p95_ms'] / onnx_results['p95_ms']

        f.write(f"| 平均延迟 | {tf_results['mean_ms']:.2f} ms | ")
        f.write(f"{onnx_results['mean_ms']:.2f} ms | ")
        f.write(f"{speedup_mean:.2f}x {'🚀' if speedup_mean > 1 else ''} |\n")

        f.write(f"| 中位延迟 | {tf_results['median_ms']:.2f} ms | ")
        f.write(f"{onnx_results['median_ms']:.2f} ms | ")
        speedup_median = tf_results['median_ms'] / onnx_results['median_ms']
        f.write(f"{speedup_median:.2f}x |\n")

        f.write(f"| P95延迟 | {tf_results['p95_ms']:.2f} ms | ")
        f.write(f"{onnx_results['p95_ms']:.2f} ms | ")
        f.write(f"{speedup_p95:.2f}x |\n")

        f.write(f"| P99延迟 | {tf_results['p99_ms']:.2f} ms | ")
        f.write(f"{onnx_results['p99_ms']:.2f} ms | ")
        speedup_p99 = tf_results['p99_ms'] / onnx_results['p99_ms']
        f.write(f"{speedup_p99:.2f}x |\n")

        f.write("\n## 吞吐量对比\n\n")
        f.write("| 框架 | 吞吐量 (samples/s) |\n")
        f.write("|------|-------------------|\n")
        f.write(f"| TensorFlow | {tf_results['throughput_samples_per_sec']:.2f} |\n")
        f.write(f"| ONNX Runtime | {onnx_results['throughput_samples_per_sec']:.2f} |\n")

        throughput_speedup = onnx_results['throughput_samples_per_sec'] / tf_results['throughput_samples_per_sec']
        f.write(f"\n**吞吐量提升**: {throughput_speedup:.2f}x\n")

        f.write("\n## 总结\n\n")
        if speedup_mean > 1:
            f.write(f"✅ **ONNX Runtime 平均延迟更低，提速 {speedup_mean:.2f}x**\n\n")
        else:
            f.write(f"⚠️ TensorFlow 在此模型上表现更好\n\n")

        f.write("### 关键发现\n\n")
        f.write(f"- 平均延迟提升: {speedup_mean:.2f}x\n")
        f.write(f"- P95延迟提升: {speedup_p95:.2f}x\n")
        f.write(f"- 吞吐量提升: {throughput_speedup:.2f}x\n")

        f.write("\n### 建议\n\n")
        if speedup_mean > 1.5:
            f.write("✅ **推荐使用ONNX Runtime进行生产部署**\n\n")
            f.write("ONNX Runtime在此模型上有显著的性能优势\n")
        elif speedup_mean > 1.1:
            f.write("✅ **ONNX Runtime有一定优势**\n\n")
            f.write("建议根据具体场景选择合适的推理引擎\n")
        else:
            f.write("⚠️ **性能差异不明显**\n\n")
            f.write("可以根据部署便利性和生态系统选择推理引擎\n")

    print(f"\n✓ 报告已保存: {report_path}")
    return str(report_path)


def main():
    parser = argparse.ArgumentParser(description="TensorFlow vs ONNX性能对比")
    parser.add_argument("--model-type", default="cnn", choices=["cnn", "dense"],
                       help="模型类型")
    parser.add_argument("--output-dir", default="results/tf_vs_onnx_benchmark",
                       help="输出目录")
    parser.add_argument("--num-runs", type=int, default=100,
                       help="性能测试迭代次数")
    parser.add_argument("--num-warmup", type=int, default=10,
                       help="热身迭代次数")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 环境检查
    env_info = check_environment()

    # 创建测试模型
    print_section("创建测试模型")
    import tensorflow as tf
    model, input_shape = create_test_model(args.model_type)

    # 创建测试数据
    print("\n创建测试数据...")
    test_data = np.random.randn(*input_shape).astype(np.float32)
    print(f"✓ 测试数据形状: {test_data.shape}")

    # 测试TensorFlow性能
    tf_results = benchmark_tensorflow(
        model, test_data,
        num_runs=args.num_runs,
        num_warmup=args.num_warmup
    )

    # 转换为ONNX
    onnx_path = output_dir / "model.onnx"
    conversion_info = convert_to_onnx(model, onnx_path)

    if not conversion_info:
        print("✗ ONNX转换失败，无法进行性能对比")
        return

    # 测试ONNX性能
    onnx_results = benchmark_onnx(
        onnx_path, test_data,
        num_runs=args.num_runs,
        num_warmup=args.num_warmup
    )

    # 保存结果
    results = {
        "environment": env_info,
        "tensorflow": tf_results,
        "onnx": onnx_results,
        "conversion": conversion_info
    }

    results_json = output_dir / "results.json"
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 结果已保存: {results_json}")

    # 生成报告
    report_path = generate_report(env_info, tf_results, onnx_results, conversion_info, output_dir)

    # 打印总结
    print_section("✓ 测试完成!")
    print(f"\n结果文件:")
    print(f"  - JSON结果: {results_json}")
    print(f"  - 对比报告: {report_path}")

    speedup = tf_results['mean_ms'] / onnx_results['mean_ms']
    print(f"\n性能提升: {speedup:.2f}x")

    if speedup > 1:
        print(f"✅ ONNX Runtime 比 TensorFlow 快 {speedup:.2f}x")
    else:
        print(f"⚠️ TensorFlow 在此模型上表现更好")


if __name__ == "__main__":
    main()
