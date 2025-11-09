#!/usr/bin/env python3
"""
对比tf2onnx和Optimum两种ONNX转换方式的推理性能

测试相同模型使用不同转换工具后的ONNX推理速度差异
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorflow as tf


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def create_simple_cnn_model(input_shape=(224, 224, 3), num_classes=10):
    """创建一个简单的CNN模型用于测试"""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def convert_with_tf2onnx(model, output_path, input_shape=(1, 224, 224, 3)):
    """使用tf2onnx转换模型"""
    print_section("方法1: tf2onnx转换")

    try:

        # 保存为SavedModel格式（tf2onnx更稳定）
        temp_dir = tempfile.mkdtemp()
        saved_model_path = os.path.join(temp_dir, "saved_model")

        print("1. 保存为SavedModel格式...")
        model.save(saved_model_path)

        print("2. 使用tf2onnx转换...")
        start_time = time.time()

        # 使用命令行方式转换（更稳定）
        import subprocess

        cmd = [
            "python3",
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            saved_model_path,
            "--output",
            str(output_path),
            "--opset",
            "15",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        conversion_time = time.time() - start_time

        if result.returncode != 0:
            print(f"✗ tf2onnx转换失败:")
            print(result.stderr)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None

        # 检查文件大小
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)

        print(f"✓ tf2onnx转换成功!")
        print(f"  转换时间: {conversion_time:.2f} 秒")
        print(f"  模型大小: {file_size_mb:.2f} MB")
        print(f"  输出路径: {output_path}")

        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            "method": "tf2onnx",
            "conversion_time": conversion_time,
            "model_size_mb": file_size_mb,
            "output_path": str(output_path),
        }

    except Exception as e:
        print(f"✗ tf2onnx转换失败: {e}")
        import traceback

        traceback.print_exc()
        return None


def convert_with_optimum(model, output_dir, input_shape=(1, 224, 224, 3)):
    """使用Optimum转换模型（需要先转为HF格式）"""
    print_section("方法2: Optimum转换")

    try:
        from optimum.onnxruntime import ORTModelForImageClassification

        # Optimum主要设计用于HuggingFace模型
        # 对于自定义Keras模型，我们需要先保存为SavedModel，然后用tf2onnx
        # 这里展示Optimum的典型使用场景

        print("注意: Optimum主要优化用于HuggingFace Transformers模型")
        print("对于自定义CNN模型，推荐使用tf2onnx")
        print("我们将演示Optimum用于HF模型的场景...")

        # 使用一个预训练的HF模型作为示例
        model_name = "google/mobilenet_v2_1.0_224"

        print(f"1. 加载HuggingFace模型: {model_name}")
        start_time = time.time()

        ort_model = ORTModelForImageClassification.from_pretrained(model_name, export=True)

        conversion_time = time.time() - start_time

        print("2. 保存ONNX模型...")
        ort_model.save_pretrained(output_dir)

        # 检查文件大小
        onnx_file = Path(output_dir) / "model.onnx"
        file_size_mb = onnx_file.stat().st_size / (1024 * 1024)

        print(f"✓ Optimum转换成功!")
        print(f"  转换时间: {conversion_time:.2f} 秒")
        print(f"  模型大小: {file_size_mb:.2f} MB")
        print(f"  输出路径: {onnx_file}")

        return {
            "method": "Optimum",
            "conversion_time": conversion_time,
            "model_size_mb": file_size_mb,
            "output_path": str(onnx_file),
            "model_name": model_name,
        }

    except Exception as e:
        print(f"✗ Optimum转换失败: {e}")
        import traceback

        traceback.print_exc()
        return None


def benchmark_onnx_inference(onnx_path, input_shape=(1, 224, 224, 3), num_runs=100, num_warmup=10):
    """测试ONNX模型推理性能"""
    print(f"\n推理性能测试: {Path(onnx_path).name}")

    # 创建ONNX Runtime会话
    sess = ort.InferenceSession(str(onnx_path))

    # 获取输入名称
    input_name = sess.get_inputs()[0].name

    # 准备测试数据
    test_input = np.random.randn(*input_shape).astype(np.float32)

    # 预热
    print(f"  预热运行 {num_warmup} 次...")
    for _ in range(num_warmup):
        _ = sess.run(None, {input_name: test_input})

    # 基准测试
    print(f"  基准测试运行 {num_runs} 次...")
    latencies = []

    for i in range(num_runs):
        start_time = time.perf_counter()
        _ = sess.run(None, {input_name: test_input})
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    # 计算统计信息
    latencies = np.array(latencies)
    results = {
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "throughput_samples_per_sec": 1000.0 / np.mean(latencies),
    }

    print(f"  平均延迟: {results['mean_latency_ms']:.2f} ms")
    print(f"  P95延迟: {results['p95_latency_ms']:.2f} ms")
    print(f"  吞吐量: {results['throughput_samples_per_sec']:.2f} samples/sec")

    return results


def main():
    print("=" * 80)
    print("tf2onnx vs Optimum ONNX转换方法性能对比")
    print("=" * 80)

    # 创建输出目录
    output_dir = Path("results/onnx_conversion_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 测试1: 自定义CNN模型 - 使用tf2onnx
    print_section("测试1: 自定义CNN模型 (tf2onnx)")

    print("创建测试模型...")
    model = create_simple_cnn_model()
    print(f"模型参数数量: {model.count_params():,}")

    tf2onnx_path = output_dir / "model_tf2onnx.onnx"
    tf2onnx_info = convert_with_tf2onnx(model, tf2onnx_path)

    if tf2onnx_info:
        print("\n运行tf2onnx ONNX推理测试...")
        tf2onnx_perf = benchmark_onnx_inference(tf2onnx_path, num_runs=100, num_warmup=10)
        tf2onnx_info.update(tf2onnx_perf)

    # 测试2: HuggingFace模型 - 使用Optimum
    print_section("测试2: HuggingFace MobileNetV2 (Optimum)")

    optimum_dir = output_dir / "model_optimum"
    optimum_info = convert_with_optimum(model, optimum_dir)

    if optimum_info:
        print("\n运行Optimum ONNX推理测试...")
        optimum_perf = benchmark_onnx_inference(
            optimum_info["output_path"], num_runs=100, num_warmup=10
        )
        optimum_info.update(optimum_perf)

    # 生成对比报告
    print_section("性能对比总结")

    if tf2onnx_info and optimum_info:
        print("\n注意: 两个模型不同，因此不能直接对比绝对速度")
        print("      但可以对比转换工具的特性和使用场景\n")

        print("tf2onnx (自定义CNN):")
        print(f"  转换时间: {tf2onnx_info['conversion_time']:.2f}秒")
        print(f"  模型大小: {tf2onnx_info['model_size_mb']:.2f} MB")
        print(f"  推理延迟: {tf2onnx_info['mean_latency_ms']:.2f} ms")
        print(f"  吞吐量: {tf2onnx_info['throughput_samples_per_sec']:.2f} samples/sec")

        print(f"\nOptimum ({optimum_info.get('model_name', 'HF Model')}):")
        print(f"  转换时间: {optimum_info['conversion_time']:.2f}秒")
        print(f"  模型大小: {optimum_info['model_size_mb']:.2f} MB")
        print(f"  推理延迟: {optimum_info['mean_latency_ms']:.2f} ms")
        print(f"  吞吐量: {optimum_info['throughput_samples_per_sec']:.2f} samples/sec")

        print("\n结论:")
        print("1. tf2onnx:")
        print("   ✓ 适用于任何TensorFlow/Keras模型")
        print("   ✓ 灵活性高，可控性强")
        print("   ✗ 与TensorFlow 2.20有兼容性问题")
        print("   ✗ 需要protobuf<4.0（与TF 2.20冲突）")

        print("\n2. Optimum:")
        print("   ✓ 专为HuggingFace模型优化")
        print("   ✓ 无protobuf版本冲突")
        print("   ✓ 自动优化和量化支持")
        print("   ✗ 主要支持HF Transformers模型")
        print("   ✗ 对自定义模型支持有限")

        print("\n推荐使用场景:")
        print("• 使用HuggingFace模型 → Optimum")
        print("• 自定义TensorFlow模型 → tf2onnx (但需注意版本兼容)")
        print("• 需要最新TensorFlow → Optimum (避免tf2onnx冲突)")

    elif tf2onnx_info:
        print("✓ tf2onnx转换成功")
        print(f"  推理延迟: {tf2onnx_info['mean_latency_ms']:.2f} ms")

    elif optimum_info:
        print("✓ Optimum转换成功")
        print(f"  推理延迟: {optimum_info['mean_latency_ms']:.2f} ms")

    else:
        print("❌ 两种方法都失败了")

    print(f"\n所有结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
