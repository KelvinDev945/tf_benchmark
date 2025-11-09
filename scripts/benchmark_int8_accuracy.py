#!/usr/bin/env python3
"""
INT8 量化性能与准确率对比工具

对比FP32原始模型和INT8量化模型的：
1. 推理延迟和吞吐量
2. 模型准确率
3. 模型大小

Usage:
    python3 scripts/benchmark_int8_accuracy.py --model-type mobilenet
    python3 scripts/benchmark_int8_accuracy.py --model-type custom --model-path path/to/model
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*70)
print("INT8 量化 vs FP32 准确率和性能对比")
print("="*70)
print(f"TensorFlow 版本: {tf.__version__}")
print()


def create_test_model(model_type='mobilenet', input_shape=(224, 224, 3), num_classes=10):
    """
    创建测试模型

    Args:
        model_type: 模型类型 (mobilenet, simple_cnn)
        input_shape: 输入形状
        num_classes: 分类数

    Returns:
        编译后的模型
    """
    print(f"创建测试模型: {model_type}")

    if model_type == 'mobilenet':
        # 使用MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    elif model_type == 'simple_cnn':
        # 简单CNN模型
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"✓ 模型创建完成")
    print(f"  参数总数: {model.count_params():,}")

    return model


def create_representative_dataset(input_shape, num_samples=100):
    """创建代表性数据集用于量化校准"""
    def representative_dataset_gen():
        for _ in range(num_samples):
            data = np.random.random_sample((1,) + input_shape).astype(np.float32)
            yield [data]
    return representative_dataset_gen


def quantize_to_int8(model, input_shape, output_path, num_calibration_samples=100):
    """
    将模型量化为INT8 TFLite

    Args:
        model: Keras模型
        input_shape: 输入形状
        output_path: 输出路径
        num_calibration_samples: 校准样本数

    Returns:
        量化模型路径
    """
    print(f"\n量化模型为INT8...")

    # 创建代表性数据集
    representative_dataset_gen = create_representative_dataset(
        input_shape,
        num_calibration_samples
    )

    # 创建转换器
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen

    # INT8量化设置
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    try:
        tflite_model = converter.convert()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✓ INT8量化完成")
        print(f"  大小: {size_mb:.2f} MB")
        print(f"  路径: {output_path}")

        return output_path

    except Exception as e:
        print(f"✗ INT8量化失败: {e}")
        print("\n尝试动态范围量化作为替代...")
        return quantize_dynamic(model, output_path)


def quantize_dynamic(model, output_path):
    """动态范围量化（权重INT8，激活FP32）"""
    print(f"\n动态范围量化...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    output_path = Path(output_path)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"✓ 动态量化完成")
    print(f"  大小: {size_mb:.2f} MB")

    return output_path


def benchmark_keras_model(model, test_data, test_labels, num_warmup=10, num_test=50):
    """
    测试Keras模型性能

    Returns:
        dict: 包含延迟、吞吐量、准确率的结果
    """
    print(f"\n{'='*70}")
    print("测试 FP32 Keras 模型")
    print(f"{'='*70}")

    # Warmup
    print(f"\n热身: {num_warmup} iterations...")
    for i in range(num_warmup):
        _ = model(test_data[i:i+1], training=False)
        if (i + 1) % 5 == 0:
            print(f"  Warmup: {i+1}/{num_warmup}")

    # 延迟测试
    print(f"\n延迟测试: {num_test} iterations...")
    latencies = []

    for i in range(num_test):
        start = time.perf_counter()
        _ = model(test_data[i:i+1], training=False)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        if (i + 1) % 10 == 0:
            print(f"  测试: {i+1}/{num_test}")

    latencies = np.array(latencies)

    # 准确率测试
    print(f"\n准确率测试...")
    predictions = model.predict(test_data, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_classes == test_labels)

    results = {
        "model_type": "FP32 Keras",
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_median_ms": float(np.median(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "throughput_samples_per_sec": num_test / (np.sum(latencies) / 1000),
        "accuracy": float(accuracy),
    }

    print(f"\n✓ FP32 Keras 测试完成")
    print(f"  延迟 (mean): {results['latency_mean_ms']:.2f} ms")
    print(f"  延迟 (p95):  {results['latency_p95_ms']:.2f} ms")
    print(f"  吞吐量:      {results['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"  准确率:      {results['accuracy']*100:.2f}%")

    return results


def benchmark_tflite_model(tflite_path, test_data, test_labels, num_warmup=10, num_test=50):
    """
    测试TFLite模型性能

    Returns:
        dict: 包含延迟、吞吐量、准确率的结果
    """
    print(f"\n{'='*70}")
    print("测试 INT8 TFLite 模型")
    print(f"{'='*70}")

    # 加载TFLite模型
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\n模型信息:")
    print(f"  输入: {input_details[0]['shape']}, {input_details[0]['dtype']}")
    print(f"  输出: {output_details[0]['shape']}, {output_details[0]['dtype']}")

    # 获取输入输出的缩放参数（用于INT8）
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    # Warmup
    print(f"\n热身: {num_warmup} iterations...")
    for i in range(num_warmup):
        # 量化输入
        input_data = test_data[i:i+1]
        if input_details[0]['dtype'] == np.uint8:
            input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)
        elif input_details[0]['dtype'] == np.int8:
            input_data = (input_data / input_scale + input_zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        if (i + 1) % 5 == 0:
            print(f"  Warmup: {i+1}/{num_warmup}")

    # 延迟测试
    print(f"\n延迟测试: {num_test} iterations...")
    latencies = []
    predictions = []

    for i in range(num_test):
        # 准备输入
        input_data = test_data[i:i+1]
        if input_details[0]['dtype'] == np.uint8:
            input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)
        elif input_details[0]['dtype'] == np.int8:
            input_data = (input_data / input_scale + input_zero_point).astype(np.int8)

        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        # 反量化输出
        if output_details[0]['dtype'] == np.uint8 or output_details[0]['dtype'] == np.int8:
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

        predictions.append(np.argmax(output_data))

        if (i + 1) % 10 == 0:
            print(f"  测试: {i+1}/{num_test}")

    latencies = np.array(latencies)
    predictions = np.array(predictions)

    # 计算准确率
    accuracy = np.mean(predictions == test_labels[:num_test])

    results = {
        "model_type": "INT8 TFLite",
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_median_ms": float(np.median(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "throughput_samples_per_sec": num_test / (np.sum(latencies) / 1000),
        "accuracy": float(accuracy),
    }

    print(f"\n✓ INT8 TFLite 测试完成")
    print(f"  延迟 (mean): {results['latency_mean_ms']:.2f} ms")
    print(f"  延迟 (p95):  {results['latency_p95_ms']:.2f} ms")
    print(f"  吞吐量:      {results['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"  准确率:      {results['accuracy']*100:.2f}%")

    return results


def generate_comparison_report(fp32_results, int8_results, output_dir):
    """生成对比报告"""
    print(f"\n{'='*70}")
    print("生成对比报告")
    print(f"{'='*70}")

    report_file = output_dir / "int8_vs_fp32_report.md"

    with open(report_file, "w") as f:
        f.write("# INT8 量化 vs FP32 性能和准确率对比\n\n")

        f.write("## 性能对比\n\n")
        f.write("| 指标 | FP32 | INT8 | 变化 |\n")
        f.write("|------|------|------|------|\n")

        # 延迟对比
        speedup_mean = fp32_results['latency_mean_ms'] / int8_results['latency_mean_ms']
        speedup_p95 = fp32_results['latency_p95_ms'] / int8_results['latency_p95_ms']

        f.write(f"| 延迟 (mean) | {fp32_results['latency_mean_ms']:.2f} ms | "
                f"{int8_results['latency_mean_ms']:.2f} ms | "
                f"{speedup_mean:.2f}x 🚀 |\n")

        f.write(f"| 延迟 (median) | {fp32_results['latency_median_ms']:.2f} ms | "
                f"{int8_results['latency_median_ms']:.2f} ms | "
                f"{fp32_results['latency_median_ms']/int8_results['latency_median_ms']:.2f}x |\n")

        f.write(f"| 延迟 (p95) | {fp32_results['latency_p95_ms']:.2f} ms | "
                f"{int8_results['latency_p95_ms']:.2f} ms | "
                f"{speedup_p95:.2f}x |\n")

        # 吞吐量对比
        throughput_improvement = int8_results['throughput_samples_per_sec'] / fp32_results['throughput_samples_per_sec']

        f.write(f"| 吞吐量 | {fp32_results['throughput_samples_per_sec']:.2f} samples/s | "
                f"{int8_results['throughput_samples_per_sec']:.2f} samples/s | "
                f"{throughput_improvement:.2f}x 📈 |\n")

        f.write("\n## 准确率对比\n\n")
        f.write("| 指标 | FP32 | INT8 | 差异 |\n")
        f.write("|------|------|------|------|\n")

        accuracy_diff = (fp32_results['accuracy'] - int8_results['accuracy']) * 100

        f.write(f"| 准确率 | {fp32_results['accuracy']*100:.2f}% | "
                f"{int8_results['accuracy']*100:.2f}% | "
                f"{accuracy_diff:+.2f}% |\n")

        f.write("\n## 总结\n\n")

        if speedup_mean > 1.0:
            f.write(f"✅ **INT8量化提速 {speedup_mean:.2f}x**\n\n")
        else:
            f.write(f"⚠️ **INT8量化未提速** ({1/speedup_mean:.2f}x slower)\n\n")

        if abs(accuracy_diff) < 1.0:
            f.write(f"✅ **准确率损失可忽略** ({accuracy_diff:+.2f}%)\n\n")
        elif accuracy_diff > 0 and accuracy_diff < 3.0:
            f.write(f"⚠️ **准确率轻微下降** ({accuracy_diff:+.2f}%)\n\n")
        else:
            f.write(f"❌ **准确率明显下降** ({accuracy_diff:+.2f}%)\n\n")

        f.write("### 关键指标\n\n")
        f.write(f"- 平均延迟提升: **{speedup_mean:.2f}x**\n")
        f.write(f"- P95延迟提升: **{speedup_p95:.2f}x**\n")
        f.write(f"- 吞吐量提升: **{throughput_improvement:.2f}x**\n")
        f.write(f"- 准确率变化: **{accuracy_diff:+.2f}%**\n\n")

        f.write("## 建议\n\n")

        if speedup_mean > 1.5 and abs(accuracy_diff) < 2.0:
            f.write("✅ **推荐使用INT8量化** - 性能提升显著且准确率损失可接受\n")
        elif speedup_mean > 1.2 and abs(accuracy_diff) < 1.0:
            f.write("✅ **可以使用INT8量化** - 性能和准确率都在可接受范围\n")
        elif abs(accuracy_diff) > 3.0:
            f.write("⚠️ **谨慎使用INT8量化** - 准确率下降较多，建议重新校准或使用Float16量化\n")
        else:
            f.write("⚠️ **评估使用场景** - 根据应用需求权衡性能和准确率\n")

    print(f"✓ 报告已保存: {report_file}")

    return report_file


def main():
    parser = argparse.ArgumentParser(description="INT8 vs FP32 Benchmark")
    parser.add_argument("--model-type", type=str, default="simple_cnn",
                        choices=['mobilenet', 'simple_cnn', 'custom'],
                        help="Model type to test")
    parser.add_argument("--model-path", type=str, help="Path to custom model")
    parser.add_argument("--input-shape", type=str, default="28,28,1",
                        help="Input shape (comma-separated)")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--num-test-samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--num-warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--num-test", type=int, default=50, help="Test iterations")
    parser.add_argument("--output", type=str, default="./results/int8_benchmark",
                        help="Output directory")

    args = parser.parse_args()

    # 解析输入形状
    input_shape = tuple(map(int, args.input_shape.split(',')))

    print(f"配置:")
    print(f"  模型类型: {args.model_type}")
    print(f"  输入形状: {input_shape}")
    print(f"  类别数: {args.num_classes}")
    print(f"  测试样本: {args.num_test_samples}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建或加载模型
    if args.model_type == 'custom' and args.model_path:
        print(f"\n加载自定义模型: {args.model_path}")
        model = tf.keras.models.load_model(args.model_path)
    else:
        model = create_test_model(args.model_type, input_shape, args.num_classes)

    # 创建测试数据
    print(f"\n创建测试数据...")
    test_data = np.random.random((args.num_test_samples,) + input_shape).astype(np.float32)
    test_labels = np.random.randint(0, args.num_classes, args.num_test_samples)
    print(f"✓ 测试数据准备完成: {test_data.shape}")

    # 保存FP32模型
    fp32_model_path = output_dir / "model_fp32.h5"
    model.save(fp32_model_path)
    fp32_size_mb = os.path.getsize(fp32_model_path) / (1024 * 1024)
    print(f"\n✓ FP32模型已保存: {fp32_model_path}")
    print(f"  大小: {fp32_size_mb:.2f} MB")

    # 量化为INT8
    int8_model_path = output_dir / "model_int8.tflite"
    quantize_to_int8(model, input_shape, int8_model_path)

    int8_size_mb = os.path.getsize(int8_model_path) / (1024 * 1024)
    compression_ratio = fp32_size_mb / int8_size_mb
    print(f"\n✓ 模型大小对比:")
    print(f"  FP32: {fp32_size_mb:.2f} MB")
    print(f"  INT8: {int8_size_mb:.2f} MB")
    print(f"  压缩比: {compression_ratio:.2f}x")

    # 测试FP32模型
    fp32_results = benchmark_keras_model(
        model, test_data, test_labels,
        args.num_warmup, args.num_test
    )

    # 测试INT8模型
    int8_results = benchmark_tflite_model(
        int8_model_path, test_data, test_labels,
        args.num_warmup, args.num_test
    )

    # 保存结果
    results = {
        "fp32": fp32_results,
        "int8": int8_results,
        "model_size_mb": {
            "fp32": fp32_size_mb,
            "int8": int8_size_mb,
            "compression_ratio": compression_ratio
        }
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ 结果已保存: {results_file}")

    # 生成报告
    report_file = generate_comparison_report(fp32_results, int8_results, output_dir)

    # 最终总结
    print(f"\n{'='*70}")
    print("✓ INT8 vs FP32 对比测试完成!")
    print(f"{'='*70}")
    print(f"\n结果文件:")
    print(f"  - FP32模型: {fp32_model_path}")
    print(f"  - INT8模型: {int8_model_path}")
    print(f"  - 测试结果: {results_file}")
    print(f"  - 对比报告: {report_file}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
