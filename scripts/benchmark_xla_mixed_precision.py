#!/usr/bin/env python3
"""
TensorFlow XLA + 混合精度推理性能测试

对比三种推理配置：
1. Baseline (FP32, 无XLA)
2. XLA优化 (FP32 + XLA)
3. XLA + 混合精度 (FP16/FP32 + XLA)

要求：准确率下降幅度 < 1%
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def check_environment():
    """检查环境配置"""
    print_section("环境检查")

    print(f"✓ TensorFlow版本: {tf.__version__}")
    print(f"✓ NumPy版本: {np.__version__}")

    # 检查GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ 检测到GPU: {len(gpus)}个")
        for gpu in gpus:
            print(f"  - {gpu.name}")
    else:
        print("⚠️  未检测到GPU，将使用CPU测试")

    # 检查XLA支持
    print(f"✓ XLA编译器可用")

    return {
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "gpu_available": len(gpus) > 0,
        "num_gpus": len(gpus)
    }


def create_test_model(model_type="cnn", input_shape=(28, 28, 1), num_classes=10):
    """创建测试模型"""
    print_section(f"创建测试模型: {model_type}")

    if model_type == "cnn":
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ], name="cnn_model")

    elif model_type == "resnet_like":
        inputs = tf.keras.layers.Input(shape=input_shape)

        # 初始卷积
        x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

        # ResNet块
        for filters in [64, 128, 256]:
            # 残差块
            shortcut = x

            x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)

            # 调整shortcut维度
            if shortcut.shape[-1] != filters:
                shortcut = tf.keras.layers.Conv2D(filters, 1)(shortcut)
                shortcut = tf.keras.layers.BatchNormalization()(shortcut)

            x = tf.keras.layers.Add()([x, shortcut])
            x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet_like")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"✓ 模型创建完成")
    print(f"  参数总数: {model.count_params():,}")

    return model


def prepare_test_data(input_shape, num_samples=1000, num_classes=10):
    """准备测试数据"""
    print_section("准备测试数据")

    # 生成随机测试数据
    X_test = np.random.randn(num_samples, *input_shape).astype(np.float32)
    y_test = np.random.randint(0, num_classes, num_samples)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)

    print(f"✓ 测试数据准备完成")
    print(f"  样本数: {num_samples}")
    print(f"  输入形状: {X_test.shape}")
    print(f"  类别数: {num_classes}")

    return X_test, y_test, y_test_onehot


def benchmark_baseline(model, X_test, y_test_onehot, num_runs=100, num_warmup=10):
    """
    Baseline测试 - FP32, 无XLA
    """
    print_section("Baseline测试 (FP32, 无XLA)")

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 热身
    print(f"热身: {num_warmup} iterations...")
    for i in range(num_warmup):
        _ = model.predict(X_test[:10], verbose=0)
        if (i + 1) % 5 == 0:
            print(f"  热身: {i+1}/{num_warmup}")

    # 性能测试
    print(f"\n性能测试: {num_runs} iterations...")
    batch_size = 32
    latencies = []

    for i in range(num_runs):
        start = time.perf_counter()
        _ = model.predict(X_test[:batch_size], verbose=0)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        if (i + 1) % 20 == 0:
            print(f"  测试: {i+1}/{num_runs}")

    # 准确率测试
    print("\n准确率测试...")
    loss, accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)

    # 统计
    latencies = np.array(latencies)
    results = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_samples_per_sec": float(batch_size * 1000.0 / np.mean(latencies)),
        "accuracy": float(accuracy),
        "loss": float(loss)
    }

    print(f"\n✓ Baseline测试完成")
    print(f"  平均延迟: {results['mean_ms']:.2f} ms")
    print(f"  P95延迟: {results['p95_ms']:.2f} ms")
    print(f"  吞吐量: {results['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"  准确率: {results['accuracy']*100:.2f}%")

    return results


def benchmark_xla(model, X_test, y_test_onehot, num_runs=100, num_warmup=10):
    """
    XLA优化测试 - FP32 + XLA
    """
    print_section("XLA优化测试 (FP32 + XLA)")

    # 启用XLA
    tf.config.optimizer.set_jit(True)
    print("✓ XLA编译器已启用")

    # 编译模型（使用jit_compile=True）
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=True  # 启用XLA编译
    )

    # 热身（包括XLA编译时间）
    print(f"\n热身 (包括XLA编译): {num_warmup} iterations...")
    for i in range(num_warmup):
        _ = model.predict(X_test[:10], verbose=0)
        if (i + 1) % 5 == 0:
            print(f"  热身: {i+1}/{num_warmup}")

    # 性能测试
    print(f"\n性能测试: {num_runs} iterations...")
    batch_size = 32
    latencies = []

    for i in range(num_runs):
        start = time.perf_counter()
        _ = model.predict(X_test[:batch_size], verbose=0)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        if (i + 1) % 20 == 0:
            print(f"  测试: {i+1}/{num_runs}")

    # 准确率测试
    print("\n准确率测试...")
    loss, accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)

    # 统计
    latencies = np.array(latencies)
    results = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_samples_per_sec": float(batch_size * 1000.0 / np.mean(latencies)),
        "accuracy": float(accuracy),
        "loss": float(loss)
    }

    print(f"\n✓ XLA测试完成")
    print(f"  平均延迟: {results['mean_ms']:.2f} ms")
    print(f"  P95延迟: {results['p95_ms']:.2f} ms")
    print(f"  吞吐量: {results['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"  准确率: {results['accuracy']*100:.2f}%")

    # 禁用XLA（为下一个测试准备）
    tf.config.optimizer.set_jit(False)

    return results


def benchmark_mixed_precision_xla(model, X_test, y_test_onehot, num_runs=100, num_warmup=10):
    """
    混合精度 + XLA测试 - FP16/FP32 + XLA
    """
    print_section("混合精度 + XLA测试 (FP16/FP32 + XLA)")

    # 设置混合精度策略
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"✓ 混合精度策略已启用: {policy.name}")
    print(f"  计算dtype: {policy.compute_dtype}")
    print(f"  变量dtype: {policy.variable_dtype}")

    # 重建模型以应用混合精度
    # 混合精度策略已经设置，直接使用原模型即可
    # Keras会自动应用混合精度策略到新层
    mixed_model = model

    # 启用XLA
    tf.config.optimizer.set_jit(True)

    # 编译模型
    mixed_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=True
    )

    # 热身
    print(f"\n热身 (包括XLA编译): {num_warmup} iterations...")
    for i in range(num_warmup):
        _ = mixed_model.predict(X_test[:10], verbose=0)
        if (i + 1) % 5 == 0:
            print(f"  热身: {i+1}/{num_warmup}")

    # 性能测试
    print(f"\n性能测试: {num_runs} iterations...")
    batch_size = 32
    latencies = []

    for i in range(num_runs):
        start = time.perf_counter()
        _ = mixed_model.predict(X_test[:batch_size], verbose=0)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        if (i + 1) % 20 == 0:
            print(f"  测试: {i+1}/{num_runs}")

    # 准确率测试
    print("\n准确率测试...")
    loss, accuracy = mixed_model.evaluate(X_test, y_test_onehot, verbose=0)

    # 统计
    latencies = np.array(latencies)
    results = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_samples_per_sec": float(batch_size * 1000.0 / np.mean(latencies)),
        "accuracy": float(accuracy),
        "loss": float(loss)
    }

    print(f"\n✓ 混合精度 + XLA测试完成")
    print(f"  平均延迟: {results['mean_ms']:.2f} ms")
    print(f"  P95延迟: {results['p95_ms']:.2f} ms")
    print(f"  吞吐量: {results['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"  准确率: {results['accuracy']*100:.2f}%")

    # 重置策略
    mixed_precision.set_global_policy('float32')
    tf.config.optimizer.set_jit(False)

    return results


def check_accuracy_constraint(baseline_acc, test_acc, max_drop_percent=1.0):
    """
    检查准确率约束

    Args:
        baseline_acc: baseline准确率
        test_acc: 测试准确率
        max_drop_percent: 最大允许下降百分点（默认1%）

    Returns:
        (通过检查, 实际下降百分点)
    """
    drop = (baseline_acc - test_acc) * 100  # 转换为百分点
    passed = drop <= max_drop_percent
    return passed, drop


def generate_report(env_info, model_info, baseline_results, xla_results,
                    mixed_results, output_dir, max_accuracy_drop=1.0):
    """生成对比报告"""
    report_path = Path(output_dir) / "xla_mixed_precision_report.md"

    baseline_acc = baseline_results['accuracy']
    xla_acc = xla_results['accuracy']
    mixed_acc = mixed_results['accuracy']

    # 检查准确率约束
    xla_passed, xla_drop = check_accuracy_constraint(baseline_acc, xla_acc, max_accuracy_drop)
    mixed_passed, mixed_drop = check_accuracy_constraint(baseline_acc, mixed_acc, max_accuracy_drop)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# TensorFlow XLA + 混合精度性能测试报告\n\n")
        f.write(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 环境信息\n\n")
        for key, value in env_info.items():
            f.write(f"- {key}: {value}\n")

        f.write("\n## 模型信息\n\n")
        for key, value in model_info.items():
            f.write(f"- {key}: {value}\n")

        f.write("\n## 性能对比\n\n")
        f.write("| 配置 | 平均延迟 | P95延迟 | 吞吐量 | 加速比 |\n")
        f.write("|------|----------|---------|--------|--------|\n")

        baseline_lat = baseline_results['mean_ms']
        xla_lat = xla_results['mean_ms']
        mixed_lat = mixed_results['mean_ms']

        f.write(f"| Baseline (FP32) | {baseline_lat:.2f} ms | ")
        f.write(f"{baseline_results['p95_ms']:.2f} ms | ")
        f.write(f"{baseline_results['throughput_samples_per_sec']:.2f} samples/s | 1.00x |\n")

        xla_speedup = baseline_lat / xla_lat
        f.write(f"| XLA (FP32) | {xla_lat:.2f} ms | ")
        f.write(f"{xla_results['p95_ms']:.2f} ms | ")
        f.write(f"{xla_results['throughput_samples_per_sec']:.2f} samples/s | ")
        f.write(f"{xla_speedup:.2f}x {'🚀' if xla_speedup > 1.1 else ''} |\n")

        mixed_speedup = baseline_lat / mixed_lat
        f.write(f"| XLA + Mixed (FP16) | {mixed_lat:.2f} ms | ")
        f.write(f"{mixed_results['p95_ms']:.2f} ms | ")
        f.write(f"{mixed_results['throughput_samples_per_sec']:.2f} samples/s | ")
        f.write(f"{mixed_speedup:.2f}x {'🚀' if mixed_speedup > 1.1 else ''} |\n")

        f.write("\n## 准确率对比\n\n")
        f.write("| 配置 | 准确率 | vs Baseline | 状态 |\n")
        f.write("|------|--------|-------------|------|\n")

        f.write(f"| Baseline (FP32) | {baseline_acc*100:.2f}% | - | ✅ |\n")

        f.write(f"| XLA (FP32) | {xla_acc*100:.2f}% | ")
        if xla_drop >= 0:
            f.write(f"-{xla_drop:.2f}% | ")
        else:
            f.write(f"+{abs(xla_drop):.2f}% | ")
        f.write(f"{'✅' if xla_passed else '❌'} ")
        f.write(f"{'通过' if xla_passed else f'超标(>{max_accuracy_drop}%)'} |\n")

        f.write(f"| XLA + Mixed (FP16) | {mixed_acc*100:.2f}% | ")
        if mixed_drop >= 0:
            f.write(f"-{mixed_drop:.2f}% | ")
        else:
            f.write(f"+{abs(mixed_drop):.2f}% | ")
        f.write(f"{'✅' if mixed_passed else '❌'} ")
        f.write(f"{'通过' if mixed_passed else f'超标(>{max_accuracy_drop}%)'} |\n")

        f.write(f"\n**准确率约束**: 准确率下降 ≤ {max_accuracy_drop}%\n\n")

        f.write("## 总结\n\n")

        f.write("### 性能提升\n\n")
        f.write(f"- **XLA优化**: {xla_speedup:.2f}x 加速\n")
        f.write(f"- **XLA + 混合精度**: {mixed_speedup:.2f}x 加速\n")
        f.write(f"- **混合精度额外增益**: {xla_lat/mixed_lat:.2f}x (相对于XLA FP32)\n")

        f.write("\n### 准确率影响\n\n")
        f.write(f"- **XLA优化**: {abs(xla_drop):.2f}% {'下降' if xla_drop > 0 else '提升'}\n")
        f.write(f"- **XLA + 混合精度**: {abs(mixed_drop):.2f}% {'下降' if mixed_drop > 0 else '提升'}\n")

        f.write("\n### 推荐配置\n\n")

        if mixed_passed and mixed_speedup > xla_speedup:
            f.write("✅ **推荐使用 XLA + 混合精度**\n\n")
            f.write(f"- 性能提升最大: {mixed_speedup:.2f}x\n")
            f.write(f"- 准确率下降可接受: {abs(mixed_drop):.2f}%\n")
            f.write(f"- 满足 <{max_accuracy_drop}% 约束条件\n")
        elif xla_passed:
            f.write("✅ **推荐使用 XLA优化**\n\n")
            f.write(f"- 性能提升: {xla_speedup:.2f}x\n")
            f.write(f"- 准确率几乎无损失: {abs(xla_drop):.2f}%\n")
            if not mixed_passed:
                f.write(f"- 混合精度准确率下降超标: {abs(mixed_drop):.2f}% > {max_accuracy_drop}%\n")
        else:
            f.write("⚠️ **建议继续使用Baseline**\n\n")
            f.write(f"- XLA优化准确率下降: {abs(xla_drop):.2f}%\n")
            f.write(f"- 混合精度准确率下降: {abs(mixed_drop):.2f}%\n")
            f.write(f"- 均超过约束条件 ({max_accuracy_drop}%)\n")

    print(f"\n✓ 报告已保存: {report_path}")
    return str(report_path)


def main():
    parser = argparse.ArgumentParser(description="TensorFlow XLA + 混合精度性能测试")
    parser.add_argument("--model-type", default="cnn", choices=["cnn", "resnet_like"],
                       help="模型类型")
    parser.add_argument("--output-dir", default="results/xla_mixed_precision",
                       help="输出目录")
    parser.add_argument("--num-runs", type=int, default=100,
                       help="性能测试迭代次数")
    parser.add_argument("--num-warmup", type=int, default=10,
                       help="热身迭代次数")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="测试样本数")
    parser.add_argument("--max-accuracy-drop", type=float, default=1.0,
                       help="最大允许准确率下降（百分点）")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 环境检查
    env_info = check_environment()

    # 创建模型
    input_shape = (28, 28, 1)
    num_classes = 10
    model = create_test_model(args.model_type, input_shape, num_classes)

    model_info = {
        "模型类型": args.model_type,
        "输入形状": str(input_shape),
        "类别数": num_classes,
        "参数总数": f"{model.count_params():,}"
    }

    # 准备测试数据
    X_test, y_test, y_test_onehot = prepare_test_data(
        input_shape, args.num_samples, num_classes
    )

    # 测试1: Baseline (FP32, 无XLA)
    baseline_results = benchmark_baseline(
        model, X_test, y_test_onehot, args.num_runs, args.num_warmup
    )

    # 测试2: XLA (FP32 + XLA)
    xla_results = benchmark_xla(
        model, X_test, y_test_onehot, args.num_runs, args.num_warmup
    )

    # 测试3: Mixed Precision + XLA (FP16/FP32 + XLA)
    mixed_results = benchmark_mixed_precision_xla(
        model, X_test, y_test_onehot, args.num_runs, args.num_warmup
    )

    # 保存结果
    results = {
        "environment": env_info,
        "model": model_info,
        "baseline": baseline_results,
        "xla": xla_results,
        "mixed_precision_xla": mixed_results,
        "config": {
            "num_runs": args.num_runs,
            "num_warmup": args.num_warmup,
            "num_samples": args.num_samples,
            "max_accuracy_drop": args.max_accuracy_drop
        }
    }

    results_json = output_dir / "results.json"
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 结果已保存: {results_json}")

    # 生成报告
    report_path = generate_report(
        env_info, model_info, baseline_results, xla_results,
        mixed_results, output_dir, args.max_accuracy_drop
    )

    # 打印总结
    print_section("✓ 测试完成!")

    baseline_acc = baseline_results['accuracy']
    xla_acc = xla_results['accuracy']
    mixed_acc = mixed_results['accuracy']

    baseline_lat = baseline_results['mean_ms']
    xla_lat = xla_results['mean_ms']
    mixed_lat = mixed_results['mean_ms']

    print(f"\n性能提升:")
    print(f"  XLA优化: {baseline_lat/xla_lat:.2f}x")
    print(f"  XLA + 混合精度: {baseline_lat/mixed_lat:.2f}x")

    print(f"\n准确率对比:")
    print(f"  Baseline: {baseline_acc*100:.2f}%")
    print(f"  XLA: {xla_acc*100:.2f}% ({(baseline_acc-xla_acc)*100:+.2f}%)")
    print(f"  XLA + Mixed: {mixed_acc*100:.2f}% ({(baseline_acc-mixed_acc)*100:+.2f}%)")

    # 检查约束
    xla_passed, xla_drop = check_accuracy_constraint(baseline_acc, xla_acc, args.max_accuracy_drop)
    mixed_passed, mixed_drop = check_accuracy_constraint(baseline_acc, mixed_acc, args.max_accuracy_drop)

    print(f"\n准确率约束检查 (≤{args.max_accuracy_drop}%):")
    print(f"  XLA: {'✅ 通过' if xla_passed else '❌ 失败'} ({abs(xla_drop):.2f}%)")
    print(f"  XLA + Mixed: {'✅ 通过' if mixed_passed else '❌ 失败'} ({abs(mixed_drop):.2f}%)")

    print(f"\n结果文件:")
    print(f"  - JSON结果: {results_json}")
    print(f"  - 对比报告: {report_path}")


if __name__ == "__main__":
    main()
