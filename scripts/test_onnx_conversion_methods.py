#!/usr/bin/env python3
"""
测试两种ONNX转换方法
方法1: TensorFlow 2.15 + tf2onnx
方法2: 当前TensorFlow版本 + HuggingFace Optimum

解决 TODO.md Issue #3: ONNX Runtime NumPy 兼容性问题
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np

print("=" * 70)
print("ONNX转换方法对比测试")
print("=" * 70)

# 检查TensorFlow版本
import tensorflow as tf
print(f"\n当前 TensorFlow 版本: {tf.__version__}\n")

def create_test_model():
    """创建一个简单的测试模型"""
    print("创建测试模型...")
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
    print(f"✓ 模型创建完成，参数总数: {model.count_params():,}")
    return model


def save_model_for_conversion(model, save_dir):
    """保存模型为不同格式供转换使用"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # SavedModel格式（用于tf2onnx）
    saved_model_path = save_dir / "saved_model"
    model.save(saved_model_path)
    print(f"✓ SavedModel 已保存: {saved_model_path}")

    # H5格式（备用）
    h5_path = save_dir / "model.h5"
    model.save(h5_path)
    print(f"✓ H5 模型已保存: {h5_path}")

    return str(saved_model_path), str(h5_path)


def test_method1_tf2onnx(saved_model_path, output_path):
    """
    方法1: 使用 tf2onnx 转换
    注意: 此方法需要 tf2onnx 包
    """
    print("\n" + "=" * 70)
    print("方法1: TensorFlow + tf2onnx")
    print("=" * 70)

    try:
        import tf2onnx
        print(f"✓ tf2onnx 版本: {tf2onnx.__version__}")
    except ImportError:
        print("✗ tf2onnx 未安装")
        print("\n安装命令:")
        print("  pip install tf2onnx")
        return None

    try:
        # 使用tf2onnx转换
        print(f"\n转换模型: {saved_model_path} -> {output_path}")
        start_time = time.time()

        # 方法1a: 使用命令行工具
        cmd = [
            "python3", "-m", "tf2onnx.convert",
            "--saved-model", saved_model_path,
            "--output", output_path,
            "--opset", "13"
        ]

        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        conversion_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ 转换成功 (耗时: {conversion_time:.2f}s)")
            print(f"✓ ONNX模型已保存: {output_path}")

            # 检查文件大小
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  模型大小: {file_size:.2f} MB")

            return {
                "method": "tf2onnx",
                "tensorflow_version": tf.__version__,
                "tf2onnx_version": tf2onnx.__version__,
                "success": True,
                "conversion_time": conversion_time,
                "model_size_mb": file_size,
                "output_path": output_path
            }
        else:
            print(f"✗ 转换失败")
            print(f"错误输出:\n{result.stderr}")
            return {
                "method": "tf2onnx",
                "success": False,
                "error": result.stderr
            }

    except Exception as e:
        print(f"✗ 转换过程出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            "method": "tf2onnx",
            "success": False,
            "error": str(e)
        }


def test_method2_optimum(h5_path, output_path):
    """
    方法2: 使用 HuggingFace Optimum 转换
    注意: Optimum 主要针对 Transformers 模型，对于自定义Keras模型可能不适用
    """
    print("\n" + "=" * 70)
    print("方法2: HuggingFace Optimum")
    print("=" * 70)

    try:
        from optimum.onnxruntime import ORTModelForCustomTasks
        print("✓ Optimum 已安装")
    except ImportError:
        print("✗ Optimum 未安装")
        print("\n安装命令:")
        print("  pip install optimum[onnxruntime]")
        return None

    print("\n⚠️  注意: Optimum 主要用于 HuggingFace Transformers 模型")
    print("对于自定义 Keras 模型，建议使用 tf2onnx 方法")

    # Optimum 不直接支持自定义Keras模型
    # 需要使用 onnx 和 tf2onnx 作为后端
    print("\n✗ Optimum 不适用于自定义 Keras 模型")
    print("Optimum 设计用于转换 HuggingFace Transformers 模型")

    return {
        "method": "optimum",
        "success": False,
        "error": "Optimum不支持自定义Keras模型，仅支持HuggingFace Transformers模型"
    }


def test_onnx_inference(onnx_path, test_data):
    """测试ONNX模型推理性能"""
    print(f"\n测试 ONNX 模型推理: {onnx_path}")

    try:
        import onnxruntime as ort
        print(f"✓ ONNXRuntime 版本: {ort.__version__}")
    except ImportError:
        print("✗ ONNXRuntime 未安装")
        return None

    try:
        # 创建推理会话
        sess = ort.InferenceSession(onnx_path)

        # 获取输入输出名称
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        print(f"  输入名称: {input_name}")
        print(f"  输出名称: {output_name}")

        # 热身
        print("\n热身运行...")
        for _ in range(5):
            _ = sess.run([output_name], {input_name: test_data[:1]})

        # 性能测试
        print("性能测试...")
        num_runs = 100
        latencies = []

        for i in range(num_runs):
            start = time.perf_counter()
            result = sess.run([output_name], {input_name: test_data[:1]})
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)

            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{num_runs}")

        # 计算统计信息
        latencies = np.array(latencies)
        stats = {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "std_ms": float(np.std(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "throughput_samples_per_sec": 1000.0 / np.mean(latencies)
        }

        print("\n✓ ONNX 推理测试完成")
        print(f"  平均延迟: {stats['mean_ms']:.2f} ms")
        print(f"  P95延迟: {stats['p95_ms']:.2f} ms")
        print(f"  吞吐量: {stats['throughput_samples_per_sec']:.2f} samples/sec")

        return stats

    except Exception as e:
        print(f"✗ ONNX 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_report(results, output_dir):
    """生成测试报告"""
    report_path = Path(output_dir) / "onnx_conversion_comparison.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# ONNX 转换方法对比测试报告\n\n")
        f.write(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**TensorFlow 版本**: {tf.__version__}\n\n")

        f.write("## 转换方法对比\n\n")
        f.write("| 方法 | 状态 | 转换时间 | 模型大小 |\n")
        f.write("|------|------|----------|----------|\n")

        for result in results:
            if result and result.get('success'):
                status = "✅ 成功"
                conv_time = f"{result.get('conversion_time', 0):.2f}s"
                size = f"{result.get('model_size_mb', 0):.2f} MB"
            else:
                status = "❌ 失败"
                conv_time = "N/A"
                size = "N/A"

            method = result.get('method', 'unknown') if result else 'unknown'
            f.write(f"| {method} | {status} | {conv_time} | {size} |\n")

        f.write("\n## 推理性能对比\n\n")

        has_perf_data = False
        for result in results:
            if result and result.get('success') and result.get('performance'):
                has_perf_data = True
                break

        if has_perf_data:
            f.write("| 方法 | 平均延迟 | P95延迟 | 吞吐量 |\n")
            f.write("|------|----------|---------|--------|\n")

            for result in results:
                if result and result.get('success') and result.get('performance'):
                    perf = result['performance']
                    f.write(f"| {result['method']} | ")
                    f.write(f"{perf['mean_ms']:.2f} ms | ")
                    f.write(f"{perf['p95_ms']:.2f} ms | ")
                    f.write(f"{perf['throughput_samples_per_sec']:.2f} samples/s |\n")
        else:
            f.write("*没有性能测试数据*\n")

        f.write("\n## 详细结果\n\n")

        for i, result in enumerate(results, 1):
            if result:
                f.write(f"### 方法 {i}: {result.get('method', 'unknown')}\n\n")

                if result.get('success'):
                    f.write("**状态**: ✅ 成功\n\n")

                    if 'tensorflow_version' in result:
                        f.write(f"- TensorFlow 版本: {result['tensorflow_version']}\n")
                    if 'tf2onnx_version' in result:
                        f.write(f"- tf2onnx 版本: {result['tf2onnx_version']}\n")
                    if 'conversion_time' in result:
                        f.write(f"- 转换时间: {result['conversion_time']:.2f}s\n")
                    if 'model_size_mb' in result:
                        f.write(f"- 模型大小: {result['model_size_mb']:.2f} MB\n")
                    if 'output_path' in result:
                        f.write(f"- 输出路径: `{result['output_path']}`\n")

                    if result.get('performance'):
                        f.write("\n**推理性能**:\n")
                        perf = result['performance']
                        f.write(f"- 平均延迟: {perf['mean_ms']:.2f} ms\n")
                        f.write(f"- 中位延迟: {perf['median_ms']:.2f} ms\n")
                        f.write(f"- P95延迟: {perf['p95_ms']:.2f} ms\n")
                        f.write(f"- P99延迟: {perf['p99_ms']:.2f} ms\n")
                        f.write(f"- 吞吐量: {perf['throughput_samples_per_sec']:.2f} samples/s\n")
                else:
                    f.write("**状态**: ❌ 失败\n\n")
                    if 'error' in result:
                        f.write(f"**错误**: {result['error']}\n")

                f.write("\n")

        f.write("## 建议\n\n")

        # 找出成功的方法
        successful_methods = [r for r in results if r and r.get('success')]

        if successful_methods:
            f.write("### 推荐的转换方法\n\n")

            # 如果有性能数据，推荐最快的
            methods_with_perf = [r for r in successful_methods if r.get('performance')]
            if methods_with_perf:
                best = min(methods_with_perf, key=lambda x: x['performance']['mean_ms'])
                f.write(f"✅ **推荐使用 {best['method']}**\n\n")
                f.write(f"- 转换稳定，性能最优\n")
                f.write(f"- 平均延迟: {best['performance']['mean_ms']:.2f} ms\n")
            else:
                f.write(f"✅ **推荐使用 {successful_methods[0]['method']}**\n\n")
                f.write(f"- 转换成功\n")
        else:
            f.write("⚠️ 所有测试方法均未成功\n\n")
            f.write("建议:\n")
            f.write("1. 检查依赖包版本\n")
            f.write("2. 查看详细错误信息\n")
            f.write("3. 考虑使用 Docker 环境隔离测试\n")

    print(f"\n✓ 报告已保存: {report_path}")
    return str(report_path)


def main():
    parser = argparse.ArgumentParser(description="测试ONNX转换方法")
    parser.add_argument("--output-dir", default="results/onnx_conversion_test",
                       help="输出目录")
    parser.add_argument("--test-inference", action="store_true", default=True,
                       help="测试ONNX推理性能")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建测试模型
    model = create_test_model()

    # 保存模型
    print("\n保存模型...")
    saved_model_path, h5_path = save_model_for_conversion(model, output_dir / "models")

    # 创建测试数据
    print("\n创建测试数据...")
    test_data = np.random.randn(10, 28, 28, 1).astype(np.float32)
    print(f"✓ 测试数据形状: {test_data.shape}")

    results = []

    # 测试方法1: tf2onnx
    onnx_path_1 = str(output_dir / "model_tf2onnx.onnx")
    result1 = test_method1_tf2onnx(saved_model_path, onnx_path_1)

    if result1 and result1.get('success') and args.test_inference:
        print("\n" + "-" * 70)
        print("测试方法1的ONNX模型推理性能")
        print("-" * 70)
        perf1 = test_onnx_inference(onnx_path_1, test_data)
        if perf1:
            result1['performance'] = perf1

    if result1:
        results.append(result1)

    # 测试方法2: Optimum
    onnx_path_2 = str(output_dir / "model_optimum.onnx")
    result2 = test_method2_optimum(h5_path, onnx_path_2)

    if result2 and result2.get('success') and args.test_inference:
        print("\n" + "-" * 70)
        print("测试方法2的ONNX模型推理性能")
        print("-" * 70)
        perf2 = test_onnx_inference(onnx_path_2, test_data)
        if perf2:
            result2['performance'] = perf2

    if result2:
        results.append(result2)

    # 保存结果JSON
    results_json = output_dir / "results.json"
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 结果已保存: {results_json}")

    # 生成报告
    report_path = generate_report(results, output_dir)

    print("\n" + "=" * 70)
    print("✓ 测试完成!")
    print("=" * 70)
    print(f"\n结果文件:")
    print(f"  - JSON结果: {results_json}")
    print(f"  - 测试报告: {report_path}")

    # 显示总结
    print("\n总结:")
    successful = sum(1 for r in results if r and r.get('success'))
    print(f"  成功的方法: {successful}/{len(results)}")


if __name__ == "__main__":
    main()
