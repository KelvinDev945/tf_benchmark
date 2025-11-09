#!/usr/bin/env python3
"""
对比tf2onnx和Optimum转换BERT模型后的ONNX推理性能

使用相同的BERT模型，分别用两种工具转换，然后对比推理速度
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def method1_optimum_bert():
    """方法1: 使用Optimum转换BERT"""
    print_section("方法1: Optimum转换BERT")

    import onnxruntime as ort  # noqa: F401
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    model_name = "prajjwal1/bert-tiny"  # 使用小模型快速测试

    print(f"1. 使用Optimum加载和转换: {model_name}")
    start_time = time.time()

    # Optimum自动转换为ONNX
    model = ORTModelForSequenceClassification.from_pretrained(model_name, export=True)

    conversion_time = time.time() - start_time

    # 保存到临时目录
    temp_dir = tempfile.mkdtemp()
    model.save_pretrained(temp_dir)

    onnx_path = Path(temp_dir) / "model.onnx"
    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)

    print(f"✓ 转换完成!")
    print(f"  转换时间: {conversion_time:.2f} 秒")
    print(f"  模型大小: {file_size_mb:.2f} MB")

    # 准备测试数据
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_text = "This is a test sentence for benchmarking."
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, max_length=128)

    # 推理性能测试
    print("\n2. 推理性能测试 (Optimum)")
    num_runs = 1000
    num_warmup = 100

    # 预热
    print(f"  预热中 ({num_warmup} 次)...")
    for i in range(num_warmup):
        if (i + 1) % 20 == 0:
            print(f"    预热进度: {i+1}/{num_warmup}")
        _ = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    # 基准测试
    print(f"  开始推理测试 ({num_runs} 次)...")
    latencies = []
    for i in range(num_runs):
        if (i + 1) % 200 == 0:
            print(f"    推理进度: {i+1}/{num_runs}")
        start = time.perf_counter()
        _ = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        latencies.append((time.perf_counter() - start) * 1000)

    mean_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)

    print(f"  平均延迟: {mean_latency:.2f} ms")
    print(f"  P95延迟: {p95_latency:.2f} ms")
    print(f"  吞吐量: {1000/mean_latency:.2f} samples/sec")

    # 清理
    shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "method": "Optimum",
        "conversion_time": conversion_time,
        "model_size_mb": file_size_mb,
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
        "throughput": 1000 / mean_latency,
        "onnx_path": str(onnx_path),
    }


def method2_tf2onnx_bert():
    """方法2: 使用tf2onnx转换BERT"""
    print_section("方法2: tf2onnx转换BERT")

    try:
        import subprocess

        import onnxruntime as ort
        import tensorflow as tf
        import tf2onnx  # noqa: F401
        from transformers import BertTokenizer, TFBertForSequenceClassification

        model_name = "prajjwal1/bert-tiny"

        print(f"1. 加载TensorFlow BERT模型: {model_name}")
        model = TFBertForSequenceClassification.from_pretrained(model_name, from_pt=True)

        # 保存为SavedModel
        temp_dir = tempfile.mkdtemp()
        saved_model_path = os.path.join(temp_dir, "saved_model")

        print("2. 导出为SavedModel...")
        # 使用export而不是save (Keras 3兼容)
        tf.saved_model.save(model, saved_model_path)

        # 使用tf2onnx转换
        onnx_path = os.path.join(temp_dir, "model.onnx")

        print("3. 使用tf2onnx转换...")
        start_time = time.time()

        cmd = [
            "python3",
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            saved_model_path,
            "--output",
            onnx_path,
            "--opset",
            "15",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"✗ tf2onnx转换失败:")
            print(result.stderr[:500])
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None

        conversion_time = time.time() - start_time
        file_size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)

        print(f"✓ 转换完成!")
        print(f"  转换时间: {conversion_time:.2f} 秒")
        print(f"  模型大小: {file_size_mb:.2f} MB")

        # 推理性能测试
        print("\n4. 推理性能测试 (tf2onnx)")

        tokenizer = BertTokenizer.from_pretrained(model_name)
        test_text = "This is a test sentence for benchmarking."
        inputs = tokenizer(test_text, return_tensors="np", padding=True, max_length=128)

        # 创建ONNX Runtime会话
        sess = ort.InferenceSession(onnx_path)

        # 获取输入名称
        input_names = [inp.name for inp in sess.get_inputs()]
        print(f"  ONNX模型需要的输入: {input_names}")

        # 准备输入字典 - 智能匹配输入名称
        onnx_inputs = {}

        # 为每个ONNX输入找到对应的tokenizer输出
        # TensorFlow模型通常使用int32，而PyTorch使用int64
        for onnx_input_name in input_names:
            lower_name = onnx_input_name.lower()

            # 匹配input_ids
            if "input" in lower_name and "ids" in lower_name:
                onnx_inputs[onnx_input_name] = inputs["input_ids"].astype(np.int32)
            # 匹配attention_mask
            elif "attention" in lower_name or "mask" in lower_name:
                onnx_inputs[onnx_input_name] = inputs["attention_mask"].astype(np.int32)
            # 匹配token_type_ids
            elif "token_type" in lower_name or "segment" in lower_name or "type" in lower_name:
                if "token_type_ids" in inputs:
                    onnx_inputs[onnx_input_name] = inputs["token_type_ids"].astype(np.int32)
                else:
                    # 如果tokenizer没有提供，创建全0数组
                    onnx_inputs[onnx_input_name] = np.zeros_like(
                        inputs["input_ids"], dtype=np.int32
                    )

        print(f"  提供的输入: {list(onnx_inputs.keys())}")

        num_runs = 1000
        num_warmup = 100

        # 预热
        print(f"\n  预热中 ({num_warmup} 次)...")
        for i in range(num_warmup):
            if (i + 1) % 20 == 0:
                print(f"    预热进度: {i+1}/{num_warmup}")
            _ = sess.run(None, onnx_inputs)

        # 基准测试
        print(f"  开始推理测试 ({num_runs} 次)...")
        latencies = []
        for i in range(num_runs):
            if (i + 1) % 200 == 0:
                print(f"    推理进度: {i+1}/{num_runs}")
            start = time.perf_counter()
            _ = sess.run(None, onnx_inputs)
            latencies.append((time.perf_counter() - start) * 1000)

        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        print(f"  平均延迟: {mean_latency:.2f} ms")
        print(f"  P95延迟: {p95_latency:.2f} ms")
        print(f"  吞吐量: {1000/mean_latency:.2f} samples/sec")

        # 清理
        shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            "method": "tf2onnx",
            "conversion_time": conversion_time,
            "model_size_mb": file_size_mb,
            "mean_latency_ms": mean_latency,
            "p95_latency_ms": p95_latency,
            "throughput": 1000 / mean_latency,
            "onnx_path": onnx_path,
        }

    except Exception as e:
        print(f"✗ tf2onnx方法失败: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    print("=" * 80)
    print("BERT模型: tf2onnx vs Optimum ONNX转换推理性能对比")
    print("=" * 80)
    print("模型: prajjwal1/bert-tiny (2层BERT)")
    print("任务: 序列分类")

    # 测试方法1: Optimum
    try:
        optimum_result = method1_optimum_bert()
    except Exception as e:
        print(f"\n✗ Optimum测试失败: {e}")
        import traceback

        traceback.print_exc()
        optimum_result = None

    # 测试方法2: tf2onnx
    try:
        tf2onnx_result = method2_tf2onnx_bert()
    except Exception as e:
        print(f"\n✗ tf2onnx测试失败: {e}")
        import traceback

        traceback.print_exc()
        tf2onnx_result = None

    # 对比总结
    print_section("性能对比总结")

    if optimum_result and tf2onnx_result:
        print("\n✅ 两种方法都成功!\n")

        print("转换性能对比:")
        print(f"  Optimum转换时间:  {optimum_result['conversion_time']:6.2f} 秒")
        print(f"  tf2onnx转换时间:  {tf2onnx_result['conversion_time']:6.2f} 秒")
        print(
            f"  差异: {abs(optimum_result['conversion_time'] - tf2onnx_result['conversion_time']):.2f} 秒"  # noqa: E501
        )

        print(f"\n模型大小对比:")
        print(f"  Optimum模型大小:  {optimum_result['model_size_mb']:6.2f} MB")
        print(f"  tf2onnx模型大小:  {tf2onnx_result['model_size_mb']:6.2f} MB")

        print(f"\n推理性能对比 (重要!):")
        print(f"  {'指标':<20} {'Optimum':>12} {'tf2onnx':>12} {'差异':>12}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
        print(
            f"  {'平均延迟 (ms)':<20} {optimum_result['mean_latency_ms']:>12.2f} {tf2onnx_result['mean_latency_ms']:>12.2f} {abs(optimum_result['mean_latency_ms'] - tf2onnx_result['mean_latency_ms']):>12.2f}"  # noqa: E501
        )
        print(
            f"  {'P95延迟 (ms)':<20} {optimum_result['p95_latency_ms']:>12.2f} {tf2onnx_result['p95_latency_ms']:>12.2f} {abs(optimum_result['p95_latency_ms'] - tf2onnx_result['p95_latency_ms']):>12.2f}"  # noqa: E501
        )
        print(
            f"  {'吞吐量 (samples/s)':<20} {optimum_result['throughput']:>12.2f} {tf2onnx_result['throughput']:>12.2f} {abs(optimum_result['throughput'] - tf2onnx_result['throughput']):>12.2f}"  # noqa: E501
        )

        # 计算性能差异百分比
        latency_diff_pct = (
            abs(optimum_result["mean_latency_ms"] - tf2onnx_result["mean_latency_ms"])
            / min(optimum_result["mean_latency_ms"], tf2onnx_result["mean_latency_ms"])
            * 100
        )

        print(f"\n结论:")
        if latency_diff_pct < 5:
            print(f"  📊 推理性能基本相同 (差异 < 5%: {latency_diff_pct:.1f}%)")
            print("  → 两种转换方法产生的ONNX模型推理速度没有显著差异")
            print("  → 选择转换方法应基于工具兼容性和易用性，而非推理性能")
        elif latency_diff_pct < 10:
            print(f"  📊 推理性能略有差异 (5-10%: {latency_diff_pct:.1f}%)")
            if optimum_result["mean_latency_ms"] < tf2onnx_result["mean_latency_ms"]:
                print("  → Optimum转换的模型略快")
            else:
                print("  → tf2onnx转换的模型略快")
        else:
            print(f"  📊 推理性能有明显差异 (> 10%: {latency_diff_pct:.1f}%)")
            if optimum_result["mean_latency_ms"] < tf2onnx_result["mean_latency_ms"]:
                faster_pct = (
                    tf2onnx_result["mean_latency_ms"] / optimum_result["mean_latency_ms"] - 1
                ) * 100
                print(f"  → Optimum转换的模型快 {faster_pct:.1f}%")
            else:
                faster_pct = (
                    optimum_result["mean_latency_ms"] / tf2onnx_result["mean_latency_ms"] - 1
                ) * 100
                print(f"  → tf2onnx转换的模型快 {faster_pct:.1f}%")

        print("\n推荐:")
        print("  • 对于HuggingFace模型 → 使用Optimum (更简单，无版本冲突)")
        print("  • 对于TensorFlow自定义模型 → 使用tf2onnx (兼容性问题待解决)")

    elif optimum_result:
        print("\n✓ 仅Optimum成功")
        print(f"  推理延迟: {optimum_result['mean_latency_ms']:.2f} ms")
        print("  → tf2onnx在当前环境遇到兼容性问题")

    elif tf2onnx_result:
        print("\n✓ 仅tf2onnx成功")
        print(f"  推理延迟: {tf2onnx_result['mean_latency_ms']:.2f} ms")

    else:
        print("\n❌ 两种方法都失败")
        print("  可能原因: 依赖版本兼容性问题")


if __name__ == "__main__":
    main()
