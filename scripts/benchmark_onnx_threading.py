#!/usr/bin/env python3
"""
ONNX Runtime多线程配置性能测试

测试ONNX Runtime在不同线程配置下的性能
对比TensorFlow多线程优化后 vs ONNX Runtime多线程优化
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import onnxruntime as ort
import multiprocessing

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def get_cpu_info():
    """获取CPU核心数信息"""
    physical_cores = multiprocessing.cpu_count()
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            physical_ids = set()
            for line in cpuinfo.split('\n'):
                if line.startswith('physical id'):
                    physical_ids.add(line.split(':')[1].strip())
            cores_per_cpu = 0
            for line in cpuinfo.split('\n'):
                if line.startswith('cpu cores'):
                    cores_per_cpu = int(line.split(':')[1].strip())
                    break
            actual_physical_cores = len(physical_ids) * cores_per_cpu if physical_ids else physical_cores
    except:
        actual_physical_cores = physical_cores

    return {
        'logical_cores': physical_cores,
        'physical_cores': actual_physical_cores,
        'hyperthreading': physical_cores > actual_physical_cores
    }

def create_bert_base_model():
    """创建BERT-Base模型"""
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12
    seq_length = 128
    vocab_size = 10000
    intermediate_size = hidden_size * 4

    input_ids = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32, name='input_ids')
    embeddings = tf.keras.layers.Embedding(vocab_size, hidden_size, name='embedding')(input_ids)
    position_embeddings = tf.keras.layers.Embedding(seq_length, hidden_size, name='position_embedding')(tf.range(seq_length))

    x = embeddings + position_embeddings
    x = tf.keras.layers.LayerNormalization(epsilon=1e-12)(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    for i in range(num_hidden_layers):
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=hidden_size // num_attention_heads,
            name=f'attention_{i}'
        )(x, x)
        attention_output = tf.keras.layers.Dropout(0.1)(attention_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-12)(x + attention_output)

        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(intermediate_size, activation='relu'),
            tf.keras.layers.Dense(hidden_size)
        ], name=f'ffn_{i}')
        ffn_output = ffn(x)
        ffn_output = tf.keras.layers.Dropout(0.1)(ffn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-12)(x + ffn_output)

    pooled_output = tf.keras.layers.Lambda(lambda x: x[:, 0])(x)
    pooled_output = tf.keras.layers.Dense(hidden_size, activation='tanh', name='pooler')(pooled_output)
    output = tf.keras.layers.Dense(2, activation='softmax', name='classifier')(pooled_output)

    model = tf.keras.Model(inputs=input_ids, outputs=output, name='bert_base_model')
    return model

def create_mobilenet_model():
    """创建MobileNetV2模型"""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1000, activation='softmax')
    ], name='mobilenet_v2')
    return model

def convert_to_onnx(model, output_path, model_type):
    """转换TensorFlow模型到ONNX"""
    import subprocess
    import tempfile

    # 保存为SavedModel
    temp_dir = tempfile.mkdtemp()
    saved_model_path = os.path.join(temp_dir, "saved_model")
    model.export(saved_model_path)

    # 转换为ONNX
    cmd = [
        "python3", "-m", "tf2onnx.convert",
        "--saved-model", saved_model_path,
        "--output", output_path,
        "--opset", "13"
    ]

    subprocess.run(cmd, check=True, capture_output=True)

    # 清理临时文件
    import shutil
    shutil.rmtree(temp_dir)

def benchmark_onnx_with_threads(onnx_path, X_test, intra_threads, inter_threads,
                                num_runs=30, batch_size=1):
    """
    使用指定线程配置测试ONNX Runtime性能

    Args:
        onnx_path: ONNX模型路径
        X_test: 测试数据
        intra_threads: intra_op线程数
        inter_threads: inter_op线程数
        num_runs: 测试迭代次数
        batch_size: 批次大小
    """
    # 配置ONNX Runtime Session Options
    sess_options = ort.SessionOptions()

    # 设置线程配置
    if intra_threads > 0:
        sess_options.intra_op_num_threads = intra_threads
    if inter_threads > 0:
        sess_options.inter_op_num_threads = inter_threads

    # 其他优化选项
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # 创建推理session
    session = ort.InferenceSession(onnx_path, sess_options)

    input_name = session.get_inputs()[0].name

    # 热身
    num_warmup = 5
    for _ in range(num_warmup):
        _ = session.run(None, {input_name: X_test[:batch_size]})

    # 性能测试
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: X_test[:batch_size]})
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

    # 统计
    latencies = np.array(latencies)
    results = {
        "intra_threads": intra_threads if intra_threads > 0 else "default",
        "inter_threads": inter_threads if inter_threads > 0 else "default",
        "batch_size": batch_size,
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_samples_per_sec": float(batch_size * 1000.0 / np.mean(latencies))
    }

    return results

def test_onnx_threading_configs(onnx_path, X_test, model_name, cpu_info, num_runs=30):
    """测试多种ONNX Runtime线程配置"""
    print_section(f"测试 {model_name} ONNX - 不同线程配置")

    logical_cores = cpu_info['logical_cores']
    physical_cores = cpu_info['physical_cores']

    print(f"CPU信息:")
    print(f"  逻辑核心数: {logical_cores}")
    print(f"  物理核心数: {physical_cores}")

    # 测试配置列表
    test_configs = [
        (1, 1, "单线程"),
        (2, 1, "2线程 (intra)"),
        (4, 1, "4线程 (intra)"),
        (8, 1, "8线程 (intra)"),
        (physical_cores, 1, f"{physical_cores}线程 (物理核心)"),
        (physical_cores, 2, f"{physical_cores}线程 + 2 inter"),
        (0, 0, "默认配置"),
    ]

    results = []

    for intra, inter, desc in test_configs:
        print(f"\n测试配置: {desc}")
        print(f"  intra_op_num_threads: {intra if intra > 0 else 'default'}")
        print(f"  inter_op_num_threads: {inter if inter > 0 else 'default'}")

        result = benchmark_onnx_with_threads(
            onnx_path, X_test, intra, inter, num_runs=num_runs
        )
        result['description'] = desc
        results.append(result)

        print(f"  ✓ 平均延迟: {result['mean_ms']:.2f} ms")
        print(f"  ✓ 吞吐量: {result['throughput_samples_per_sec']:.2f} samples/sec")

    return results

def generate_report(bert_results, mobilenet_results, cpu_info, output_file):
    """生成对比报告"""
    print_section("生成测试报告")

    # 找到最优配置
    bert_optimal = min(bert_results, key=lambda x: x['mean_ms'])
    mobilenet_optimal = min(mobilenet_results, key=lambda x: x['mean_ms'])

    bert_baseline = next(r for r in bert_results if r['intra_threads'] == 1)
    mobilenet_baseline = next(r for r in mobilenet_results if r['intra_threads'] == 1)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# ONNX Runtime多线程配置性能测试报告\n\n")
        f.write(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 系统信息\n\n")
        f.write(f"- ONNX Runtime版本: {ort.__version__}\n")
        f.write(f"- CPU逻辑核心数: {cpu_info['logical_cores']}\n")
        f.write(f"- CPU物理核心数: {cpu_info['physical_cores']}\n")
        f.write(f"- 超线程: {'启用' if cpu_info['hyperthreading'] else '禁用'}\n\n")

        # BERT-Base结果
        f.write("## BERT-Base ONNX 线程配置测试\n\n")
        f.write("| 配置 | Intra线程 | Inter线程 | 平均延迟 | P95延迟 | 吞吐量 | vs单线程 |\n")
        f.write("|------|-----------|-----------|----------|---------|--------|----------|\n")

        for result in bert_results:
            speedup = bert_baseline['mean_ms'] / result['mean_ms']
            intra_str = str(result['intra_threads'])
            inter_str = str(result['inter_threads'])

            f.write(f"| {result['description']} | {intra_str} | {inter_str} | ")
            f.write(f"{result['mean_ms']:.2f} ms | {result['p95_ms']:.2f} ms | ")
            f.write(f"{result['throughput_samples_per_sec']:.2f} samples/s | ")
            f.write(f"{speedup:.2f}x {'🚀' if speedup > 1.2 else ''} |\n")

        f.write(f"\n**最优配置**: {bert_optimal['description']}\n")
        f.write(f"- 延迟: {bert_optimal['mean_ms']:.2f} ms\n")
        f.write(f"- 相对单线程加速: {bert_baseline['mean_ms'] / bert_optimal['mean_ms']:.2f}x\n\n")

        # MobileNet结果
        f.write("## MobileNetV2 ONNX 线程配置测试\n\n")
        f.write("| 配置 | Intra线程 | Inter线程 | 平均延迟 | P95延迟 | 吞吐量 | vs单线程 |\n")
        f.write("|------|-----------|-----------|----------|---------|--------|----------|\n")

        for result in mobilenet_results:
            speedup = mobilenet_baseline['mean_ms'] / result['mean_ms']
            intra_str = str(result['intra_threads'])
            inter_str = str(result['inter_threads'])

            f.write(f"| {result['description']} | {intra_str} | {inter_str} | ")
            f.write(f"{result['mean_ms']:.2f} ms | {result['p95_ms']:.2f} ms | ")
            f.write(f"{result['throughput_samples_per_sec']:.2f} samples/s | ")
            f.write(f"{speedup:.2f}x {'🚀' if speedup > 1.2 else ''} |\n")

        f.write(f"\n**最优配置**: {mobilenet_optimal['description']}\n")
        f.write(f"- 延迟: {mobilenet_optimal['mean_ms']:.2f} ms\n")
        f.write(f"- 相对单线程加速: {mobilenet_baseline['mean_ms'] / mobilenet_optimal['mean_ms']:.2f}x\n\n")

        # 总结
        f.write("## 总结\n\n")

        bert_speedup = bert_baseline['mean_ms'] / bert_optimal['mean_ms']
        mobilenet_speedup = mobilenet_baseline['mean_ms'] / mobilenet_optimal['mean_ms']

        f.write("### 多线程性能提升\n\n")
        f.write(f"- **BERT-Base**: {bert_speedup:.2f}x 加速 (单线程 → {bert_optimal['description']})\n")
        f.write(f"- **MobileNetV2**: {mobilenet_speedup:.2f}x 加速 (单线程 → {mobilenet_optimal['description']})\n\n")

        f.write("### 推荐配置\n\n")
        f.write("**BERT-Base ONNX推荐**:\n```python\n")
        f.write("sess_options = ort.SessionOptions()\n")
        f.write(f"sess_options.intra_op_num_threads = {bert_optimal['intra_threads']}\n")
        f.write(f"sess_options.inter_op_num_threads = {bert_optimal['inter_threads']}\n")
        f.write("```\n\n")

        f.write("**MobileNetV2 ONNX推荐**:\n```python\n")
        f.write("sess_options = ort.SessionOptions()\n")
        f.write(f"sess_options.intra_op_num_threads = {mobilenet_optimal['intra_threads']}\n")
        f.write(f"sess_options.inter_op_num_threads = {mobilenet_optimal['inter_threads']}\n")
        f.write("```\n\n")

        f.write("### 关键发现\n\n")
        if bert_speedup > 1.5:
            f.write("- ✅ BERT-Base ONNX受益于多线程配置\n")
        else:
            f.write("- ⚠️ BERT-Base ONNX默认配置已接近最优\n")

        if mobilenet_speedup > 1.5:
            f.write("- ✅ MobileNetV2 ONNX受益于多线程配置\n")
        else:
            f.write("- ⚠️ MobileNetV2 ONNX默认配置已接近最优\n")

        f.write("\n### 与TensorFlow对比\n\n")
        f.write("参考TensorFlow多线程测试结果：\n")
        f.write("- TensorFlow BERT (auto): ~161ms → ONNX最优: 需测试验证\n")
        f.write("- TensorFlow MobileNet (单线程): ~82ms → ONNX最优: 需测试验证\n\n")

        f.write("## 参考\n\n")
        f.write("- [ONNX Runtime性能调优](https://onnxruntime.ai/docs/performance/tune-performance.html)\n")
        f.write("- [TensorFlow多线程测试结果](results/threading_benchmark/threading_benchmark_report.md)\n")

    print(f"✓ 报告已保存: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="ONNX Runtime多线程配置性能测试")
    parser.add_argument("--output-dir", default="results/onnx_threading",
                       help="输出目录")
    parser.add_argument("--num-runs", type=int, default=30,
                       help="每个配置的测试迭代次数")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_section("ONNX Runtime多线程配置性能测试")
    print(f"ONNX Runtime版本: {ort.__version__}")

    # 获取CPU信息
    cpu_info = get_cpu_info()

    # 创建和转换BERT模型
    print_section("准备BERT-Base模型")
    bert_model = create_bert_base_model()
    bert_onnx_path = output_dir / "bert_base.onnx"

    if not bert_onnx_path.exists():
        print("转换BERT模型到ONNX...")
        convert_to_onnx(bert_model, str(bert_onnx_path), "bert")
    print(f"✓ BERT ONNX模型: {bert_onnx_path}")

    # 准备测试数据
    bert_X = np.random.randint(0, 10000, size=(200, 128), dtype=np.int32)

    # 创建和转换MobileNet模型
    print_section("准备MobileNetV2模型")
    mobilenet_model = create_mobilenet_model()
    mobilenet_onnx_path = output_dir / "mobilenet_v2.onnx"

    if not mobilenet_onnx_path.exists():
        print("转换MobileNet模型到ONNX...")
        convert_to_onnx(mobilenet_model, str(mobilenet_onnx_path), "mobilenet")
    print(f"✓ MobileNet ONNX模型: {mobilenet_onnx_path}")

    # 准备测试数据
    mobilenet_X = np.random.randn(200, 224, 224, 3).astype(np.float32)

    # 测试BERT ONNX
    bert_results = test_onnx_threading_configs(
        str(bert_onnx_path), bert_X, "BERT-Base",
        cpu_info, num_runs=args.num_runs
    )

    # 测试MobileNet ONNX
    mobilenet_results = test_onnx_threading_configs(
        str(mobilenet_onnx_path), mobilenet_X, "MobileNetV2",
        cpu_info, num_runs=args.num_runs
    )

    # 保存原始结果
    results_json = output_dir / "results.json"
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump({
            "cpu_info": cpu_info,
            "bert_results": bert_results,
            "mobilenet_results": mobilenet_results,
            "config": {
                "num_runs": args.num_runs,
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 原始结果已保存: {results_json}")

    # 生成报告
    report_path = output_dir / "onnx_threading_report.md"
    generate_report(bert_results, mobilenet_results, cpu_info, report_path)

    # 打印总结
    print_section("✓ 测试完成!")

    bert_optimal = min(bert_results, key=lambda x: x['mean_ms'])
    mobilenet_optimal = min(mobilenet_results, key=lambda x: x['mean_ms'])

    bert_baseline = next(r for r in bert_results if r['intra_threads'] == 1)
    mobilenet_baseline = next(r for r in mobilenet_results if r['intra_threads'] == 1)

    print(f"\nBERT-Base ONNX最优配置: {bert_optimal['description']}")
    print(f"  延迟: {bert_optimal['mean_ms']:.2f} ms")
    print(f"  加速比: {bert_baseline['mean_ms'] / bert_optimal['mean_ms']:.2f}x")

    print(f"\nMobileNetV2 ONNX最优配置: {mobilenet_optimal['description']}")
    print(f"  延迟: {mobilenet_optimal['mean_ms']:.2f} ms")
    print(f"  加速比: {mobilenet_baseline['mean_ms'] / mobilenet_optimal['mean_ms']:.2f}x")

    print(f"\n报告文件: {report_path}")

if __name__ == "__main__":
    main()
