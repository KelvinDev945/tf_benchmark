#!/usr/bin/env python3
"""
TensorFlow多线程推理性能测试

对比不同线程配置对推理性能的影响
测试BERT-Base和MobileNet两个模型
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
import multiprocessing

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def get_cpu_info():
    """获取CPU核心数信息"""
    physical_cores = multiprocessing.cpu_count()

    # 尝试获取物理核心数（Linux）
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            # 统计物理CPU ID数量
            physical_ids = set()
            for line in cpuinfo.split('\n'):
                if line.startswith('physical id'):
                    physical_ids.add(line.split(':')[1].strip())

            # 统计每个物理CPU的核心数
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

def run_benchmark_worker(intra_threads, inter_threads, model_type, num_runs, batch_size):
    """调用worker脚本执行benchmark"""
    script_path = Path(__file__).parent / "benchmark_threading_worker.py"

    cmd = [
        "python3",
        str(script_path),
        str(intra_threads),
        str(inter_threads),
        model_type,
        str(num_runs),
        str(batch_size)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5分钟超时
            check=True
        )

        # 解析JSON输出
        return json.loads(result.stdout.strip().split('\n')[-1])
    except subprocess.CalledProcessError as e:
        print(f"  ❌ 测试失败:")
        print(f"     Error: {e}")
        if e.stderr:
            print(f"     Stderr: {e.stderr[:500]}")
        return None
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON解析失败:")
        print(f"     Error: {e}")
        print(f"     Output: {result.stdout[:500]}")
        return None
    except Exception as e:
        print(f"  ❌ 意外错误: {e}")
        return None

def test_threading_configs(model_type, model_name, cpu_info, num_runs=30, batch_size=1):
    """测试多种线程配置"""
    print_section(f"测试 {model_name} - 不同线程配置")

    logical_cores = cpu_info['logical_cores']
    physical_cores = cpu_info['physical_cores']

    print(f"CPU信息:")
    print(f"  逻辑核心数: {logical_cores}")
    print(f"  物理核心数: {physical_cores}")
    print(f"  超线程: {'启用' if cpu_info['hyperthreading'] else '禁用'}")

    # 测试配置列表
    # (intra_threads, inter_threads, description)
    test_configs = [
        (1, 1, "单线程"),
        (2, 1, "2线程 (intra)"),
        (4, 1, "4线程 (intra)"),
        (8, 1, "8线程 (intra)"),
        (physical_cores, 1, f"{physical_cores}线程 (物理核心)"),
        (physical_cores, 2, f"{physical_cores}线程 + 2 inter"),
        (0, 0, "自动配置"),
    ]

    results = []

    for intra, inter, desc in test_configs:
        print(f"\n测试配置: {desc}")
        print(f"  intra_op_parallelism_threads: {intra if intra > 0 else 'auto'}")
        print(f"  inter_op_parallelism_threads: {inter if inter > 0 else 'auto'}")

        result = run_benchmark_worker(intra, inter, model_type, num_runs, batch_size)

        if result:
            result['description'] = desc
            results.append(result)

            print(f"  ✓ 平均延迟: {result['mean_ms']:.2f} ms")
            print(f"  ✓ 吞吐量: {result['throughput_samples_per_sec']:.2f} samples/sec")
        else:
            print(f"  ⚠️  跳过此配置")

    return results

def find_optimal_config(results):
    """找到最优配置"""
    if not results:
        return None
    # 按吞吐量排序
    sorted_results = sorted(results, key=lambda x: x['throughput_samples_per_sec'], reverse=True)
    return sorted_results[0]

def generate_report(bert_results, mobilenet_results, cpu_info, output_file):
    """生成对比报告"""
    print_section("生成测试报告")

    # 找到最优配置
    bert_optimal = find_optimal_config(bert_results)
    mobilenet_optimal = find_optimal_config(mobilenet_results)

    if not bert_optimal or not mobilenet_optimal:
        print("⚠️ 缺少测试结果，无法生成完整报告")
        return

    # 计算相对单线程的加速比
    bert_baseline = next((r for r in bert_results if r['intra_threads'] == 1), None)
    mobilenet_baseline = next((r for r in mobilenet_results if r['intra_threads'] == 1), None)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# TensorFlow多线程推理性能测试报告\n\n")
        f.write(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 系统信息\n\n")
        f.write(f"- CPU逻辑核心数: {cpu_info['logical_cores']}\n")
        f.write(f"- CPU物理核心数: {cpu_info['physical_cores']}\n")
        f.write(f"- 超线程: {'启用' if cpu_info['hyperthreading'] else '禁用'}\n\n")

        # BERT-Base结果
        f.write("## BERT-Base 线程配置测试\n\n")
        f.write("| 配置 | Intra线程 | Inter线程 | 平均延迟 | P95延迟 | 吞吐量 | vs单线程 |\n")
        f.write("|------|-----------|-----------|----------|---------|--------|----------|\n")

        for result in bert_results:
            if bert_baseline:
                speedup = result['throughput_samples_per_sec'] / bert_baseline['throughput_samples_per_sec']
            else:
                speedup = 1.0

            intra_str = str(result['intra_threads']) if result['intra_threads'] > 0 else 'auto'
            inter_str = str(result['inter_threads']) if result['inter_threads'] > 0 else 'auto'

            f.write(f"| {result['description']} | {intra_str} | {inter_str} | ")
            f.write(f"{result['mean_ms']:.2f} ms | {result['p95_ms']:.2f} ms | ")
            f.write(f"{result['throughput_samples_per_sec']:.2f} samples/s | ")
            f.write(f"{speedup:.2f}x {'🚀' if speedup > 1.5 else ''} |\n")

        f.write(f"\n**最优配置**: {bert_optimal['description']}\n")
        f.write(f"- 吞吐量: {bert_optimal['throughput_samples_per_sec']:.2f} samples/sec\n")
        if bert_baseline:
            f.write(f"- 相对单线程加速: {bert_optimal['throughput_samples_per_sec'] / bert_baseline['throughput_samples_per_sec']:.2f}x\n\n")

        # MobileNet结果
        f.write("## MobileNetV2 线程配置测试\n\n")
        f.write("| 配置 | Intra线程 | Inter线程 | 平均延迟 | P95延迟 | 吞吐量 | vs单线程 |\n")
        f.write("|------|-----------|-----------|----------|---------|--------|----------|\n")

        for result in mobilenet_results:
            if mobilenet_baseline:
                speedup = result['throughput_samples_per_sec'] / mobilenet_baseline['throughput_samples_per_sec']
            else:
                speedup = 1.0

            intra_str = str(result['intra_threads']) if result['intra_threads'] > 0 else 'auto'
            inter_str = str(result['inter_threads']) if result['inter_threads'] > 0 else 'auto'

            f.write(f"| {result['description']} | {intra_str} | {inter_str} | ")
            f.write(f"{result['mean_ms']:.2f} ms | {result['p95_ms']:.2f} ms | ")
            f.write(f"{result['throughput_samples_per_sec']:.2f} samples/s | ")
            f.write(f"{speedup:.2f}x {'🚀' if speedup > 1.5 else ''} |\n")

        f.write(f"\n**最优配置**: {mobilenet_optimal['description']}\n")
        f.write(f"- 吞吐量: {mobilenet_optimal['throughput_samples_per_sec']:.2f} samples/sec\n")
        if mobilenet_baseline:
            f.write(f"- 相对单线程加速: {mobilenet_optimal['throughput_samples_per_sec'] / mobilenet_baseline['throughput_samples_per_sec']:.2f}x\n\n")

        # 总结
        f.write("## 总结\n\n")

        f.write("### 性能提升\n\n")
        if bert_baseline and mobilenet_baseline:
            bert_speedup = bert_optimal['throughput_samples_per_sec'] / bert_baseline['throughput_samples_per_sec']
            mobilenet_speedup = mobilenet_optimal['throughput_samples_per_sec'] / mobilenet_baseline['throughput_samples_per_sec']

            f.write(f"- **BERT-Base**: {bert_speedup:.2f}x 加速 (单线程 → {bert_optimal['description']})\n")
            f.write(f"- **MobileNetV2**: {mobilenet_speedup:.2f}x 加速 (单线程 → {mobilenet_optimal['description']})\n\n")

        f.write("### 推荐配置\n\n")
        f.write(f"**BERT-Base推荐**:\n")
        f.write(f"```python\n")
        f.write(f"tf.config.threading.set_intra_op_parallelism_threads({bert_optimal['intra_threads']})\n")
        f.write(f"tf.config.threading.set_inter_op_parallelism_threads({bert_optimal['inter_threads']})\n")
        f.write(f"```\n\n")

        f.write(f"**MobileNetV2推荐**:\n")
        f.write(f"```python\n")
        f.write(f"tf.config.threading.set_intra_op_parallelism_threads({mobilenet_optimal['intra_threads']})\n")
        f.write(f"tf.config.threading.set_inter_op_parallelism_threads({mobilenet_optimal['inter_threads']})\n")
        f.write(f"```\n\n")

        f.write("### 使用建议\n\n")
        f.write("1. **生产部署**: 根据CPU核心数和并发需求调整线程配置\n")
        f.write("2. **单实例高吞吐**: 使用物理核心数作为intra_threads\n")
        f.write("3. **多实例并发**: 限制每个实例的线程数，避免资源竞争\n")
        f.write("4. **实时推理**: 使用较少线程数减少延迟抖动\n\n")

        f.write("## 参考\n\n")
        f.write("- [TensorFlow线程配置文档](https://www.tensorflow.org/api_docs/python/tf/config/threading)\n")
        f.write("- [CPU优化完整指南](TENSORFLOW_CPU_OPTIMIZATION.md)\n")
        f.write("- [综合Benchmark结果](BENCHMARK_RESULTS.md)\n")

    print(f"✓ 报告已保存: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="TensorFlow多线程推理性能测试")
    parser.add_argument("--output-dir", default="results/threading_benchmark",
                       help="输出目录")
    parser.add_argument("--num-runs", type=int, default=30,
                       help="每个配置的测试迭代次数")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="批次大小")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_section("TensorFlow多线程推理性能测试")
    print(f"测试配置:")
    print(f"  每个配置迭代次数: {args.num_runs}")
    print(f"  批次大小: {args.batch_size}")

    # 获取CPU信息
    cpu_info = get_cpu_info()

    # 测试BERT
    bert_results = test_threading_configs(
        "bert", "BERT-Base",
        cpu_info,
        num_runs=args.num_runs,
        batch_size=args.batch_size
    )

    # 测试MobileNet
    mobilenet_results = test_threading_configs(
        "mobilenet", "MobileNetV2",
        cpu_info,
        num_runs=args.num_runs,
        batch_size=args.batch_size
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
                "batch_size": args.batch_size
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 原始结果已保存: {results_json}")

    # 生成报告
    report_path = output_dir / "threading_benchmark_report.md"
    generate_report(bert_results, mobilenet_results, cpu_info, report_path)

    # 打印总结
    print_section("✓ 测试完成!")

    bert_optimal = find_optimal_config(bert_results)
    mobilenet_optimal = find_optimal_config(mobilenet_results)

    bert_baseline = next((r for r in bert_results if r['intra_threads'] == 1), None)
    mobilenet_baseline = next((r for r in mobilenet_results if r['intra_threads'] == 1), None)

    if bert_optimal and bert_baseline:
        print(f"\nBERT-Base最优配置: {bert_optimal['description']}")
        print(f"  加速比: {bert_optimal['throughput_samples_per_sec'] / bert_baseline['throughput_samples_per_sec']:.2f}x")
        print(f"  吞吐量: {bert_optimal['throughput_samples_per_sec']:.2f} samples/sec")

    if mobilenet_optimal and mobilenet_baseline:
        print(f"\nMobileNetV2最优配置: {mobilenet_optimal['description']}")
        print(f"  加速比: {mobilenet_optimal['throughput_samples_per_sec'] / mobilenet_baseline['throughput_samples_per_sec']:.2f}x")
        print(f"  吞吐量: {mobilenet_optimal['throughput_samples_per_sec']:.2f} samples/sec")

    print(f"\n报告文件: {report_path}")

if __name__ == "__main__":
    main()
