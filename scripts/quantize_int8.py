#!/usr/bin/env python3
"""
TFLite INT8 量化工具

将TensorFlow/Keras模型转换为TFLite INT8量化模型
解决TODO.md Issue #2: TFLite INT8量化转换错误

Usage:
    python3 scripts/quantize_int8.py --model path/to/model --output path/to/output
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*70)
print("TFLite INT8 量化工具")
print("="*70)
print(f"TensorFlow 版本: {tf.__version__}")
print()


def create_representative_dataset(input_shape, num_samples=100):
    """
    创建代表性数据集用于量化校准

    Args:
        input_shape: 输入形状 (不包含batch维度)
        num_samples: 样本数量

    Returns:
        生成器函数
    """
    print(f"\n创建代表性数据集...")
    print(f"  输入形状: {input_shape}")
    print(f"  样本数: {num_samples}")

    def representative_dataset_gen():
        """代表性数据集生成器"""
        for _ in range(num_samples):
            # 生成随机数据
            # 注意：这里使用随机数据，实际应用中应使用真实的训练/验证数据
            data = np.random.random_sample((1,) + input_shape).astype(np.float32)
            yield [data]

    print(f"✓ 代表性数据集生成器创建完成")
    return representative_dataset_gen


def quantize_model_int8(model, representative_dataset_gen, output_path):
    """
    将模型量化为INT8 TFLite格式

    Args:
        model: TensorFlow/Keras模型
        representative_dataset_gen: 代表性数据集生成器
        output_path: 输出路径

    Returns:
        量化后的模型路径
    """
    print(f"\n开始INT8量化...")

    # 创建TFLite转换器
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 设置优化选项 - INT8量化
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 设置代表性数据集（用于校准量化参数）
    converter.representative_dataset = representative_dataset_gen

    # 确保输入输出都是INT8（全整数量化）
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # 或 tf.uint8
    converter.inference_output_type = tf.int8  # 或 tf.uint8

    try:
        # 执行转换
        print("  正在转换...")
        tflite_model = converter.convert()

        # 保存模型
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        model_size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✓ INT8量化完成")
        print(f"  模型大小: {model_size_mb:.2f} MB")
        print(f"  保存路径: {output_path}")

        return output_path

    except Exception as e:
        print(f"✗ 量化失败: {e}")
        print("\n常见问题:")
        print("  1. 代表性数据集格式不正确")
        print("  2. 模型包含不支持INT8量化的操作")
        print("  3. 输入/输出类型设置不正确")
        raise


def quantize_model_dynamic(model, output_path):
    """
    动态范围量化（权重INT8，激活FP32）

    这是一种更简单的量化方式，不需要代表性数据集

    Args:
        model: TensorFlow/Keras模型
        output_path: 输出路径

    Returns:
        量化后的模型路径
    """
    print(f"\n开始动态范围量化...")

    # 创建TFLite转换器
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 设置优化选项 - 动态范围量化
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    try:
        # 执行转换
        print("  正在转换...")
        tflite_model = converter.convert()

        # 保存模型
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        model_size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✓ 动态范围量化完成")
        print(f"  模型大小: {model_size_mb:.2f} MB")
        print(f"  保存路径: {output_path}")

        return output_path

    except Exception as e:
        print(f"✗ 量化失败: {e}")
        raise


def quantize_model_float16(model, output_path):
    """
    Float16量化

    Args:
        model: TensorFlow/Keras模型
        output_path: 输出路径

    Returns:
        量化后的模型路径
    """
    print(f"\n开始Float16量化...")

    # 创建TFLite转换器
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 设置优化选项 - Float16量化
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    try:
        # 执行转换
        print("  正在转换...")
        tflite_model = converter.convert()

        # 保存模型
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        model_size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✓ Float16量化完成")
        print(f"  模型大小: {model_size_mb:.2f} MB")
        print(f"  保存路径: {output_path}")

        return output_path

    except Exception as e:
        print(f"✗ 量化失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="TFLite INT8 Quantization Tool")
    parser.add_argument("--model", type=str, required=True, help="Path to TensorFlow/Keras model")
    parser.add_argument("--output-dir", type=str, default="./models/quantized", help="Output directory")
    parser.add_argument("--input-shape", type=str, default="224,224,3", help="Input shape (comma-separated)")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of calibration samples")
    parser.add_argument("--quantization", type=str, choices=['int8', 'dynamic', 'float16', 'all'],
                        default='all', help="Quantization type")

    args = parser.parse_args()

    # 解析输入形状
    input_shape = tuple(map(int, args.input_shape.split(',')))

    print(f"配置:")
    print(f"  模型路径: {args.model}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  输入形状: {input_shape}")
    print(f"  校准样本: {args.num_samples}")
    print(f"  量化类型: {args.quantization}")

    # 加载模型
    print(f"\n加载模型: {args.model}")
    try:
        model = tf.keras.models.load_model(args.model)
        print(f"✓ 模型加载成功")
        print(f"  参数总数: {model.count_params():,}")

        # 获取原始模型大小
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            model.save(tmp.name)
            original_size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
            os.unlink(tmp.name)
        print(f"  原始大小: {original_size_mb:.2f} MB")

    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 动态范围量化
    if args.quantization in ['dynamic', 'all']:
        try:
            dynamic_path = output_dir / "model_dynamic.tflite"
            quantize_model_dynamic(model, dynamic_path)
            results['dynamic'] = dynamic_path
        except Exception as e:
            print(f"⚠ 动态范围量化失败: {e}")

    # Float16量化
    if args.quantization in ['float16', 'all']:
        try:
            float16_path = output_dir / "model_float16.tflite"
            quantize_model_float16(model, float16_path)
            results['float16'] = float16_path
        except Exception as e:
            print(f"⚠ Float16量化失败: {e}")

    # INT8量化
    if args.quantization in ['int8', 'all']:
        try:
            # 创建代表性数据集
            representative_dataset_gen = create_representative_dataset(
                input_shape,
                args.num_samples
            )

            int8_path = output_dir / "model_int8.tflite"
            quantize_model_int8(model, representative_dataset_gen, int8_path)
            results['int8'] = int8_path
        except Exception as e:
            print(f"⚠ INT8量化失败: {e}")
            print("\n提示:")
            print("  INT8全整数量化要求:")
            print("  1. 提供代表性数据集用于校准")
            print("  2. 模型所有操作支持INT8")
            print("  3. 正确设置输入/输出类型")

    # 总结
    print(f"\n{'='*70}")
    print("量化完成总结")
    print(f"{'='*70}")

    if results:
        print(f"\n成功生成的量化模型:")
        for quant_type, path in results.items():
            size_mb = os.path.getsize(path) / (1024 * 1024)
            compression_ratio = original_size_mb / size_mb
            print(f"  {quant_type:12s}: {path}")
            print(f"              大小: {size_mb:.2f} MB (压缩 {compression_ratio:.2f}x)")
    else:
        print("\n✗ 没有成功生成任何量化模型")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
