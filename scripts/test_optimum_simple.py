#!/usr/bin/env python3
"""
快速测试Optimum ONNX转换功能

验证HuggingFace Optimum是否正确安装并能进行ONNX转换
"""

import sys


def test_imports():
    """测试所有必要的包是否能正常导入"""
    print("=" * 70)
    print("测试包导入")
    print("=" * 70)

    imports_ok = True

    # Test TensorFlow
    try:
        import tensorflow as tf

        print(f"✓ TensorFlow: {tf.__version__}")
    except Exception as e:
        print(f"✗ TensorFlow: {e}")
        imports_ok = False

    # Test ONNX Runtime
    try:
        import onnxruntime as ort

        print(f"✓ ONNX Runtime: {ort.__version__}")
    except Exception as e:
        print(f"✗ ONNX Runtime: {e}")
        imports_ok = False

    # Test Optimum
    try:
        import optimum  # noqa: F401

        # optimum doesn't have __version__, check via importlib
        try:
            import importlib.metadata

            version = importlib.metadata.version("optimum")
            print(f"✓ Optimum: {version}")
        except Exception:
            print("✓ Optimum: installed")
    except Exception as e:
        print(f"✗ Optimum: {e}")
        imports_ok = False

    # Test Transformers
    try:
        import transformers

        print(f"✓ Transformers: {transformers.__version__}")
    except Exception as e:
        print(f"✗ Transformers: {e}")
        imports_ok = False

    # Test NumPy
    try:
        import numpy as np

        print(f"✓ NumPy: {np.__version__}")
    except Exception as e:
        print(f"✗ NumPy: {e}")
        imports_ok = False

    # Test Protobuf
    try:
        import google.protobuf

        print(f"✓ Protobuf: {google.protobuf.__version__}")
    except Exception as e:
        print(f"✗ Protobuf: {e}")

    return imports_ok


def test_optimum_onnx():
    """测试Optimum ONNX转换功能"""
    print("\n" + "=" * 70)
    print("测试Optimum ONNX转换")
    print("=" * 70)

    try:
        import shutil
        import tempfile
        from pathlib import Path

        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        # 使用最小的BERT模型进行测试
        model_name = "prajjwal1/bert-tiny"
        print(f"\n测试模型: {model_name}")
        print("正在下载并转换为ONNX（这可能需要1-2分钟）...")

        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)

        try:
            # 转换为ONNX
            print("1. 转换为ONNX...")
            ort_model = ORTModelForSequenceClassification.from_pretrained(model_name, export=True)

            # 保存模型
            print("2. 保存ONNX模型...")
            ort_model.save_pretrained(temp_path)

            # 加载tokenizer
            print("3. 加载tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # 测试推理
            print("4. 测试ONNX推理...")
            test_text = "This is a test sentence."
            inputs = tokenizer(test_text, return_tensors="pt")
            outputs = ort_model(**inputs)

            print("\n✓ ONNX转换和推理测试成功!")
            print(f"  - 输出形状: {outputs.logits.shape}")
            print(f"  - ONNX文件位置: {temp_path / 'model.onnx'}")

            # 检查文件大小
            onnx_file = temp_path / "model.onnx"
            if onnx_file.exists():
                size_mb = onnx_file.stat().st_size / (1024 * 1024)
                print(f"  - ONNX模型大小: {size_mb:.2f} MB")

            return True

        finally:
            # 清理临时文件
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"\n✗ ONNX转换测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("\n🚀 Optimum ONNX 功能测试")
    print("=" * 70)

    # 测试导入
    if not test_imports():
        print("\n❌ 包导入测试失败!")
        print("请检查依赖是否正确安装: pip install optimum[onnxruntime]")
        sys.exit(1)

    # 测试ONNX转换
    if not test_optimum_onnx():
        print("\n❌ ONNX转换测试失败!")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ 所有测试通过! Optimum ONNX功能正常工作")
    print("=" * 70)
    print("\n下一步:")
    print("  运行完整基准测试:")
    print("  python3 scripts/benchmark_bert_optimum.py --num-runs 100")
    print()


if __name__ == "__main__":
    main()
