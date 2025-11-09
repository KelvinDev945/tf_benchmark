#!/usr/bin/env python3
"""简单ONNX转换测试 - 测试当前环境配置"""

import sys

print("=" * 70)
print("环境检查")
print("=" * 70)

# 检查TensorFlow
try:
    import tensorflow as tf
    print(f"✓ TensorFlow: {tf.__version__}")
except Exception as e:
    print(f"✗ TensorFlow 导入失败: {e}")
    sys.exit(1)

# 检查protobuf
try:
    import google.protobuf
    print(f"✓ Protobuf: {google.protobuf.__version__}")
except Exception as e:
    print(f"✗ Protobuf: {e}")

# 检查tf2onnx
try:
    import tf2onnx
    print(f"✓ tf2onnx: {tf2onnx.__version__}")
except Exception as e:
    print(f"✗ tf2onnx 导入失败: {e}")
    print("\n注意: tf2onnx可能与当前protobuf版本不兼容")
    sys.exit(1)

# 检查onnxruntime
try:
    import onnxruntime as ort
    print(f"✓ ONNXRuntime: {ort.__version__}")
except Exception as e:
    print(f"✗ ONNXRuntime: {e}")

print("\n" + "=" * 70)
print("创建和转换测试模型")
print("=" * 70)

# 创建简单模型
import numpy as np
from pathlib import Path

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print("✓ 测试模型创建完成")

# 保存为SavedModel
output_dir = Path("results/onnx_test")
output_dir.mkdir(parents=True, exist_ok=True)

saved_model_path = output_dir / "test_model"
# TF 2.20+ 使用 export() 而不是 save() 来保存SavedModel
model.export(saved_model_path)
print(f"✓ SavedModel已保存: {saved_model_path}")

# 尝试转换为ONNX
onnx_path = output_dir / "test_model.onnx"

print(f"\n尝试转换为ONNX...")
try:
    import subprocess
    cmd = [
        "python3", "-m", "tf2onnx.convert",
        "--saved-model", str(saved_model_path),
        "--output", str(onnx_path),
        "--opset", "13"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode == 0:
        print("✓ ONNX转换成功!")
        print(f"✓ ONNX模型已保存: {onnx_path}")

        # 测试ONNX推理
        print("\n测试ONNX推理...")
        sess = ort.InferenceSession(str(onnx_path))
        test_input = np.random.randn(1, 10).astype(np.float32)
        input_name = sess.get_inputs()[0].name
        output = sess.run(None, {input_name: test_input})
        print(f"✓ ONNX推理成功! 输出形状: {output[0].shape}")

        print("\n" + "=" * 70)
        print("✅ 所有测试通过! 当前环境可以使用tf2onnx转换ONNX")
        print("=" * 70)
    else:
        print("✗ ONNX转换失败")
        print(f"\n错误输出:\n{result.stderr}")
        sys.exit(1)

except Exception as e:
    print(f"✗ 转换过程出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
