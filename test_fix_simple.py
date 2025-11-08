#!/usr/bin/env python3
"""
简化版 TensorFlow Engine 修复测试
直接导入模块，不依赖其他引擎
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras

print('=' * 70)
print('TensorFlow Engine Type Check Fix - 简化测试')
print('=' * 70)

# 直接导入 TensorFlowEngine 类，绕过 __init__.py
sys.path.insert(0, os.path.dirname(__file__))
exec(open('src/engines/base_engine.py').read(), globals())
exec(open('src/engines/tensorflow_engine.py').read(), globals())

print('\n✓ 成功导入 TensorFlowEngine')
print(f'  TensorFlow 版本: {tf.__version__}')
print(f'  NumPy 版本: {np.__version__}')

# 创建简单的 Keras 模型
print('\n测试 1: 创建 Keras Sequential 模型')
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
print(f'  ✓ 模型类型: {type(model).__name__}')
print(f'  ✓ 有 __call__ 方法: {hasattr(model, "__call__")}')
print(f'  ✓ 有 predict 方法: {hasattr(model, "predict")}')
print(f'  ✓ isinstance(tf.keras.Model): {isinstance(model, tf.keras.Model)}')

# 测试加载 Keras 模型对象
print('\n测试 2: TensorFlowEngine 加载 Keras 模型对象')
try:
    engine = TensorFlowEngine(config={'xla': False, 'mixed_precision': False})
    engine.load_model(model)
    print('  ✓ 成功加载 Keras 模型对象！')

    # 测试推理
    dummy_input = np.random.randn(1, 10).astype(np.float32)
    output = engine.infer(dummy_input)
    print(f'  ✓ 推理成功！输出 shape: {output.shape}')

    engine.cleanup()
    print('  ✓ 清理成功')
except Exception as e:
    print(f'  ✗ 失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试从路径加载
print('\n测试 3: 从 SavedModel 路径加载')
try:
    model_path = '/tmp/test_keras_model'
    model.save(model_path)
    print(f'  ✓ 模型保存到: {model_path}')

    engine = TensorFlowEngine(config={'xla': False, 'mixed_precision': False})
    engine.load_model(model_path)
    print('  ✓ 成功从路径加载模型！')

    # 测试推理
    dummy_input = np.random.randn(1, 10).astype(np.float32)
    output = engine.infer(dummy_input)
    print(f'  ✓ 推理成功！输出 shape: {output.shape}')

    engine.cleanup()

    # 清理
    import shutil
    shutil.rmtree(model_path)
    print('  ✓ 清理成功')
except Exception as e:
    print(f'  ✗ 失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 .h5 格式
print('\n测试 4: 从 .h5 文件加载')
try:
    h5_path = '/tmp/test_model.h5'
    model.save(h5_path)
    print(f'  ✓ 模型保存到: {h5_path}')

    engine = TensorFlowEngine(config={'xla': False, 'mixed_precision': False})
    engine.load_model(h5_path)
    print('  ✓ 成功从 .h5 文件加载模型！')

    # 测试推理
    dummy_input = np.random.randn(1, 10).astype(np.float32)
    output = engine.infer(dummy_input)
    print(f'  ✓ 推理成功！输出 shape: {output.shape}')

    engine.cleanup()

    # 清理
    os.remove(h5_path)
    print('  ✓ 清理成功')
except Exception as e:
    print(f'  ✗ 失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试无效输入
print('\n测试 5: 无效输入类型（应该优雅地失败）')
try:
    engine = TensorFlowEngine(config={'xla': False, 'mixed_precision': False})
    engine.load_model(12345)
    print('  ✗ 应该抛出错误！')
    sys.exit(1)
except Exception as e:
    error_msg = str(e)
    if 'Invalid model_path type' in error_msg and 'int' in error_msg:
        print(f'  ✓ 正确拒绝无效类型')
        print(f'    错误信息: "{error_msg}"')
    else:
        print(f'  ✗ 错误信息不正确: {error_msg}')
        sys.exit(1)

print('\n' + '=' * 70)
print('✓ 所有测试通过！')
print('=' * 70)
print('\n修复验证成功:')
print('  ✓ 可以加载 Keras 模型对象')
print('  ✓ 可以从 SavedModel 路径加载')
print('  ✓ 可以从 .h5 文件加载')
print('  ✓ 正确拒绝无效输入')
print('\n这意味着 HuggingFace Transformers 模型也应该能正常工作，')
print('因为它们同样具有 __call__ 和 predict 方法。')
print('=' * 70)
