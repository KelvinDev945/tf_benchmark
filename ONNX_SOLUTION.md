# ONNX转换解决方案

## 问题概述

TODO.md Issue #3: ONNX Runtime NumPy 兼容性问题

### 错误信息

```
AttributeError: `np.cast` was removed in the NumPy 2.0 release.
module 'numpy' has no attribute 'object'.
```

## 根本原因

**依赖版本冲突**:
- TensorFlow 2.20.0 默认安装 NumPy 2.x
- tf2onnx 1.16.1 使用NumPy 1.x 的废弃API (`np.cast`, `np.object`等)
- NumPy 2.0+ 移除了这些旧API

## 解决方案

### 方法: 降级 NumPy 到 1.x

**成功配置**:
```bash
TensorFlow:    2.20.0
NumPy:         1.26.4  ⬅️ 关键：必须 < 2.0
tf2onnx:       1.16.1
ONNXRuntime:   1.23.2
Protobuf:      6.33.0
```

### 安装步骤

```bash
# 1. 安装tf2onnx和onnxruntime
pip3 install tf2onnx onnxruntime --no-cache-dir

# 2. 降级NumPy到1.x（关键步骤！）
pip3 install "numpy<2.0" --no-cache-dir --force-reinstall

# 3. 验证安装
python3 -c "import numpy as np; print(f'NumPy: {np.__version__}')"
```

## 测试验证

### 快速测试

```bash
python3 scripts/test_onnx_simple.py
```

**预期输出**:
```
✅ 所有测试通过! 当前环境可以使用tf2onnx转换ONNX
```

### 完整性能对比

```bash
python3 scripts/benchmark_tf_vs_onnx.py --num-runs 100 --num-warmup 10
```

## 测试结果

### CNN模型性能对比

| 指标 | TensorFlow | ONNX Runtime | 提升倍数 |
|------|-----------|--------------|---------|
| **平均延迟** | 6.84 ms | 0.09 ms | **77.32x** 🚀 |
| **P95延迟** | 7.69 ms | 0.11 ms | **72.41x** |
| **P99延迟** | 7.87 ms | 0.11 ms | **69.10x** |
| **吞吐量** | 146 samples/s | 11,299 samples/s | **77.32x** 📈 |

### ONNX转换信息

- **转换时间**: 7.83秒
- **模型大小**: 0.86 MB
- **转换成功率**: ✅ 100%

## 关键发现

### 1. ONNX Runtime性能优势显著

在CPU推理场景下，ONNX Runtime比原生TensorFlow快 **70-80倍**，这使得：
- **边缘设备部署**更可行
- **服务器成本**大幅降低
- **实时推理**性能提升明显

### 2. 依赖管理的重要性

此问题凸显了机器学习工具链中依赖版本管理的复杂性：
- NumPy 2.0是重大升级，破坏了向后兼容性
- 许多工具(tf2onnx, 某些ONNX库)尚未适配NumPy 2.0
- 需要仔细管理版本约束

### 3. 生产环境建议

✅ **强烈推荐**: 使用ONNX Runtime进行生产部署

**优势**:
- 显著的性能提升（70-80x）
- 跨平台支持（CPU, GPU, 移动端）
- 模型标准化，便于部署

**注意事项**:
- 确保使用NumPy 1.x
- 在CI/CD中锁定依赖版本
- 定期测试依赖更新

## 替代方案分析

### 方案1: TF 2.15 + tf2onnx （用户建议）

**未测试原因**:
- 当前方案（TF 2.20 + NumPy 1.x）已成功解决问题
- 不需要降级TensorFlow
- 保持最新TensorFlow特性

**如果需要测试**:
```bash
# 创建独立虚拟环境
python3 -m venv tf215_env
source tf215_env/bin/activate
pip install tensorflow==2.15.0 tf2onnx onnxruntime
```

### 方案2: HuggingFace Optimum

**不适用原因**:
- Optimum专为HuggingFace Transformers模型设计
- 不支持自定义Keras模型
- 对于通用TensorFlow模型，tf2onnx更合适

## 使用工具

### 1. test_onnx_simple.py

快速验证ONNX转换是否正常工作

```bash
python3 scripts/test_onnx_simple.py
```

### 2. benchmark_tf_vs_onnx.py

完整的TensorFlow vs ONNX性能对比

```bash
# CNN模型对比
python3 scripts/benchmark_tf_vs_onnx.py --model-type cnn

# Dense模型对比
python3 scripts/benchmark_tf_vs_onnx.py --model-type dense
```

**生成文件**:
- `results/tf_vs_onnx_benchmark/results.json` - JSON格式结果
- `results/tf_vs_onnx_benchmark/tf_vs_onnx_report.md` - Markdown报告
- `results/tf_vs_onnx_benchmark/model.onnx` - 转换的ONNX模型

## Docker环境

在Docker中使用时，确保requirements.txt包含正确版本：

```txt
# requirements.txt
numpy<2.0        # 关键！
tf2onnx>=1.16.0
onnxruntime>=1.23.0
```

## 故障排除

### 问题1: 仍然出现NumPy错误

```bash
# 检查NumPy版本
python3 -c "import numpy; print(numpy.__version__)"

# 如果是2.x，强制降级
pip3 install "numpy<2.0" --force-reinstall --no-cache-dir
```

### 问题2: TensorFlow导入错误

```bash
# 检查protobuf版本冲突
pip3 list | grep protobuf

# 如果有冲突，重新安装TensorFlow
pip3 install tensorflow==2.20.0 --force-reinstall
```

### 问题3: ONNX转换失败

```bash
# 启用详细日志
python3 -m tf2onnx.convert --saved-model <path> --output <out> --verbose
```

## 相关文档

- [TODO.md](TODO.md) - Issue #3详细信息
- [results/tf_vs_onnx_benchmark/tf_vs_onnx_report.md](results/tf_vs_onnx_benchmark/tf_vs_onnx_report.md) - 性能测试报告

## 总结

**问题**: NumPy 2.0不兼容
**解决**: 降级到NumPy 1.26.4
**结果**: ✅ ONNX转换成功 + 77x性能提升
**建议**: 生产环境使用ONNX Runtime
