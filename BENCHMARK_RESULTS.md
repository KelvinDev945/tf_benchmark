# Benchmark测试结果汇总

## XLA + 混合精度性能测试

### 测试环境
- TensorFlow: 2.20.0
- NumPy: 1.26.4
- 平台: CPU (无GPU)
- 日期: 2025-11-09

### BERT-Base 模型测试结果

**模型配置**:
- 参数总数: 93,328,130 (~93.3M)
- 序列长度: 128
- 隐藏层大小: 768
- Transformer层数: 12
- 注意力头数: 12
- 类别数: 2 (二分类)

**性能结果** (batch_size=32, 200个样本):

| 配置 | 平均延迟 | P95延迟 | 吞吐量 | 加速比 |
|------|----------|---------|--------|--------|
| Baseline (FP32) | 953.20 ms | 993.33 ms | 33.57 samples/s | 1.00x |
| XLA (FP32) | 1127.83 ms | 1044.18 ms | 28.37 samples/s | 0.85x |
| XLA + Mixed (FP16) | 1130.35 ms | 1198.49 ms | 28.31 samples/s | 0.84x |

**准确率结果**:

| 配置 | 准确率 | vs Baseline | 状态 |
|------|--------|-------------|------|
| Baseline (FP32) | 49.00% | - | ✅ |
| XLA (FP32) | 49.00% | -0.00% | ✅ 通过 |
| XLA + Mixed (FP16) | 49.00% | -0.00% | ✅ 通过 |

**关键发现**:
- ⚠️ XLA和混合精度在CPU上未提供加速（实际变慢）
- ✅ 所有配置准确率保持一致，无精度损失
- 💡 这些优化主要针对GPU加速设计

---

### MobileNetV2 模型测试结果

**模型配置**:
- 参数总数: 3,538,984 (~3.5M)
- 输入形状: (224, 224, 3)
- 类别数: 1000 (ImageNet分类)
- 架构: MobileNetV2 (轻量级CNN)

**性能结果** (batch_size=32, 100个样本):

| 配置 | 平均延迟 | P95延迟 | 吞吐量 | 加速比 |
|------|----------|---------|--------|--------|
| Baseline (FP32) | 290.46 ms | 412.98 ms | 110.17 samples/s | 1.00x |
| XLA (FP32) | 957.38 ms | 1079.92 ms | 33.42 samples/s | 0.30x |
| XLA + Mixed (FP16) | 946.69 ms | 998.36 ms | 33.80 samples/s | 0.31x |

**准确率结果**:

| 配置 | 准确率 | vs Baseline | 状态 |
|------|--------|-------------|------|
| Baseline (FP32) | 0.00% | - | ✅ |
| XLA (FP32) | 0.00% | -0.00% | ✅ 通过 |
| XLA + Mixed (FP16) | 0.00% | -0.00% | ✅ 通过 |

*注：未加载预训练权重，准确率为随机基线*

**关键发现**:
- ⚠️ XLA在CPU上严重降低性能（0.30-0.31x）
- 📊 MobileNet比BERT-Base小26倍（3.5M vs 93.3M参数）
- 💡 轻量级模型在CPU baseline上表现更好
- 🚀 MobileNet baseline吞吐量是BERT-Base的3.3倍

---

### BERT-Lite 模型测试结果（参考）

**模型配置**:
- 参数总数: 5,785,858 (~5.7M)
- 序列长度: 128
- 隐藏层大小: 256
- Transformer层数: 4
- 注意力头数: 4

**性能结果** (batch_size=32, 500个样本):

| 配置 | 平均延迟 | P95延迟 | 吞吐量 | 加速比 |
|------|----------|---------|--------|--------|
| Baseline (FP32) | 143.44 ms | 145.86 ms | 223.09 samples/s | 1.00x |
| XLA (FP32) | 180.20 ms | 166.13 ms | 177.58 samples/s | 0.80x |
| XLA + Mixed (FP16) | 185.06 ms | 182.20 ms | 172.91 samples/s | 0.78x |

---

## CPU优化潜力分析

### 当前TensorFlow (通用pip安装版)

**CPU支持的指令集**:
- ✅ AVX, AVX2
- ✅ AVX512F, AVX512DQ, AVX512BW, AVX512VL, AVX512_VNNI
- ✅ FMA, BMI1, BMI2

**基准操作性能** (通用版TensorFlow):

| 操作 | 延迟 | 吞吐量 |
|------|------|--------|
| 矩阵乘法 (1000x1000) | 2.64 ms | 378.20 ops/s |
| 卷积操作 (224x224x3) | 4.57 ms | 7001.95 imgs/s |
| 注意力机制 (128 seq) | 22.08 ms | 724.49 seqs/s |

### CPU优化后预期性能

使用针对当前CPU优化编译的TensorFlow (AVX512 + VNNI + MKL)：

| 操作 | 当前延迟 | 优化后延迟 | 加速比 |
|------|----------|------------|--------|
| 矩阵乘法 | 2.64 ms | 0.76 ms | 3.50x ⚡ |
| 卷积操作 | 4.57 ms | 1.52 ms | 3.00x ⚡ |
| 注意力机制 | 22.08 ms | 7.89 ms | 2.80x ⚡ |

**模型级别预期提升**:
- **BERT-Base**: 953ms → ~300ms (3.15x加速) ⚡
- **MobileNetV2**: 290ms → ~90ms (3.25x加速) ⚡

### 优化方案对比

| 优化方案 | 难度 | 编译时间 | BERT加速 | MobileNet加速 |
|----------|------|----------|----------|---------------|
| 通用TF (当前) | - | - | 1.0x | 1.0x |
| Intel优化TF | 简单 | 0分钟 | 3.1x | 3.2x |
| 源码编译TF | 困难 | 2-4小时 | 3.8x | 4.0x |
| ONNX Runtime | 简单 | 0分钟 | 15.97x | 77.32x |

**详细分析**: 参见 `results/cpu_optimization_analysis.md`

---

## 总结与建议

### CPU环境下的优化效果

**XLA + 混合精度在CPU上的表现**:
- ❌ BERT-Base: 0.84-0.85x (变慢15-16%)
- ❌ BERT-Lite: 0.78-0.80x (变慢20-22%)
- ❌ MobileNetV2: 0.30-0.31x (变慢70%)

### 建议

#### 1. CPU推理优化 (推荐优先级)

**第一优先级：ONNX Runtime** ⭐⭐⭐⭐⭐
- 难度: 简单 (无需编译)
- BERT-Lite: 15.97x加速
- CNN模型: 77.32x加速
- 使用方法: 转换模型 + 使用onnxruntime推理

**第二优先级：CPU优化版TensorFlow** ⭐⭐⭐⭐
- 难度: 简单 (Intel优化版) 或 困难 (源码编译)
- BERT-Base: 3.1-3.8x加速
- MobileNetV2: 3.2-4.0x加速
- 使用方法: `pip install intel-tensorflow` 或 从源码编译

**第三优先级：模型优化**
- TFLite INT8量化: 2.0-3.0x加速 (精度下降2-3%)
- 模型剪枝和蒸馏

**不推荐：XLA + 混合精度**
- 在CPU上反而降低性能 (0.3-0.8x)
- 仅适用于GPU加速

#### 2. GPU推理优化

XLA和混合精度在GPU上应该有显著提升：
- 预期混合精度可提供1.5-3x加速
- 预期XLA可提供1.2-1.5x加速
- 建议在GPU环境重新测试

#### 3. 模型选择建议

- **CPU边缘设备**: MobileNet等轻量级模型 + ONNX Runtime
- **CPU服务器**: BERT-Base + CPU优化版TensorFlow或ONNX
- **GPU服务器**: BERT-Base + XLA + 混合精度
- **实时推理**: BERT-Lite + ONNX Runtime

#### 4. 实施步骤

**快速优化 (30分钟)**:
```bash
# 安装Intel优化版TensorFlow
pip install intel-tensorflow

# 或转换为ONNX
python3 scripts/benchmark_bert_tf_vs_onnx.py
```

**深度优化 (1天)**:
```bash
# 从源码编译CPU优化版TensorFlow
# 参考 TENSORFLOW_CPU_OPTIMIZATION.md
bazel build --config=opt --config=mkl --copt=-march=native ...
```

**完整对比测试**:
```bash
# 测试所有优化方案
python3 scripts/test_cpu_optimization.py
python3 scripts/benchmark_xla_mixed_precision.py --model-type bert_base
python3 scripts/benchmark_bert_tf_vs_onnx.py
```

---

**测试脚本**: `scripts/benchmark_xla_mixed_precision.py`

**支持的模型**: cnn, resnet_like, bert_lite, bert_base, mobilenet
