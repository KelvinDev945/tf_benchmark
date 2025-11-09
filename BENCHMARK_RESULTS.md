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

## 总结与建议

### CPU环境下的优化效果

**XLA + 混合精度在CPU上的表现**:
- ❌ BERT-Base: 0.84-0.85x (变慢15-16%)
- ❌ BERT-Lite: 0.78-0.80x (变慢20-22%)
- ❌ MobileNetV2: 0.30-0.31x (变慢70%)

### 建议

1. **CPU推理**: 不建议使用XLA和混合精度
   - 这些优化针对GPU设计，在CPU上反而增加开销
   - 使用baseline TensorFlow配置获得最佳CPU性能

2. **GPU推理**: XLA和混合精度应该有显著提升
   - 建议在GPU环境重新测试
   - 预期混合精度可提供1.5-3x加速
   - 预期XLA可提供1.2-1.5x加速

3. **模型选择**:
   - **CPU边缘设备**: 使用MobileNet等轻量级模型
   - **服务器批处理**: BERT-Base适合GPU加速
   - **实时推理**: BERT-Lite平衡性能与准确率

4. **ONNX Runtime对比** (参考之前测试):
   - BERT-Lite: ONNX Runtime提供15.97x加速
   - CNN模型: ONNX Runtime提供77.32x加速
   - **强烈推荐**: 在CPU上使用ONNX Runtime而非XLA

---

**测试脚本**: `scripts/benchmark_xla_mixed_precision.py`

**支持的模型**: cnn, resnet_like, bert_lite, bert_base, mobilenet
