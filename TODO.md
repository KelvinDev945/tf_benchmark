# TODO - TensorFlow Benchmark 待办事项

**最后更新**: 2025-11-08

---

## 🔴 高优先级 - 阻塞性问题

### 1. 修复 TensorFlow Engine 类型检查错误

**文件**: `src/engines/tensorflow_engine.py:84-102`

**问题描述**:
```
Invalid model_path type: TFBertForSequenceClassification.
Expected str or tf.keras.Model
```

**根本原因**: 
- HuggingFace 的 `TFBertForSequenceClassification` 不是 `tf.keras.Model` 的直接实例
- 当前代码使用 `isinstance(model_path, tf.keras.Model)` 检查失败
- Transformers 模型虽然基于 Keras，但有自己的基类

**修复方案**:
```python
# 修改 src/engines/tensorflow_engine.py 第84行
# 原代码:
if isinstance(model_path, tf.keras.Model):

# 改为:
if hasattr(model_path, '__call__') and hasattr(model_path, 'predict'):
    # 接受任何可调用的 TensorFlow 模型（Keras 或 Transformers）
```

**影响**: 🔴 阻塞所有 TensorFlow 相关的 BERT 测试

**状态**: ❌ 未修复

---

## 🟡 中优先级 - 功能问题

### 2. 修复 TFLite INT8 量化转换错误

**文件**: `src/models/model_converter.py` 或相关转换代码

**问题描述**:
```
TFLite conversion failed: object of type 'function' has no len()
```

**根本原因**:
- TFLite 量化需要 representative dataset generator
- 代码传入了函数对象，但某处尝试获取其长度
- 可能是 generator 函数使用不正确

**需要调查**:
- [ ] 检查 representative dataset 的实现
- [ ] 确认 generator 函数的正确用法
- [ ] 查看 TFLite 转换代码中的数据格式要求

**影响**: 🟡 影响 INT8 量化模型测试

**状态**: ❌ 未修复

---

### 3. 解决 ONNX Runtime NumPy 兼容性问题

**文件**: ONNX 转换相关代码

**问题描述**:
```
module 'numpy' has no attribute 'object'.
`np.object` was a deprecated alias for the builtin `object`.
```

**根本原因**:
- NumPy 1.20+ 废弃了 `np.object` 别名
- tf2onnx 或相关库使用了过时的 NumPy API
- 环境中的 NumPy 版本较新，与 tf2onnx 不兼容

**可能的解决方案**:
1. 更新 tf2onnx 到最新版本
2. 降级 NumPy 版本到 < 1.20（可能影响其他包）
3. 使用 monkey patch 临时修复

**影响**: 🟡 影响 ONNX Runtime 测试

**状态**: ❌ 未修复

---

## 🟢 低优先级 - 优化和增强

### 4. 添加更多模型支持

- [ ] GPT 系列模型
- [ ] T5 模型
- [ ] Vision Transformer (ViT)
- [ ] 目标检测模型 (YOLO, SSD)

### 5. 性能优化

- [ ] 添加批处理优化
- [ ] 实现多线程并行测试
- [ ] 添加模型缓存机制
- [ ] 优化数据加载流程

### 6. 报告增强

- [ ] 添加交互式图表 (Plotly)
- [ ] 支持 PDF 导出
- [ ] 添加历史对比功能
- [ ] 生成 CI/CD 集成报告

### 7. 文档改进

- [ ] 添加更多使用示例
- [ ] 创建视频教程
- [ ] 添加最佳实践指南
- [ ] 完善 API 文档

---

## 📝 技术债务

### 8. 代码质量改进

- [ ] 增加单元测试覆盖率到 90%+
- [ ] 添加集成测试
- [ ] 完善错误处理
- [ ] 添加更多类型注解

### 9. 配置管理

- [ ] 支持配置文件模板
- [ ] 添加配置验证器
- [ ] 支持环境变量配置
- [ ] 添加配置迁移工具

---

## 🐛 已知问题（非阻塞）

### PyTorch 依赖问题

**说明**:
在某次运行中看到：
```
Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed.
✗ TensorFlow baseline benchmark failed: No module named 'torch'
```

**解决方案**:
- HuggingFace 模型尝试从 PyTorch 权重转换
- 通过设置 `from_pt=False` 和 `use_safetensors=False` 可以强制使用 TF 权重
- 或预先转换模型为 TensorFlow SavedModel 格式

**影响**: 仅影响某些 HuggingFace 模型

---

### SafeTensors 格式兼容性

**说明**:
另一次运行看到：
```
✗ TensorFlow baseline benchmark failed: 'builtins.safe_open' object is not iterable
```

**解决方案**:
- 使用 `model.safetensors` 时可能出现兼容性问题
- 建议使用 `tf_model.h5` 格式（已在代码中设置）

**影响**: 仅影响特定模型格式

---

## ✅ 最近完成

- [x] 精简项目文档（从 12 个减少到 3 个）
- [x] 合并 TODO 内容到 README.md
- [x] 创建 BERT 专项测试框架
- [x] 添加综合报告生成工具
- [x] 实现完整的 benchmark 流程

---

## 📋 测试环境信息

- **TensorFlow 版本**: 2.20.0
- **Python 版本**: 3.11
- **Docker 镜像**: tf-cpu-benchmark:latest
- **测试模型**: google-bert/bert-base-uncased
- **测试数据集**: glue/sst2 (validation split)

---

## 🎯 近期目标

1. **本周**: 修复 TensorFlow Engine 类型检查问题（Issue #1）
2. **本月**: 解决所有量化和 ONNX 相关问题
3. **下月**: 添加更多模型支持和性能优化

---

## 📚 相关文档

- [README.md](README.md) - 项目主文档
- [PROJECT_COMPLETE.md](PROJECT_COMPLETE.md) - 完整项目文档
- [BERT_BENCHMARK_GUIDE.md](BERT_BENCHMARK_GUIDE.md) - BERT 使用指南

---

**维护者**: 请定期更新此文档，标记已完成的任务 ✅

