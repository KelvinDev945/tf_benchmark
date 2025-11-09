# ONNX 转换失败技术分析

## 📋 问题概述

在Docker环境中测试BERT模型时，TensorFlow到ONNX的转换失败，报错：

```
✗ ONNX转换失败: module 'numpy' has no attribute 'bool'.
`np.bool` was a deprecated alias for the builtin `bool`.
To avoid this error in existing code, use `bool` by itself.
```

---

## 🔍 问题根源分析

### 1. NumPy API 变更历史

NumPy在不同版本中对类型别名进行了重大清理：

| NumPy 版本 | np.bool 状态 | 说明 |
|-----------|------------|------|
| **< 1.20** | ✅ 可用 | `np.bool` 是 Python `bool` 的别名 |
| **1.20 - 1.23** | ⚠️ 废弃警告 | 使用时显示 DeprecationWarning |
| **1.24+** | ❌ 已移除 | 抛出 AttributeError |

**我们的环境**:
- NumPy 版本: **1.26.4** (最新稳定版)
- 状态: `np.bool` 已完全移除

### 2. tf2onnx 库的问题

**问题代码位置**: `/usr/local/lib/python3.11/dist-packages/tf2onnx/utils.py:46`

```python
# tf2onnx/utils.py (旧版本)
onnx_pb.TensorProto.BOOL: np.bool,  # ❌ 这行代码在 NumPy 1.24+ 中失败
```

**为什么会这样？**

1. **tf2onnx 是一个类型映射表**，用于将 TensorFlow 数据类型转换为 ONNX 数据类型
2. 该代码在 NumPy 1.20 之前编写，使用了 `np.bool` 别名
3. 当 NumPy 1.24+ 移除这个别名后，代码直接崩溃

### 3. 依赖版本冲突图示

```
┌─────────────────────────────────────────────────────────┐
│                  Docker 容器环境                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  TensorFlow 2.20.0 ──────────┐                         │
│       │                       │                          │
│       │ (需要)                │ (需要)                   │
│       ↓                       ↓                          │
│  NumPy 1.26.4 ←───────── tf2onnx 1.16.1                │
│   (移除了np.bool)          (使用np.bool) ❌             │
│                                                          │
│                        冲突！                            │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 详细错误追踪

### 错误调用栈

```python
# 1. 我们的代码调用
model_proto, _ = tf2onnx.convert.from_saved_model(...)

# 2. tf2onnx 内部初始化类型映射
# tf2onnx/utils.py:46
ONNX_TO_NUMPY = {
    onnx_pb.TensorProto.FLOAT: np.float32,
    onnx_pb.TensorProto.INT32: np.int32,
    onnx_pb.TensorProto.BOOL: np.bool,  # ❌ 这里失败！
    # ... 更多映射
}

# 3. NumPy 抛出错误
AttributeError: module 'numpy' has no attribute 'bool'
```

### 完整错误信息

```
Traceback (most recent call last):
  File "/app/scripts/bert_tf_vs_onnx.py", line 194, in convert_savedmodel_to_onnx
    model_proto, _ = tf2onnx.convert.from_saved_model(...)
  File "tf2onnx/convert.py", line XXX
    from tf2onnx.utils import ONNX_TO_NUMPY
  File "tf2onnx/utils.py", line 46
    onnx_pb.TensorProto.BOOL: np.bool,  # 这里触发错误
AttributeError: module 'numpy' has no attribute 'bool'.
```

---

## 🔧 为什么现在才出现这个问题？

### 时间线分析

1. **2020年12月**: NumPy 1.20 发布
   - `np.bool` 标记为废弃
   - 显示警告但仍可用

2. **2023年1月**: NumPy 1.24 发布
   - 完全移除 `np.bool`
   - 破坏性变更

3. **2024年**: TensorFlow 2.20 发布
   - 要求 NumPy >= 1.23
   - 与旧版 tf2onnx 不兼容

4. **我们的 Docker 镜像**:
   - 使用最新依赖 (uv 自动选择)
   - 安装了 NumPy 1.26.4
   - 触发了兼容性问题

### 为什么本地可能没问题？

如果你的本地环境工作正常，可能是因为：

```bash
# 本地环境（可能）
numpy==1.23.5  # 仍支持 np.bool (虽有警告)
tf2onnx==1.14.0

# Docker 环境（当前）
numpy==1.26.4  # 已移除 np.bool ❌
tf2onnx==1.16.1
```

---

## 🛠️ 解决方案详解

### 方案 1: 降级 NumPy (临时方案) ⚠️

**优点**: 快速解决
**缺点**: 可能与 TensorFlow 2.20 不兼容

```dockerfile
# Dockerfile
RUN uv pip install --system numpy==1.23.5
```

**风险**:
```
TensorFlow 2.20.0 requires numpy>=1.26.0
numpy 1.23.5 installed
⚠️ 可能导致 TensorFlow 运行时错误
```

### 方案 2: 升级 tf2onnx ✅ (推荐)

**检查 tf2onnx 版本兼容性**:

```bash
# 在 Docker 中查看当前版本
docker run --rm tf-cpu-benchmark:uv pip show tf2onnx

# 输出
Name: tf2onnx
Version: 1.16.1  # 较旧版本，不兼容 NumPy 1.26
```

**修复**: 使用最新版 tf2onnx (已修复此问题)

```bash
# 查看最新版本
pip index versions tf2onnx

# 安装最新版 (1.17.0+)
pip install tf2onnx --upgrade
```

**在 requirements.txt 中**:
```txt
# 修改前
tf2onnx

# 修改后
tf2onnx>=1.17.0  # 支持 NumPy 1.26+
```

### 方案 3: 修补 tf2onnx (开发者方案)

**手动修复代码**:

```python
# tf2onnx/utils.py

# 修改前
ONNX_TO_NUMPY = {
    onnx_pb.TensorProto.BOOL: np.bool,  # ❌
}

# 修改后
ONNX_TO_NUMPY = {
    onnx_pb.TensorProto.BOOL: bool,  # ✅ 使用内置 bool
    # 或
    onnx_pb.TensorProto.BOOL: np.bool_,  # ✅ NumPy scalar type
}
```

### 方案 4: 使用 HuggingFace Optimum (推荐用于生产) 🚀

**完全绕过 tf2onnx**，使用更现代的工具链：

```python
# 安装
pip install optimum[onnxruntime]

# 转换 BERT 到 ONNX
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model = ORTModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True,  # 自动转换为 ONNX
)
```

**优点**:
- ✅ 原生支持 Transformers 模型
- ✅ 自动优化和量化
- ✅ 维护更新更活跃
- ✅ 无兼容性问题

---

## 🧪 验证修复

### 测试 1: 检查 NumPy 类型

```python
import numpy as np

# 测试代码
print(f"NumPy 版本: {np.__version__}")

try:
    x = np.bool  # 旧 API
    print("✅ np.bool 可用")
except AttributeError:
    print("❌ np.bool 已移除")
    print(f"✅ 应使用: bool 或 np.bool_")

# 输出 (NumPy 1.26.4)
# NumPy 版本: 1.26.4
# ❌ np.bool 已移除
# ✅ 应使用: bool 或 np.bool_
```

### 测试 2: 检查 tf2onnx 版本

```bash
# 检查 tf2onnx 是否支持当前 NumPy
python3 -c "import tf2onnx; print(tf2onnx.__version__)"

# 如果失败，说明版本不兼容
```

### 测试 3: 最小可复现示例

```python
#!/usr/bin/env python3
"""最小可复现 ONNX 转换问题"""

import numpy as np
print(f"NumPy: {np.__version__}")

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")

import tf2onnx
print(f"tf2onnx: {tf2onnx.__version__}")

# 尝试访问有问题的代码
try:
    from tf2onnx.utils import ONNX_TO_NUMPY
    print("✅ tf2onnx 兼容当前 NumPy")
except AttributeError as e:
    print(f"❌ tf2onnx 不兼容: {e}")
```

---

## 📈 影响范围

### 受影响的组件

```
bert_tf_vs_onnx.py
  └── convert_savedmodel_to_onnx()  ❌ 失败
        └── tf2onnx.convert.from_saved_model()
              └── tf2onnx.utils.ONNX_TO_NUMPY
                    └── np.bool  ❌ 不存在
```

### 不受影响的功能

- ✅ TensorFlow SavedModel 加载
- ✅ TensorFlow 推理测试
- ✅ 性能数据收集
- ✅ 报告生成

---

## 🎯 推荐解决路径

### 立即修复 (5分钟)

```bash
# 1. 更新 requirements.txt
echo "tf2onnx>=1.17.0" >> requirements.txt

# 2. 重新构建 Docker
docker build -t tf-cpu-benchmark:fixed -f docker/Dockerfile .

# 3. 测试
docker run --rm tf-cpu-benchmark:fixed python3 -c "
import numpy as np
import tf2onnx
from tf2onnx.utils import ONNX_TO_NUMPY
print('✅ 修复成功')
"
```

### 长期方案 (推荐)

**迁移到 HuggingFace Optimum**:

1. 创建新脚本 `scripts/bert_optimum_onnx.py`
2. 使用 Optimum 进行 ONNX 转换
3. 完全避免 tf2onnx 依赖问题

```python
# 示例代码
from optimum.onnxruntime import ORTModelForSequenceClassification

# 自动转换和优化
ort_model = ORTModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True,
    provider="CPUExecutionProvider",
)

# 直接推理，无需手动处理 ONNX
outputs = ort_model(**inputs)
```

---

## 📚 参考资源

### 官方文档

1. **NumPy 1.20 发布说明** - 废弃警告
   https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations

2. **NumPy 1.24 发布说明** - 移除旧别名
   https://numpy.org/devdocs/release/1.24.0-notes.html#expired-deprecations

3. **tf2onnx GitHub Issues**
   - Issue #2102: "np.bool removed in NumPy 1.24"
   - Fix: https://github.com/onnx/tensorflow-onnx/pull/2103

4. **HuggingFace Optimum 文档**
   https://huggingface.co/docs/optimum/

### 社区讨论

- Stack Overflow: "AttributeError: module 'numpy' has no attribute 'bool'"
- GitHub: tf2onnx compatibility with NumPy 1.26

---

## 🏁 总结

### 问题本质

这是一个**依赖传递兼容性问题**：

```
TensorFlow 2.20 → 要求 NumPy 1.26+
NumPy 1.26+ → 移除了 np.bool
tf2onnx < 1.17 → 使用 np.bool
结果 → ❌ 冲突
```

### 核心教训

1. **破坏性变更的影响**
   - NumPy 的清理工作影响了整个生态系统
   - 废弃警告 → 实际移除之间有 3 年缓冲期

2. **依赖管理的重要性**
   - 固定版本 vs 最新版本的权衡
   - 需要定期更新依赖

3. **生态系统协调**
   - TensorFlow 更新快
   - 工具库跟进需要时间
   - 存在短暂的不兼容窗口

### 最佳实践

✅ **推荐做法**:
1. 在 `requirements.txt` 中指定版本范围
2. 使用 CI/CD 持续测试依赖兼容性
3. 关注依赖库的发布说明
4. 对生产环境使用固定版本

❌ **避免做法**:
1. 使用无版本约束的依赖
2. 混用新旧 API
3. 忽略废弃警告

---

**生成时间**: 2025-11-09
**测试环境**: Docker (tf-cpu-benchmark:uv)
**相关文件**: `scripts/bert_tf_vs_onnx.py`
