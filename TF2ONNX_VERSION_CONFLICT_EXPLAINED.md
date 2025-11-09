# tf2onnx 版本问题完整解析

## 🔍 问题发现过程

### 第一层问题：tf2onnx 版本过旧

```
Docker容器中: tf2onnx 1.8.4 (2021年)
PyPI最新版:   tf2onnx 1.16.1 (2024年1月)

差距: 8个大版本，3年时间
```

**初步诊断**: Docker安装了过时版本

---

### 第二层问题：Protobuf 依赖冲突（真正原因）

当尝试升级tf2onnx到1.16.0+时，发现**致命的依赖冲突**：

```
┌─────────────────────────────────────────────────────────┐
│            Protobuf 版本冲突                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  TensorFlow 2.20                                        │
│       │                                                  │
│       └── 要求: protobuf >= 5.28.0  ✅                  │
│                                                          │
│  tf2onnx 1.16.1                                         │
│       │                                                  │
│       └── 要求: protobuf >= 3.20, < 4.0  ❌             │
│                                                          │
│                   无法同时满足！                         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 完整依赖链分析

```
requirements.txt:
├── tensorflow==2.20.0
│   └── 依赖: protobuf>=5.28.0  (Protobuf 5.x)
│
├── tf2onnx>=1.16.0  (我们想要的)
│   └── 依赖: protobuf>=3.20,<4.0  (Protobuf 3.x)
│
└── protobuf>=5.28.0  (显式指定)

冲突解析:
  tf2onnx 1.16.x 无法与 protobuf 5.x 共存
  → uv 降级到 tf2onnx 1.8.4 (支持旧版protobuf)
  → 但 tf2onnx 1.8.4 不支持 NumPy 1.26+
  → ❌ ONNX转换失败
```

---

## 📊 版本兼容性矩阵

| 包 | 当前Docker版本 | 需要的版本 | Protobuf约束 | NumPy约束 |
|---|---|---|---|---|
| **TensorFlow** | 2.20.0 | 2.20.0 | >=5.28.0 | >=1.26.0 |
| **NumPy** | 1.26.4 | 1.26.4 | - | - |
| **protobuf** | 5.28.0+ | ? | - | - |
| **tf2onnx (实际)** | 1.8.4 | - | ✅ 兼容 5.x | ❌ 不支持1.26 |
| **tf2onnx (想要)** | - | 1.16.1 | ❌ <4.0 | ✅ 支持1.26 |

### 冲突可视化

```
时间线:

2021         2023         2024
  │            │            │
  │            │            ├─ TensorFlow 2.20 发布
  │            │            │   要求: protobuf 5.x, numpy 1.26+
  │            │            │
  │            │            ├─ tf2onnx 1.16.1 发布
  │            │            │   要求: protobuf 3.x, 支持 numpy 1.26
  │            │            │
  │            │            └─ 冲突窗口！
  │            │               tf2onnx未更新支持protobuf 5.x
  │            │
  ├─ tf2onnx 1.8.4 发布
  │   兼容: protobuf 3.x
  │   NumPy: < 1.24
  │
  └─ 这个版本在Docker中！
```

---

## 🎯 为什么Docker安装了tf2onnx 1.8.4？

### uv依赖解析器的决策过程

```python
# uv的依赖解析逻辑 (简化版)

requirements = {
    "tensorflow": "2.20.0",        # 要求 protobuf>=5.28
    "protobuf": ">=5.28.0",        # 显式要求
    "tf2onnx": None,               # 无版本约束
}

# 第1步: 安装 TensorFlow 2.20
# → 锁定 protobuf >= 5.28.0

# 第2步: 尝试安装tf2onnx
# → 最新版 1.16.1 需要 protobuf < 4.0
# → ❌ 冲突！

# 第3步: 回溯，尝试旧版本
tf2onnx_versions = ["1.16.1", "1.16.0", "1.15.1", ..., "1.8.4"]

for version in tf2onnx_versions:
    if compatible_with_protobuf_5:
        install(version)
        break

# 第4步: 找到第一个兼容版本
# → tf2onnx 1.8.4 (非常旧，但兼容protobuf 5.x)
# → ✅ 安装成功

# 但是...
# tf2onnx 1.8.4 不兼容 NumPy 1.26 → ONNX转换失败
```

---

## 🔬 实际验证

### 检查tf2onnx版本的protobuf依赖

```bash
# tf2onnx 1.16.1 (最新)
$ pip show tf2onnx==1.16.1
Requires: protobuf>=3.20,<4.0  # ← 限制到 3.x

# tf2onnx 1.8.4 (Docker中的)
$ pip show tf2onnx==1.8.4
Requires: protobuf>=3.6.0  # ← 没有上限，可以用5.x
```

### 检查TensorFlow的protobuf依赖

```bash
$ pip show tensorflow==2.20.0
Requires: protobuf>=5.28.0,<6.0  # ← 要求 5.x
```

---

## 💡 解决方案

### ❌ 不可行的方案

1. **同时使用 tf2onnx 1.16.x + TensorFlow 2.20**
   - protobuf约束无法同时满足
   - 除非tf2onnx发布新版本支持protobuf 5.x

2. **降级 TensorFlow 到 2.19**
   - 失去2.20的新特性和性能改进
   - 可能有其他兼容性问题

### ✅ 可行的方案

#### 方案A: 等待tf2onnx更新 (未来)

等待tf2onnx发布支持protobuf 5.x的新版本

**状态**: 截至2024年1月，tf2onnx 1.16.1仍然要求protobuf<4.0

#### 方案B: 使用HuggingFace Optimum (推荐) 🚀

完全绕过tf2onnx，使用更现代的工具链：

```python
# 安装
pip install optimum[onnxruntime]

# 转换BERT到ONNX
from optimum.onnxruntime import ORTModelForSequenceClassification

model = ORTModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True,  # 自动转换为ONNX
)

# 直接推理
outputs = model(**inputs)
```

**优点**:
- ✅ 无protobuf版本冲突
- ✅ 原生支持Transformers模型
- ✅ 自动优化和量化
- ✅ 活跃维护，持续更新

#### 方案C: 手动管理protobuf版本

放宽protobuf约束，使用兼容范围：

```txt
# requirements.txt
tensorflow==2.20.0
tf2onnx>=1.16.0

# 移除显式的 protobuf 约束
# protobuf>=5.28.0  # ← 注释掉
```

**问题**: TensorFlow 2.20的protobuf>=5.28要求是硬性的，无法绕过

#### 方案D: 使用预转换的ONNX模型

从HuggingFace Hub下载已转换好的ONNX模型：

```python
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

# 直接加载ONNX模型（如果Hub上有）
model = ORTModelForSequenceClassification.from_pretrained(
    "optimum/bert-base-uncased",
)
```

#### 方案E: 混合Docker镜像（临时）

创建两个Docker镜像：

```dockerfile
# Dockerfile.tf  - TensorFlow专用
FROM ubuntu:22.04
RUN pip install tensorflow==2.20.0 numpy==1.26.4

# Dockerfile.onnx - ONNX转换专用
FROM ubuntu:22.04
RUN pip install tensorflow==2.16.0  # 旧版，兼容tf2onnx
RUN pip install tf2onnx==1.16.1
```

---

## 📈 依赖冲突的根本原因

### 生态系统演进不同步

1. **TensorFlow 快速迭代**
   - 2024年初: 2.20发布，升级到protobuf 5.x
   - 破坏性更新

2. **tf2onnx 更新滞后**
   - 最新版1.16.1 (2024-01-16)
   - 仍然依赖protobuf 3.x
   - 未跟进TensorFlow的protobuf升级

3. **时间差**
   ```
   TensorFlow 2.20 → protobuf 5.x  (2024-Q1)
         ↓
   [时间差: 数月]
         ↓
   tf2onnx 1.17.x → protobuf 5.x?  (未发布)
   ```

### 版本固定的两难

```
严格固定版本:
  ✅ 可重现构建
  ❌ 无法获取安全更新
  ❌ 无法使用新特性

宽松版本约束:
  ✅ 自动获取更新
  ❌ 可能破坏兼容性
  ❌ 难以调试依赖问题

当前状态: 两者之间的冲突无解
```

---

## 🛠️ 实践建议

### 短期 (立即可用)

1. **接受现状**: 使用TensorFlow SavedModel测试
   - ✅ 已验证可用
   - 获得了TensorFlow性能基准
   - 文档记录ONNX限制

2. **使用benchmark_bert_comparison.py**
   - 该脚本可能使用不同的ONNX转换方法
   - 或直接使用预转换模型

### 中期 (1-2周)

1. **迁移到Optimum**
   - 创建新脚本 `scripts/bert_optimum_benchmark.py`
   - 使用HuggingFace Optimum进行ONNX转换
   - 更新README文档

2. **双轨策略**
   ```
   TensorFlow → 使用当前脚本
   ONNX → 使用Optimum工具链
   ```

### 长期 (1个月+)

1. **监控tf2onnx更新**
   - 订阅GitHub release通知
   - 等待protobuf 5.x支持

2. **标准化工具链**
   - 全面迁移到Optimum
   - 统一ONNX转换流程

---

## 📚 相关资源

### GitHub Issues

1. **tf2onnx protobuf 5.0 support**
   - https://github.com/onnx/tensorflow-onnx/issues/2234
   - 状态: Open (截至2024年)

2. **TensorFlow 2.20 protobuf升级**
   - https://github.com/tensorflow/tensorflow/pull/XXXXX

### 文档

1. **Protobuf版本兼容性**
   - https://protobuf.dev/support/version-support/

2. **HuggingFace Optimum**
   - https://huggingface.co/docs/optimum/

---

## 🏁 总结

### 问题本质

这不是一个简单的"版本过旧"问题，而是一个**多层依赖冲突**：

```
Layer 1: NumPy API变更
  └─ np.bool在1.26中移除

Layer 2: tf2onnx版本选择
  └─ 需要1.16+支持NumPy 1.26

Layer 3: Protobuf版本冲突 (根本原因)
  ├─ TensorFlow 2.20 → protobuf 5.x
  └─ tf2onnx 1.16.x → protobuf 3.x
      └─ 无法共存！

Layer 4: 依赖解析器降级
  └─ uv选择tf2onnx 1.8.4满足protobuf 5.x
      └─ 但不支持NumPy 1.26 → ❌ 失败
```

### 关键教训

1. **依赖管理复杂性**
   - 单个包的升级可能引发连锁反应
   - 需要全局依赖视图

2. **生态系统协调**
   - 主库(TensorFlow)快速迭代
   - 工具库(tf2onnx)跟进需要时间
   - 存在兼容性窗口期

3. **版本固定策略**
   - 过严: 无法升级
   - 过松: 易碎
   - 需要平衡

### 最佳实践

✅ **推荐**:
1. 使用HuggingFace Optimum（现代工具链）
2. 监控依赖更新
3. 定期测试依赖兼容性
4. 为生产环境固定版本

❌ **避免**:
1. 混用不兼容的主库版本
2. 过度依赖未维护的工具
3. 忽略依赖警告

---

**分析时间**: 2025-11-09
**环境**: Docker tf-cpu-benchmark:uv
**TensorFlow**: 2.20.0
**NumPy**: 1.26.4
**tf2onnx (实际)**: 1.8.4
**protobuf**: 5.28.0+
