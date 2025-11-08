# Docker 环境测试指南

本指南说明如何使用 Docker 环境测试 TensorFlow Engine 修复。

## 前提条件

确保已安装 Docker：
```bash
docker --version
```

## 快速测试步骤

### 1. 构建 Docker 镜像

```bash
# 进入项目根目录
cd /path/to/tf_benchmark

# 运行构建脚本（自动检测架构）
bash scripts/build_images.sh
```

构建脚本会：
- 自动检测系统架构（x86_64 或 ARM64）
- 构建 `tf-cpu-benchmark:latest` 镜像
- 安装所有依赖（TensorFlow 2.20.0, transformers, 等）
- 运行基本验证测试

### 2. 测试 TensorFlow Engine 修复

#### 方法 A：使用我们创建的测试脚本

```bash
# 运行纯 TensorFlow/Keras 测试（不需要 transformers）
docker run --rm \
  -v $(pwd):/app \
  tf-cpu-benchmark:latest \
  scripts/test_tf_engine_fix.py
```

#### 方法 B：交互式测试

```bash
# 启动交互式容器
docker run --rm -it \
  -v $(pwd):/app \
  tf-cpu-benchmark:latest \
  /bin/bash

# 在容器内运行测试
python3 scripts/test_tf_engine_fix.py

# 或手动测试
python3 -c "
from src.engines.tensorflow_engine import TensorFlowEngine
import tensorflow as tf
import numpy as np

# 创建简单的 Keras 模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(5,))
])

# 测试 TensorFlowEngine 是否能加载 Keras 模型对象
engine = TensorFlowEngine()
engine.load_model(model)  # 这应该成功！
print('✓ 修复验证成功！')
"
```

#### 方法 C：使用 HuggingFace Transformers 测试（需要下载模型）

```bash
docker run --rm \
  -v $(pwd):/app \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  tf-cpu-benchmark:latest \
  -c "
from transformers import TFBertForSequenceClassification
from src.engines.tensorflow_engine import TensorFlowEngine

# 加载 HuggingFace BERT 模型
model = TFBertForSequenceClassification.from_pretrained(
    'google-bert/bert-base-uncased',
    num_labels=2,
    from_pt=False,
    use_safetensors=False
)

# 测试 TensorFlowEngine 是否能加载 Transformers 模型
engine = TensorFlowEngine()
engine.load_model(model)  # 修复后应该成功！
print('✓ HuggingFace Transformers 模型加载成功！')
print(f'  模型类型: {type(model).__name__}')
"
```

### 3. 运行完整的 BERT 基准测试

```bash
# 快速模式
docker run --rm \
  -v $(pwd)/results:/app/results \
  tf-cpu-benchmark:latest \
  scripts/benchmark_bert_comparison.py \
  --mode quick \
  --output /app/results/bert_test

# 查看结果
ls -la results/bert_test/
```

## 测试验证要点

修复成功的标志：

1. **Keras 模型对象加载**：
   ```
   ✓ Loaded TensorFlow model from object (Sequential)
   ```

2. **HuggingFace Transformers 模型加载**：
   ```
   ✓ Loaded TensorFlow model from object (TFBertForSequenceClassification)
   ```

3. **推理成功**：
   ```
   ✓ Inference successful! Output shape: (1, 2)
   ```

4. **无类型错误**：
   不应再出现：
   ```
   Invalid model_path type: TFBertForSequenceClassification.
   Expected str or tf.keras.Model
   ```

## 修复详情

### 问题
原代码使用 `isinstance(model_path, tf.keras.Model)` 检查，导致 HuggingFace 的 `TFBertForSequenceClassification` 被拒绝。

### 解决方案
改用 `hasattr(model_path, '__call__') and hasattr(model_path, 'predict')` 检查，接受任何可调用的 TensorFlow 模型。

### 修改文件
- `src/engines/tensorflow_engine.py:84-104`

### 提交记录
- 主修复：`894d3ba`
- TODO 更新：`d81946f`

## 故障排除

### Docker 镜像构建失败

```bash
# 清理旧镜像
docker rmi tf-cpu-benchmark:latest

# 重新构建（带详细输出）
docker build -f docker/Dockerfile -t tf-cpu-benchmark:latest . --progress=plain
```

### 容器内缺少依赖

```bash
# 检查已安装的包
docker run --rm tf-cpu-benchmark:latest -c "pip3 list | grep tensorflow"

# 手动安装缺失的包
docker run --rm -it tf-cpu-benchmark:latest /bin/bash
pip3 install <missing-package>
```

### 模型下载慢

```bash
# 使用国内镜像（如有）
export HF_ENDPOINT=https://hf-mirror.com

# 或预先下载模型到本地，然后挂载
docker run --rm \
  -v $(pwd):/app \
  -v /path/to/models:/models \
  tf-cpu-benchmark:latest \
  <test-command>
```

## 性能测试

验证修复不影响性能：

```bash
# 运行基准测试
docker run --rm \
  -v $(pwd)/results:/app/results \
  tf-cpu-benchmark:latest \
  scripts/benchmark_bert_comparison.py \
  --mode standard \
  --output /app/results/performance_test

# 查看性能报告
cat results/performance_test/bert_comparison_report.md
```

## 清理

```bash
# 删除测试结果
rm -rf results/

# 删除 Docker 镜像
docker rmi tf-cpu-benchmark:latest

# 清理 HuggingFace 缓存（可选）
rm -rf ~/.cache/huggingface/
```

## 参考

- **构建脚本**: `scripts/build_images.sh`
- **Dockerfile**: `docker/Dockerfile`
- **测试脚本**: `scripts/test_tf_engine_fix.py`
- **TODO 文档**: `TODO.md` (Issue #1)
