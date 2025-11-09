# TODO - TensorFlow Benchmark 待办事项

**最后更新**: 2025-11-09 (ONNX 转换问题分析完成)

---

## 🔴 高优先级 - 阻塞性问题

### ⚠️ ONNX 转换失败问题 (新发现)

**状态**: ❌ 阻塞中 - 依赖冲突无法直接解决

**核心问题**: tf2onnx 与 TensorFlow 2.20 存在 protobuf 版本冲突

#### 三层嵌套的依赖冲突

```
第1层: NumPy API变更
  └─ NumPy 1.26+ 移除了 np.bool

第2层: tf2onnx版本兼容
  ├─ tf2onnx 1.8.4 (Docker中) → ❌ 不支持 NumPy 1.26
  └─ tf2onnx 1.16.1 (最新)   → ✅ 支持 NumPy 1.26

第3层: Protobuf版本冲突 (根本原因)
  ├─ TensorFlow 2.20   → 要求 protobuf >= 5.28.0
  └─ tf2onnx 1.16.1    → 要求 protobuf < 4.0
      └─ ❌ 无法共存！
```

#### 为什么 Docker 安装了 tf2onnx 1.8.4？

1. ✅ uv 安装 TensorFlow 2.20 → 锁定 protobuf >= 5.28
2. ❌ 尝试 tf2onnx 1.16.1 → 冲突 (需要 protobuf < 4.0)
3. 🔄 回溯尝试旧版本...
4. ✅ tf2onnx 1.8.4 → **唯一兼容 protobuf 5.x 的版本**
5. ⚠️ 但 tf2onnx 1.8.4 使用 np.bool (NumPy 1.26 已移除)
6. ❌ **ONNX 转换失败**

#### 影响

- ❌ 无法使用 tf2onnx 进行 ONNX 转换
- ❌ 无法直接对比 TensorFlow vs ONNX Runtime 性能
- ✅ TensorFlow 测试正常工作 (已获得性能数据)

#### 推荐解决方案

✅ **使用 HuggingFace Optimum** (强烈推荐)
```bash
pip install optimum[onnxruntime]
```

**优点**:
- ✅ 无 protobuf 版本冲突
- ✅ 原生支持 Transformers 模型
- ✅ 自动优化和量化
- ✅ 活跃维护

**待办**:
- [ ] 添加 `optimum[onnxruntime]` 到 requirements.txt
- [ ] 创建 `scripts/bert_optimum_benchmark.py`
- [ ] 运行 TF vs ONNX 完整对比测试
- [ ] 更新 README 和文档

#### 详细分析

参见以下文档获取完整技术分析:
- `ONNX_CONVERSION_ISSUE_ANALYSIS.md` - NumPy 兼容性问题
- `TF2ONNX_VERSION_CONFLICT_EXPLAINED.md` - Protobuf 依赖冲突详解
- `scripts/diagnose_onnx_issue.py` - 自动诊断工具

---

## 🔴 高优先级 - 阻塞性问题 (已解决)

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

**状态**: ✅ 已修复

**修复详情**:
- **提交**: 894d3ba
- **日期**: 2025-11-08
- **修改内容**:
  - 将类型检查从 `isinstance(model_path, tf.keras.Model)` 改为 `hasattr(model_path, '__call__') and hasattr(model_path, 'predict')`
  - 调整检查顺序：先检查字符串路径，再检查模型对象
  - 改进错误消息，显示实际的类型名称
  - 添加模型类型到成功日志以便调试
- **测试脚本**: `scripts/test_tf_engine_fix.py` (纯 TensorFlow/Keras 测试)

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
### 4, 是否需要根据cpu型号rebuild tensorflow来获得最佳性能
因为如下log
I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


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

## 🔧 CI/CD 自动化

### 10. GitHub Actions Workflow 集成

#### 10.1 基础 CI/CD Workflow ✅ 高优先级

**功能**:
- [ ] 自动运行单元测试（pytest）
- [ ] 生成代码覆盖率报告（pytest-cov）
- [ ] 多 Python 版本测试矩阵（3.11, 3.12）
- [ ] 在 PR 和 push 到主分支时自动触发
- [ ] 上传覆盖率报告到 Codecov/Coveralls

**触发条件**:
- Push to main/develop branches
- Pull requests

**预期效果**:
- 每次提交自动验证代码质量
- 防止破坏性更改合并到主分支
- 快速反馈（< 5分钟）

---

#### 10.2 代码质量检查 Workflow ✅ 高优先级

**功能**:
- [ ] Black 代码格式化检查
- [ ] Flake8 语法和风格检查
- [ ] isort import 排序检查
- [ ] mypy 类型注解检查
- [ ] 在 PR 中添加代码质量报告注释

**工具配置**:
- 使用项目中已有的 black、flake8、isort、mypy
- 配置文件：pyproject.toml 或 setup.cfg

**可选功能**:
- [ ] 自动修复格式问题并提交（通过 bot）
- [ ] 代码质量评分和趋势分析

---

#### 10.3 依赖安全检查 Workflow ✅ 中优先级

**功能**:
- [x] 扫描 requirements.txt 中的安全漏洞
- [x] 使用 pip-audit 或 safety 工具
- [x] 定期检查（每周一次 + 每次更新依赖时）
- [x] 发现漏洞时创建 Issue 或发送通知

**检查项目**:
- requirements.txt
- 已知 CVE 数据库
- 过时的包版本

**好处**:
- 及时发现依赖包的安全问题
- 保持项目安全性
- 符合安全最佳实践

**实施详情**:
- **文件**: `.github/workflows/security.yml`
- **工具**: pip-audit, safety
- **触发条件**:
  - Push/PR to main/develop (仅当修改 requirements.txt)
  - 定期执行 (每周一 9:00 UTC)
  - 手动触发 (workflow_dispatch)
- **功能特性**:
  - 使用 pip-audit 和 Safety 双重扫描
  - 检查过时的包版本
  - 生成详细的安全报告
  - 定期扫描时自动创建 Issue (如果发现漏洞)
  - 上传报告到 Artifacts (保留90天)

---

#### 10.4 Docker 镜像构建测试 ✅ 中优先级

**功能**:
- [x] 验证 Dockerfile 能成功构建
- [x] 测试构建的镜像能正常运行
- [x] 多平台构建测试（x86_64, ARM64）
- [x] 可选：推送到 Docker Hub 或 GitHub Container Registry
- [x] 镜像大小和层数优化检查

**相关文件**:
- docker/Dockerfile
- scripts/build_images.sh

**构建策略**:
- PR: 仅验证构建
- Main branch: 构建并推送到 registry
- Tags: 创建发布版本镜像

**实施详情**:
- **文件**: `.github/workflows/docker.yml`
- **触发条件**:
  - Push/PR to main/develop (当修改 docker/, requirements.txt, src/ 时)
  - 标签发布 (v*.*.*)
  - 手动触发 (workflow_dispatch)
- **构建任务**:
  - 多平台构建和测试 (linux/amd64, linux/arm64)
  - 版本检查 (Python, TensorFlow, ONNX Runtime, Transformers)
  - 导入测试 (验证所有关键包可导入)
  - 架构检查 (平台、CPU、内存信息)
  - 应用程序测试 (--help 命令)
  - 镜像大小分析 (显示层信息)
- **推送任务**:
  - 仅在非 PR 事件时执行
  - 推送到 GitHub Container Registry (ghcr.io)
  - 支持多种标签策略 (branch, semver, sha, latest)
- **安全扫描**:
  - 使用 Trivy 扫描镜像漏洞
  - 上传结果到 GitHub Security
  - 生成人类可读的报告

---

#### 10.5 集成测试 Workflow ✅ 低-中优先级

**功能**:
- [x] 运行标记为 `@pytest.mark.integration` 的测试
- [x] 运行标记为 `@pytest.mark.slow` 的长时间测试
- [x] 使用更大的 runner（需要更多 CPU/内存）
- [x] 定期运行（每日/每周）而不是每次 commit
- [x] 可选：在云端运行完整的 benchmark 测试

**测试类型**:
- 数据集加载测试（需要网络下载）
- 模型加载和转换测试
- 端到端 benchmark 运行测试

**运行时间预估**:
- 集成测试：10-30 分钟
- 完整 benchmark：1-6 小时（根据配置）

**实施详情**:
- **文件**: `.github/workflows/integration.yml`
- **触发条件**:
  - 定期执行 (每日 2:00 UTC)
  - Push to main (仅当修改 src/, tests/, requirements.txt)
  - 手动触发 (支持选择测试类型: integration/slow/all)
- **测试任务**:
  - 运行集成测试和慢速测试
  - 快速 benchmark 测试 (50 samples, 10 iterations)
  - 生成覆盖率报告
  - 上传测试结果到 Artifacts
- **通知功能**:
  - 定期运行失败时自动创建 Issue
  - 包含详细的失败信息和运行链接

---

#### 10.6 文档自动生成 ✅ 低优先级

**功能**:
- [x] 自动生成 API 文档（Sphinx 或 MkDocs）
- [x] 部署到 GitHub Pages
- [x] 验证 README 和文档中的示例代码
- [x] 检查文档链接有效性
- [x] 生成 changelog

**文档类型**:
- API 参考文档
- 用户指南
- 开发者文档
- 示例和教程

**实施详情**:
- **文件**: `.github/workflows/docs.yml`
- **触发条件**:
  - Push/PR to main (当修改 src/, docs/, *.md 时)
  - 手动触发
- **验证任务**:
  - Markdown 链接检查 (markdown-link-check)
  - README 代码示例语法验证
  - 文档完整性检查 (README, LICENSE, TODO 等)
  - 文档大小统计
- **生成任务**:
  - 使用 pdoc3 生成 API 文档
  - 创建文档索引页面
  - 上传文档到 Artifacts
- **部署任务**:
  - 自动部署到 GitHub Pages (仅 main 分支)
  - 生成文档网站 URL

---

#### 10.7 性能基准测试和回归检测 ✅ 低优先级

**功能**:
- [x] 定期运行 benchmark 并记录结果
- [x] 比较不同版本的性能变化
- [x] 生成性能趋势图表
- [x] 检测性能退化（超过阈值时告警）
- [x] 将结果发布到 GitHub Pages

**基准测试场景**:
- Quick mode benchmark（每次 PR）
- Standard mode benchmark（每日）
- Full mode benchmark（每周）

**性能指标**:
- 延迟（P50/P95/P99）
- 吞吐量
- 内存使用
- CPU 利用率

**实施详情**:
- **文件**: `.github/workflows/benchmark.yml`
- **触发条件**:
  - 定期执行 (每周日 3:00 UTC)
  - Push to main (当修改 src/benchmark/, src/engines/)
  - 手动触发 (支持选择模式: quick/standard/full 和样本数)
- **基准测试任务**:
  - 自动配置测试参数 (基于模式调整样本数和迭代次数)
  - 运行多引擎性能测试 (TensorFlow, TFLite, ONNX Runtime)
  - 多批次大小测试 (1, 8, 16)
  - 生成性能趋势数据
  - 创建可视化图表 (matplotlib, plotly)
- **历史追踪**:
  - 保存性能历史到 performance_data 分支
  - 与前一次运行比较检测回归
  - 生成性能分析报告
- **发布任务**:
  - 发布结果到 GitHub Pages (定期运行)
  - 上传详细结果到 Artifacts (90天保留)

---

### 实施优先级建议

**第一阶段（本周）**: ✅ 已完成
1. ✅ 基础 CI/CD Workflow（最重要）
2. ✅ 代码质量检查 Workflow

**第二阶段（本月）**: ✅ 已完成
3. ✅ 依赖安全检查
4. ✅ Docker 镜像构建测试

**第三阶段（可选）**: ✅ 已完成
5. ✅ 集成测试 Workflow
6. ✅ 文档自动生成
7. ✅ 性能基准测试

---

### 🎉 CI/CD 实施总结

**全部完成！** 项目现已拥有完整的 CI/CD 自动化流程：

#### 已实施的 7 个 GitHub Actions Workflows:

1. **CI Workflow** (`.github/workflows/ci.yml`)
   - 单元测试自动化
   - 代码覆盖率追踪
   - 多 Python 版本支持 (3.11, 3.12)

2. **Code Quality Workflow** (`.github/workflows/lint.yml`)
   - Black 代码格式检查
   - Flake8 语法检查
   - isort import 排序
   - mypy 类型检查

3. **Security Workflow** (`.github/workflows/security.yml`)
   - pip-audit 和 Safety 依赖漏洞扫描
   - 定期安全审计 (每周)
   - 自动创建安全 Issue

4. **Docker Workflow** (`.github/workflows/docker.yml`)
   - 多平台镜像构建 (amd64, arm64)
   - 完整的镜像测试套件
   - 自动推送到 GitHub Container Registry
   - Trivy 安全扫描

5. **Integration Tests Workflow** (`.github/workflows/integration.yml`)
   - 集成测试和慢速测试
   - 快速 benchmark 测试
   - 定期运行 (每日)
   - 失败自动通知

6. **Documentation Workflow** (`.github/workflows/docs.yml`)
   - API 文档自动生成 (pdoc3)
   - Markdown 链接验证
   - 代码示例验证
   - 自动部署到 GitHub Pages

7. **Performance Benchmark Workflow** (`.github/workflows/benchmark.yml`)
   - 定期性能基准测试 (每周)
   - 性能回归检测
   - 历史趋势追踪
   - 可视化报告生成

#### 主要特性:
- ✅ 自动化测试和质量检查
- ✅ 安全漏洞扫描和监控
- ✅ 多平台 Docker 支持
- ✅ 完整的文档生成和发布
- ✅ 性能监控和回归检测
- ✅ 智能的失败通知机制
- ✅ 详细的运行报告和摘要

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

### 2025-11-09: BERT 脚本修复和 ONNX 问题分析

- [x] **修复 BERT TF 2.20 兼容性问题**
  - 将 `scripts/bert_tf_vs_onnx.py` 修复为使用 SavedModel 方式
  - 避免了 KerasLayer 的 KerasTensor 转换错误
  - 清理旧版备份脚本（原 `scripts/bert_tf_vs_onnx.py.backup`）

- [x] **Docker 环境测试成功**
  - 在 Docker 容器中成功运行 BERT TensorFlow 测试
  - 获得完整性能数据 (延迟: 271.17ms, 吞吐: 3.69 samples/sec)
  - 生成详细的测试报告

- [x] **深度分析 ONNX 转换失败原因**
  - 识别出三层嵌套的依赖冲突
  - 发现 tf2onnx 与 TensorFlow 2.20 的 protobuf 版本冲突
  - 解释了为什么 Docker 安装了旧版 tf2onnx 1.8.4

- [x] **创建完整的问题分析文档**
  - `ONNX_CONVERSION_ISSUE_ANALYSIS.md` - NumPy 兼容性分析
  - `TF2ONNX_VERSION_CONFLICT_EXPLAINED.md` - Protobuf 冲突完整解析
  - `scripts/diagnose_onnx_issue.py` - 自动诊断工具

- [x] **更新项目文档**
  - 更新 README.md 添加 BERT 测试说明
  - 标记 KerasLayer 问题为已修复
  - 添加 Docker 运行示例

### 之前完成

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
