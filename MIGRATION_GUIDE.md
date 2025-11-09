# 文本模型迁移指南 / Text Model Migration Guide

> 更新日期：2025-11-09  
> 适用版本：`chore/remove-transformers` 分支及之后

本指南说明在项目移除 HuggingFace `transformers` 与 `datasets` 依赖后，如何在需要文本/NLP 功能时进行迁移或扩展。当前主分支默认提供基于 TensorFlow Hub 的 BERT 分类模型支持；若需更复杂的 NLP 流程，可按以下方案延伸。

> 📦 **缓存提示**：核心代码与相关脚本将 TensorFlow Hub 模型缓存在 `~/.cache/tfhub`（可通过环境变量 `TFHUB_CACHE_DIR` 修改）。在 Docker 场景下，建议使用 `-v ~/.cache/tfhub:/root/.cache/tfhub` 挂载主机缓存目录，避免重复下载。

---

## 1. 推荐方案：使用 TensorFlow Hub

```python
import tensorflow_hub as hub
import tensorflow as tf

bert_layer = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
    trainable=False,
)

inputs = {
    "input_word_ids": tf.keras.layers.Input(shape=(128,), dtype=tf.int32),
    "input_mask": tf.keras.layers.Input(shape=(128,), dtype=tf.int32),
    "input_type_ids": tf.keras.layers.Input(shape=(128,), dtype=tf.int32),
}
pooled_output, sequence_output = bert_layer(
    [
        inputs["input_word_ids"],
        inputs["input_mask"],
        inputs["input_type_ids"],
    ]
)
```

**优势**
- 无需引入 `transformers` 包，依赖更精简。
- 模型以 TensorFlow 原生格式提供，兼容 SavedModel、TFLite、ONNX 转换流程。

**注意事项**
- 可选模型数量相对有限。
- 需自定义 tokenizer，可考虑 `keras-nlp` 或基于 `tensorflow_text` 的实现。

---

## 2. 方案二：预转换 HuggingFace 模型为 SavedModel

若仍需使用 HuggingFace 权重，可在单独环境中一次性完成转换，再将结果拷贝到本项目中。

```bash
python - <<'PY'
from transformers import TFBertModel

model = TFBertModel.from_pretrained("bert-base-uncased")
model.save("artifacts/text/bert_base_savedmodel")
PY
```

运行完毕后，将 `artifacts/text/bert_base_savedmodel` 目录复制到本项目（建议置于 `models/` 或 `artifacts/` 子目录），随后可通过：

```python
import tensorflow as tf

model = tf.saved_model.load("models/bert_base_savedmodel")
```

**优势**
- 运行时无须安装 `transformers`。
- 保留完整 HuggingFace 模型生态。

**注意事项**
- 转换过程依赖 HuggingFace 环境，需确保网络可访问模型仓库。
- SavedModel 体积较大，应合理规划存储。

---

## 3. 方案三：将 `transformers` 作为可选依赖

如果团队仍需保留原有代码结构，可将文本相关功能拆分为独立模块或插件，并在运行时检测依赖。

示例（伪代码）：

```python
try:
    from transformers import TFAutoModelForSequenceClassification
except ImportError as exc:
    raise RuntimeError(
        "Text model support is disabled. "
        "Install transformers>=4.35.0 to re-enable."
    ) from exc
```

**操作建议**
1. 在独立的 `extras[\"text\"]` 或 `requirements-text.txt` 中声明依赖。
2. 在 CI/CD 中单独运行文本管线，避免影响主流程。
3. 对最终用户明确标注“可选功能，需要额外安装”。

---

## 4. 推荐的项目结构调整

- `src/text/`：可选，放置自定义 tokenizer、文本数据处理代码。
- `artifacts/text/`：存放预转换的 SavedModel 或量化产物。
- `scripts/text/`：独立的文本 benchmark、转换脚本。
- `docs/text/`：面向 NLP 工作流的补充文档。

通过模块化方式，既能保持主仓库轻量化，又可按需扩展文本能力。

---

## 5. 环境管理建议

| 场景 | 建议环境 | 说明 |
|------|----------|------|
| 仅运行通用图像 benchmark | 默认 `requirements.txt` | 无 HuggingFace 依赖，安装最快 |
| 需要转换/调试文本模型 | 单独虚拟环境（含 `transformers`） | 与主项目隔离，避免包冲突 |
| CI 运行 NLP 扩展测试 | 可选 job，显式安装 `requirements-text.txt` | 避免拖慢主 CI |

---

## 6. 常见问题

**Q: 能否继续使用原来的文本数据集加载器？**  
A: `TextDatasetLoader` 已移除，如需复用，可在独立模块中重建并引入 `datasets` 包。

**Q: 如何保持结果可复现？**  
A: 将转换后的 SavedModel 与校验脚本一同归档，并在 README 或文档中说明版本信息。

**Q: 是否支持混合基准测试（图像 + 文本）？**  
A: 主仓库默认仅跑图像模型。若需混合场景，建议在分支或插件中扩展，避免影响核心用户。

---

如需进一步支持或示例，请联系维护团队或在 issue 中描述使用场景，我们会协助评估最合适的方案。

