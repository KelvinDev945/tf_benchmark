# TensorFlow Multi-Engine CPU Inference Benchmark

> Comprehensive performance comparison of TensorFlow, TFLite, ONNX Runtime, and OpenVINO on CPU architectures

## ⚠️ 重要说明 / Important Note

**项目核心仍以图像模型为主，但已通过 TensorFlow Hub 恢复 BERT 文本模型的可选支持。**

The benchmark focuses on image workloads by default, while optional BERT text pipelines are available without requiring HuggingFace `transformers` / `datasets`. The TensorFlow engine continues to support:
- ✅ Native Keras models (`tf.keras.Sequential`, `tf.keras.Model`)
- ✅ TensorFlow SavedModel format
- ✅ TensorFlow Hub BERT encoders + 轻量级文本数据管线

## ✨ Features

- 🚀 **Multi-Engine Support**: TensorFlow, TFLite, ONNX Runtime, and OpenVINO
- 🎯 **Real-World Datasets**: ImageNet-1K、CIFAR-10/100 图像数据集 + 内置轻量级文本样本
- ⚡ **Multiple Optimizations**: XLA JIT, mixed precision, quantization (INT8, FP16)
- 📊 **Comprehensive Metrics**: Latency (P50/P95/P99), throughput, CPU/memory usage
- 🐳 **Docker Support**: Containerized execution for reproducibility
- 🔧 **Multi-Architecture**: x86_64 and ARM64 support
- 📈 **Rich Reporting**: Automated HTML reports with 10+ visualizations

## 📋 Supported Engines and Models

| Engine | Configurations | x86_64 | ARM64 |
|--------|----------------|--------|-------|
| **TensorFlow** | baseline, xla, threads, mixed_precision, best_combo | ✅ | ✅ |
| **TFLite** | float32, dynamic_range, int8, float16 | ✅ | ✅ |
| **ONNX Runtime** | default, optimized, quantized | ✅ | ✅ |
| **OpenVINO** | fp32, fp16, int8, dynamic | ✅ | ❌ |

### Supported Models

**Image Classification (默认支持):**
- MobileNetV2
- ResNet50
- EfficientNetB0
- InceptionV3
- VGG16
**Text Understanding (可选，基于 TensorFlow Hub):**
- BERT Base (uncased)
- 自带轻量级 `TextDatasetLoader`（无需 HuggingFace 数据集）

### BERT Demo (TensorFlow Hub)

```bash
# 挂载本地缓存目录，避免每次重新下载 TF Hub 模型
docker run --rm \
    -v ~/.cache/tfhub:/root/.cache/tfhub \
    -v $(pwd):/workspace -w /workspace \
    --entrypoint python3 tf-cpu-benchmark:uv \
    scripts/demo_bert_tf_only.py
```

> 说明：脚本与核心代码会优先从 `~/.cache/tfhub`（可通过 `TFHUB_CACHE_DIR` 覆盖）读取模型；若缓存缺失会自动下载并写回该目录。
>
> 运行完成后，基准结果与 Markdown 报告会写入 `results/bert_tf_demo/`。首次执行需要联网下载约 430 MB 的 TF Hub 模块；后续运行只要挂载相同缓存目录即可复用。

如需修改批大小、序列长度或迭代次数，可编辑 `scripts/demo_bert_tf_only.py` 顶部的 `BATCH_SIZE`、`SEQ_LENGTH`、`NUM_WARMUP` 与 `NUM_TEST` 配置；`src/models.ModelLoader.load_text_model()` 同样复用上述缓存目录，可直接在自定义流程中加载 BERT 分类器。

## 🚀 Quick Start

### Full Benchmark Suite

```bash
# 1. Build Docker image (optional)
./scripts/build_images.sh

# 2. Run comprehensive benchmark (all models + all engines)
./scripts/run_full_benchmark.sh standard

# 3. View consolidated results
cat results/full_benchmark_*/consolidated_report/consolidated_report.md
```

### Docker Quick Start

```bash
# 1. Build Docker image
./scripts/build_images.sh

# 2. Run benchmark
docker run --rm -v $(pwd)/results:/app/results \
    tf-cpu-benchmark:latest \
    src/main.py --mode quick

# 3. View results
open results/latest/report/report.html
```

### ⚡ Docker with uv (Optimized - 2-3x Faster Build)

**NEW**: Docker image optimized with [uv](https://github.com/astral-sh/uv) package manager for ultra-fast builds!

```bash
# Build optimized Docker image with uv
docker build -t tf-cpu-benchmark:uv -f docker/Dockerfile .

# Run quick environment test
docker run --rm -v $(pwd)/results:/app/results -v $(pwd)/scripts:/app/scripts \
    tf-cpu-benchmark:uv scripts/test_docker_env.py

# View test results
cat results/docker_uv_test/mobilenet_v2_results.json
```

**Performance**:
- Build time: ~1-2 minutes (vs 3-5 minutes with pip) - **2-3x faster** ⚡
- Package installation: ~25 seconds (vs 60-120 seconds) - **up to 5x faster**
- MobileNetV2 inference: 85.8ms latency, 11.66 samples/sec

See [DOCKER_UV_TEST_RESULTS.md](DOCKER_UV_TEST_RESULTS.md) for detailed benchmarks.

## 📦 Installation

### Option 1: Docker (Recommended)

**Prerequisites:**
- Docker 20.10+
- 8GB+ RAM
- 20GB+ disk space

**Build and Run:**

```bash
# Clone repository
git clone https://github.com/yourusername/tf-cpu-benchmark.git
cd tf-cpu-benchmark

# Build Docker image
./scripts/build_images.sh

# Run standard benchmark
docker run --rm \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/configs:/app/configs \
    tf-cpu-benchmark:latest \
    src/main.py --mode standard
```

### Option 2: Local Installation

**Prerequisites:**
- Python 3.11+
- pip 23.0+
- 16GB+ RAM (for full benchmark)

**Install Dependencies:**

```bash
# Clone repository
git clone https://github.com/yourusername/tf-cpu-benchmark.git
cd tf-cpu-benchmark

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install git hooks for code style checks
pip install pre-commit
pre-commit install

# Note: OpenVINO is only available on x86_64
# On x86_64, additionally run:
pip install openvino==2023.2.0 openvino-dev==2023.2.0
```

**Run Benchmark:**

```bash
python src/main.py --config configs/benchmark_config.yaml --mode standard
```

### Option 3: CPU-Optimized TensorFlow (Best Performance)

**⚡ 获得2-4倍性能提升！**

The default TensorFlow pip package is a generic build. For optimal CPU performance, compile TensorFlow from source with CPU-specific optimizations.

**Quick Start - Intel Optimized TensorFlow:**

```bash
# Easiest option: Use Intel's pre-optimized build
pip uninstall tensorflow
pip install intel-tensorflow==2.20.0

# Test performance improvement
python3 scripts/benchmark_xla_mixed_precision.py --model-type mobilenet_v2 --num-runs 30
```

**Expected Performance Gains:**
- MobileNetV2 inference: 1.8-3.8x faster
- ResNet50 inference: 1.5-3.0x faster
- Matrix operations: 2.0-4.0x faster (with AVX512)

**Build from Source (Maximum Performance):**

For maximum performance, compile TensorFlow with your CPU's specific instruction sets (AVX2, AVX512, FMA):

```bash
# See detailed guide
cat TENSORFLOW_CPU_OPTIMIZATION.md

# Quick example for AVX512 CPUs:
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
./configure
bazel build --config=opt \
    --config=mkl \
    --copt=-march=native \
    --copt=-mavx512f \
    //tensorflow/tools/pip_package:build_pip_package
```

📖 **Full Guide**: See [TENSORFLOW_CPU_OPTIMIZATION.md](TENSORFLOW_CPU_OPTIMIZATION.md) for:
- CPU instruction set detection
- Platform-specific build commands (Intel/AMD/ARM)
- Docker-based compilation
- Performance benchmarking before/after
- Troubleshooting

**When to Use CPU-Optimized TensorFlow:**
- ✅ Production deployments requiring maximum CPU performance
- ✅ Training workloads on CPU servers
- ✅ When you control the deployment hardware
- ❌ Cross-platform distribution (use generic build)
- ❌ Quick prototyping (use ONNX Runtime instead)

## 🔧 Configuration

The benchmark is configured via `configs/benchmark_config.yaml`. Key sections:

### Benchmark Parameters

```yaml
benchmark:
  warmup_iterations: 50      # Number of warmup runs
  test_iterations: 200       # Number of test runs
  repeat_runs: 5             # Repeat entire benchmark N times
  confidence_level: 0.95     # Statistical confidence level
```

### Testing Modes

Three predefined modes for different use cases:

| Mode | Warmup | Test Iterations | Repeat Runs | Use Case |
|------|--------|-----------------|-------------|----------|
| **quick** | 10 | 50 | 1 | Fast verification during development |
| **standard** | 50 | 200 | 3 | Balanced testing (default) |
| **full** | 100 | 500 | 5 | Comprehensive testing for publication |

### Engine Configuration

Each engine can be enabled/disabled and configured:

```yaml
engines:
  tensorflow:
    enabled: true
    configs:
      - name: baseline
        xla: false
        mixed_precision: false
      - name: xla
        xla: true
      # ... more configs
```

### Dataset Configuration

```yaml
dataset:
  image:
    name: "imagenet-1k"
    split: "validation"
    num_samples: 5000
  text:
    name: "glue"
    subset: "sst2"
    num_samples: 1000
```

## 🎯 Usage

### Full Benchmark Suite

**Run All Models + All Engines**:
```bash
# Quick verification (30 minutes)
./scripts/run_full_benchmark.sh quick

# Standard benchmark (2-4 hours, recommended)
./scripts/run_full_benchmark.sh standard

# Full comprehensive benchmark (6-10 hours)
./scripts/run_full_benchmark.sh full
```

**What Gets Tested**:
1. Image models（MobileNetV2、ResNet50、EfficientNetB0 等）
2. 多引擎配置（TensorFlow / TFLite / ONNX Runtime / OpenVINO）
3. 批量大小与量化策略分析
4. 统一报告生成（HTML / Markdown / 图表）

### Basic Usage

```bash
# Quick test (5 minutes)
python src/main.py --mode quick

# Standard benchmark (1-2 hours)
python src/main.py --mode standard

# Full benchmark (4-6 hours)
python src/main.py --mode full
```

### Advanced Options

```bash
# Specify custom config
python src/main.py --config configs/custom_config.yaml

# Test specific engines only
python src/main.py --engines tensorflow,tflite

# Test specific models only
python src/main.py --models mobilenet_v2,resnet50

# Custom output directory
python src/main.py --output ./my_results

# Resume from checkpoint
python src/main.py --resume

# Skip report generation
python src/main.py --no-report
```

### Using Docker

```bash
# Quick test
docker run --rm -v $(pwd)/results:/app/results \
    tf-cpu-benchmark:latest src/main.py --mode quick

# Custom configuration
docker run --rm \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/configs:/app/configs \
    tf-cpu-benchmark:latest \
    src/main.py --config /app/configs/custom_config.yaml

# OpenVINO only (x86_64)
docker run --rm -v $(pwd)/results:/app/results \
    tf-cpu-benchmark-openvino:latest \
    src/main.py --engines openvino --mode standard
```

## 📊 Output and Reports

### Directory Structure

After running a benchmark, results are organized as follows:

```
results/
└── 20250106_143022/           # Timestamp
    ├── results.json           # Raw results (JSON)
    ├── results.csv            # Results table (CSV)
    ├── system_info.json       # System information
    ├── benchmark.log          # Detailed logs
    ├── checkpoint.json        # Resume checkpoint
    └── report/
        ├── report.html        # Interactive HTML report
        ├── report.md          # Markdown report
        ├── recommendations.txt # Best configurations
        └── plots/
            ├── throughput_comparison.png
            ├── latency_boxplot.png
            ├── batch_size_analysis.png
            └── ... (10+ charts)
```

### Generated Reports

**HTML Report** includes:
- Executive summary with key findings
- System configuration details
- Image model performance comparison
- Text model performance comparison
- Engine comparison analysis
- Quantization analysis
- Recommended configurations by scenario
- Interactive visualizations

**Visualizations** (10+ charts):
- Throughput comparison bar charts
- Latency distribution box plots
- Batch size scaling curves
- Sequence length impact (text models)
- Speedup radar charts
- Resource efficiency scatter plots
- Quantization tradeoff analysis
- Model size comparison
- Confidence interval error bars
- Comprehensive ranking heatmap

## 📈 Example Results

### Expected Performance (Intel i9-13900K, x86_64)

| Engine | Model | Batch Size | Throughput (samples/sec) | Latency P50 (ms) |
|--------|-------|------------|--------------------------|------------------|
| TensorFlow (XLA) | MobileNetV2 | 32 | ~1500 | ~21 |
| TFLite (INT8) | MobileNetV2 | 32 | ~2000 | ~16 |
| ONNX (optimized) | MobileNetV2 | 32 | ~1800 | ~18 |
| OpenVINO (INT8) | MobileNetV2 | 32 | ~2500 | ~13 |

*Actual performance varies by hardware*

## 🏗️ Project Structure

```
tf-cpu-benchmark/
├── docker/                    # Docker configurations
│   ├── Dockerfile
│   └── Dockerfile.openvino
├── src/                       # Source code
│   ├── config/               # Configuration management
│   ├── dataset/              # Dataset loaders
│   ├── models/               # Model loaders and converters
│   ├── engines/              # Inference engines
│   ├── benchmark/            # Benchmark runner and metrics
│   └── reporting/            # Report generation
├── configs/                   # Configuration files
│   └── benchmark_config.yaml
├── scripts/                   # Automation scripts
│   ├── build_images.sh
│   ├── run_benchmark.sh
│   └── generate_report.py
├── tests/                     # Unit tests
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🧪 Testing

Run unit tests:

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_config.py -v
```

## 🔍 Troubleshooting

### Common Issues

**Q: Docker build fails with network timeout**

```bash
# Use pip mirror
docker build --build-arg PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple .
```

**Q: OpenVINO not working on ARM64**

This is expected. OpenVINO only supports x86_64 architecture.

**Q: Out of memory error**

```bash
# Reduce batch sizes or samples in config
# Edit configs/benchmark_config.yaml:
batch_sizes: [1, 4, 8]  # Instead of [1, 4, 8, 16, 32]
```

**Q: Permission denied on scripts**

```bash
chmod +x scripts/*.sh
```

### Testing Environment

- **TensorFlow**: 2.20.0
- **Python**: 3.11
- **Docker Image**: tf-cpu-benchmark:latest
- **Models**: google-bert/bert-base-uncased
- **Dataset**: glue/sst2 (validation split)

## 🛣️ Roadmap

- [ ] Phase 1: Infrastructure ✅ (Current)
- [ ] Phase 2: Data and Model Loaders (Day 3-4)
- [ ] Phase 3: Inference Engines (Day 5-7)
- [ ] Phase 4: Benchmark Core (Day 8-10)
- [ ] Phase 5: Reporting and Visualization (Day 11-13)
- [ ] Phase 6: Documentation and Automation (Day 14-15)

See [plan.md](plan.md) for detailed implementation plan.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/tf-cpu-benchmark.git
cd tf-cpu-benchmark

# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [TensorFlow](https://www.tensorflow.org/) team for the excellent ML framework
- [ONNX Runtime](https://onnxruntime.ai/) team for the cross-platform inference engine
- [OpenVINO](https://docs.openvino.ai/) team for the optimized inference toolkit
- [HuggingFace](https://huggingface.co/) for datasets and transformers
- MLPerf for benchmark methodology inspiration

## 📧 Contact

For questions or feedback:
- Open an issue on [GitHub](https://github.com/yourusername/tf-cpu-benchmark/issues)
- Email: your.email@example.com

---

**Project Status**: Phase 1 - Infrastructure Setup ✅

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)
