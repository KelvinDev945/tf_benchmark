# TensorFlow Multi-Engine CPU Inference Benchmark

> Comprehensive performance comparison of TensorFlow, TFLite, ONNX Runtime, and OpenVINO on CPU architectures

## ⚠️ 重要说明 / Important Note

**本项目不需要 HuggingFace Transformers 库**

This project does **NOT** require the HuggingFace `transformers` library. The TensorFlow Engine has been designed to support any callable TensorFlow model with a `predict` method, including:
- ✅ Native Keras models (`tf.keras.Sequential`, `tf.keras.Model`)
- ✅ TensorFlow SavedModel format
- ✅ Any custom model implementing `__call__` and `predict`

If you see references to BERT or Transformers models in the codebase, these are for **compatibility testing only** and are not required for the core functionality. The project focuses on benchmarking standard TensorFlow models using multiple inference engines.

## ✨ Features

- 🚀 **Multi-Engine Support**: TensorFlow, TFLite, ONNX Runtime, and OpenVINO
- 🎯 **Real-World Datasets**: ImageNet-1K for image models, GLUE/SST2 for text models
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

**Image Classification:**
- MobileNetV2
- ResNet50
- EfficientNetB0

**Text Understanding:**
- DistilBERT (base-uncased)
- BERT (base-uncased)

## 🚀 Quick Start

### BERT Model Comparison (NEW!)

Compare BERT base model vs quantized vs ONNX in minutes:

```bash
# Quick BERT comparison (5-10 minutes)
python3 scripts/benchmark_bert_comparison.py --mode quick

# Standard BERT comparison (recommended, 20-30 minutes)
python3 scripts/benchmark_bert_comparison.py --mode standard

# View results
cat results/bert_comparison/bert_comparison_report.md
```

See [BERT_BENCHMARK_GUIDE.md](BERT_BENCHMARK_GUIDE.md) for detailed instructions.

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

# Note: OpenVINO is only available on x86_64
# On x86_64, additionally run:
pip install openvino==2023.2.0 openvino-dev==2023.2.0
```

**Run Benchmark:**

```bash
python src/main.py --config configs/benchmark_config.yaml --mode standard
```

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

### BERT Model Benchmarks (NEW!)

**Quick BERT Comparison**:
```bash
# Compare BERT base vs quantized vs ONNX models
python3 scripts/benchmark_bert_comparison.py --mode quick

# Test specific model
python3 scripts/benchmark_bert_comparison.py \
    --model bert-base-uncased \
    --mode standard \
    --batch-size 1

# Test DistilBERT
python3 scripts/benchmark_bert_comparison.py \
    --model distilbert-base-uncased \
    --mode standard
```

**What Gets Tested**:
- ✅ **Base Model**: TensorFlow baseline (no optimizations)
- ✅ **Quantized Models**: TFLite INT8, Float16, Dynamic Range
- ✅ **ONNX Models**: ONNX Runtime with default and optimized settings

**Results Include**:
- Latency comparison (mean, p50, p95, p99)
- Throughput metrics
- Speedup vs baseline
- Model size analysis

See [BERT_BENCHMARK_GUIDE.md](BERT_BENCHMARK_GUIDE.md) for details.

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
1. BERT models (base, quantized, ONNX)
2. Image models (MobileNetV2, ResNet50, EfficientNetB0)
3. Additional text models (RoBERTa)
4. Batch size analysis
5. Consolidated reporting

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

**Q: HuggingFace dataset download is slow**

```bash
# Set HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com
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

### Known Issues with BERT Benchmark

#### Issue 1: TensorFlow Engine - Model Type Check Error

**File**: `src/engines/tensorflow_engine.py:84-102`

**Error**:
```
Invalid model_path type: TFBertForSequenceClassification.
Expected str or tf.keras.Model
```

**Root Cause**: HuggingFace transformers models are not direct instances of `tf.keras.Model`

**Fix**:
```python
# In src/engines/tensorflow_engine.py line 84
# Change:
if isinstance(model_path, tf.keras.Model):

# To:
if hasattr(model_path, '__call__') and hasattr(model_path, 'predict'):
```

#### Issue 2: TFLite INT8 Quantization - Representative Dataset Error

**Error**:
```
TFLite conversion failed: object of type 'function' has no len()
```

**Root Cause**: Representative dataset generator function usage incorrect

**Status**: Investigating calibration data format

#### Issue 3: ONNX Runtime - NumPy Version Compatibility

**Error**:
```
module 'numpy' has no attribute 'object'.
`np.object` was a deprecated alias for the builtin `object`.
```

**Root Cause**: NumPy 1.20+ deprecated `np.object`, incompatible with older tf2onnx versions

**Solution**:
- Update tf2onnx to latest version, or
- Downgrade NumPy to < 1.20 (may affect other packages)

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
