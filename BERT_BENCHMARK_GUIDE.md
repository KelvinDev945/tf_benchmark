# BERT Model Inference Benchmark Guide

This guide explains how to run comprehensive BERT model inference benchmarks, comparing:
- **Base Model**: TensorFlow baseline implementation
- **Quantized Models**: TFLite INT8, Float16, Dynamic Range quantization
- **ONNX Models**: ONNX Runtime with different optimization levels

## Table of Contents

- [Quick Start](#quick-start)
- [BERT-Specific Benchmark](#bert-specific-benchmark)
- [Full Benchmark Suite](#full-benchmark-suite)
- [Understanding Results](#understanding-results)
- [Configuration Options](#configuration-options)

---

## Quick Start

### 1. BERT Model Comparison (Fastest)

Run a quick comparison of BERT base vs quantized vs ONNX:

```bash
# Quick test (5-10 minutes)
python3 scripts/benchmark_bert_comparison.py --mode quick

# Standard test (20-30 minutes, recommended)
python3 scripts/benchmark_bert_comparison.py --mode standard

# Full test (1-2 hours)
python3 scripts/benchmark_bert_comparison.py --mode full
```

**Output Location**: `./results/bert_comparison/`

**Files Generated**:
- `bert_comparison_results.json` - Raw benchmark data
- `bert_comparison_report.md` - Formatted comparison report

---

## BERT-Specific Benchmark

### Command Line Options

```bash
python3 scripts/benchmark_bert_comparison.py [OPTIONS]
```

**Available Options**:

| Option | Description | Default | Choices |
|--------|-------------|---------|---------|
| `--model` | BERT model to test | `bert-base-uncased` | `bert-base-uncased`, `distilbert-base-uncased` |
| `--mode` | Testing mode | `quick` | `quick`, `standard`, `full` |
| `--batch-size` | Batch size for inference | `1` | Any positive integer |
| `--output` | Output directory | `./results/bert_comparison` | Any valid path |

### Examples

**Test BERT base-uncased**:
```bash
python3 scripts/benchmark_bert_comparison.py \
    --model bert-base-uncased \
    --mode standard \
    --batch-size 1
```

**Test DistilBERT (faster)**:
```bash
python3 scripts/benchmark_bert_comparison.py \
    --model distilbert-base-uncased \
    --mode quick \
    --batch-size 4
```

**Custom output directory**:
```bash
python3 scripts/benchmark_bert_comparison.py \
    --model bert-base-uncased \
    --mode standard \
    --output ./my_results/bert_test
```

---

## Full Benchmark Suite

Run comprehensive benchmarks across **all models** and **all engines**:

```bash
# Quick verification (30 minutes)
./scripts/run_full_benchmark.sh quick

# Standard benchmark (2-4 hours, recommended)
./scripts/run_full_benchmark.sh standard

# Full comprehensive benchmark (6-10 hours)
./scripts/run_full_benchmark.sh full
```

**What Gets Tested**:

1. **BERT Models** (Phase 1):
   - BERT base-uncased: TensorFlow baseline, TFLite INT8/Float16, ONNX
   - DistilBERT: TensorFlow baseline, TFLite INT8/Float16, ONNX

2. **Image Models** (Phase 2):
   - MobileNetV2, ResNet50, EfficientNetB0
   - Across TensorFlow, TFLite, ONNX, OpenVINO (x86_64 only)

3. **Additional Text Models** (Phase 3):
   - RoBERTa
   - Across TensorFlow, TFLite, ONNX

4. **Batch Size Analysis** (Phase 4):
   - MobileNetV2 with batch sizes: 1, 4, 8, 16, 32

5. **Consolidated Report** (Phase 5):
   - Aggregates all results
   - Generates comparison tables
   - Provides recommendations

**Output Location**: `./results/full_benchmark_<timestamp>/`

**Directory Structure**:
```
results/full_benchmark_20250108_143022/
├── bert_comparison/
│   ├── bert_base/
│   │   ├── bert_comparison_results.json
│   │   └── bert_comparison_report.md
│   └── distilbert/
│       ├── bert_comparison_results.json
│       └── bert_comparison_report.md
├── image_models/
│   ├── mobilenet_v2_tensorflow/
│   ├── mobilenet_v2_tflite/
│   ├── mobilenet_v2_onnx/
│   └── ...
├── text_models/
│   └── roberta_*/
├── batch_analysis/
│   └── mobilenet_bs*/
├── consolidated_report/
│   ├── consolidated_report.md
│   ├── all_results.csv
│   └── bert_comparison.csv
└── benchmark.log
```

---

## Understanding Results

### BERT Comparison Report

The generated report includes:

#### 1. Results Summary Table

Example:
```
| Model | Config | Latency (mean) | Latency (p95) | Throughput |
|-------|--------|----------------|---------------|------------|
| bert-base-uncased | tensorflow_baseline | 45.23 ms | 48.67 ms | 22.1 samples/sec |
| bert-base-uncased | tflite_int8 | 28.45 ms | 30.12 ms | 35.2 samples/sec |
| bert-base-uncased | tflite_float16 | 32.18 ms | 34.56 ms | 31.1 samples/sec |
| bert-base-uncased | onnx_default | 38.92 ms | 41.23 ms | 25.7 samples/sec |
| bert-base-uncased | onnx_optimized | 35.67 ms | 37.89 ms | 28.0 samples/sec |
```

#### 2. Speedup vs Baseline

Example:
```
Speedup Analysis:
- tflite_int8: 1.59x faster
- tflite_float16: 1.41x faster
- onnx_optimized: 1.27x faster
```

#### 3. Best Configuration

```
Fastest Configuration: tflite_int8
- Latency: 28.45 ms
- Throughput: 35.2 samples/sec
```

### Key Metrics Explained

- **Latency (mean)**: Average time per inference
- **Latency (p50/p95/p99)**: Percentile latencies (50th, 95th, 99th)
- **Throughput**: Number of samples processed per second
- **CPU Mean**: Average CPU usage during inference
- **Memory Peak**: Maximum memory usage

---

## Configuration Options

### Testing Modes

| Mode | Warmup | Test Iterations | Repeat Runs | Time Estimate |
|------|--------|-----------------|-------------|---------------|
| `quick` | 5 | 20 | 1 | 5-10 minutes |
| `standard` | 30 | 100 | 3 | 20-30 minutes |
| `full` | 50 | 200 | 5 | 1-2 hours |

### Custom Configuration

默认使用 `configs/benchmark_config.yaml` 中的设置，可按需调整或通过命令行覆盖：

```yaml
# Example: Faster testing
benchmark:
  warmup_iterations: 10
  test_iterations: 50
  repeat_runs: 1

# Example: More batch sizes
batch_sizes: [1, 2, 4, 8, 16]

# Example: Different sequence lengths
sequence_lengths: [64, 128, 256, 512]
```

---

## Advanced Usage

### Generate Consolidated Report Only

If you've already run benchmarks and want to regenerate the report:

```bash
python3 scripts/generate_consolidated_report.py \
    --input-dir ./results/full_benchmark_20250108_143022 \
    --output ./results/new_report
```

### Docker Usage

Run BERT benchmark in Docker:

```bash
# Build image
./scripts/build_images.sh

# Run BERT comparison
docker run --rm -v $(pwd)/results:/app/results \
    tf-cpu-benchmark:latest \
    python3 scripts/benchmark_bert_comparison.py --mode standard

# Run full benchmark
docker run --rm -v $(pwd)/results:/app/results \
    tf-cpu-benchmark:latest \
    ./scripts/run_full_benchmark.sh standard
```

---

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or number of samples

```yaml
# In configs/benchmark_config.yaml
batch_sizes: [1, 4]  # Remove larger batch sizes
dataset:
  text:
    num_samples: 100  # Reduce from 500
```

### Issue: Conversion Fails

**Solution**: Check model compatibility

- TFLite INT8 requires calibration data (automatically handled)
- ONNX conversion requires tf2onnx package (included in requirements.txt)
- Some models may not support all quantization methods

### Issue: Slow Performance

**Solution**: Use quick mode or reduce iterations

```bash
python3 scripts/benchmark_bert_comparison.py --mode quick
```

---

## Results Interpretation

### Choosing the Right Configuration

**For Production (Latency-Critical)**:
- Use configuration with lowest p95/p99 latency
- Usually: TFLite INT8 or ONNX optimized

**For Production (Throughput-Critical)**:
- Use configuration with highest throughput
- Usually: TFLite INT8 with larger batch sizes

**For Development**:
- Use TensorFlow baseline (easiest to debug)

**For Edge Devices**:
- Use TFLite INT8 (smallest model size, fast inference)

**For Server Deployment**:
- Use ONNX optimized (good balance, wide platform support)

### Model Size vs Speed Trade-off

| Model Type | Relative Size | Relative Speed | Accuracy Loss |
|------------|---------------|----------------|---------------|
| TensorFlow Baseline | 100% | 1.0x | 0% |
| TFLite Float16 | ~50% | 1.3-1.5x | <0.1% |
| TFLite Dynamic Range | ~40% | 1.4-1.6x | <0.5% |
| TFLite INT8 | ~25% | 1.5-2.0x | 0.5-2% |
| ONNX Optimized | ~100% | 1.2-1.4x | 0% |

---

## Next Steps

1. **Run Quick Test**:
   ```bash
   python3 scripts/benchmark_bert_comparison.py --mode quick
   ```

2. **Review Results**:
   - Check `./results/bert_comparison/bert_comparison_report.md`

3. **Run Full Benchmark** (optional):
   ```bash
   ./scripts/run_full_benchmark.sh standard
   ```

4. **Deploy Best Configuration**:
   - Use the fastest configuration from results
   - Convert your model using recommended settings
   - Deploy to production

---

## Support

For issues or questions:
- Check `results/*/benchmark.log` for detailed logs
- Review error messages in console output
- Ensure all dependencies are installed: `pip install -r requirements.txt`

---

**Generated with Claude Code**
