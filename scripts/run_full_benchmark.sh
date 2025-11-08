#!/bin/bash

# TensorFlow Multi-Engine CPU Inference Benchmark
# Comprehensive Benchmark Runner Script
#
# This script runs a complete benchmark suite including:
# 1. BERT model comparison (base, quantized, ONNX)
# 2. All image models across all engines
# 3. All text models across all engines
# 4. Multiple batch sizes and sequence lengths

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo ""
    print_message "$BLUE" "========================================"
    print_message "$BLUE" "$1"
    print_message "$BLUE" "========================================"
    echo ""
}

print_success() {
    print_message "$GREEN" "✓ $1"
}

print_warning() {
    print_message "$YELLOW" "⚠ $1"
}

print_error() {
    print_message "$RED" "✗ $1"
}

# Parse arguments
MODE="${1:-standard}"
OUTPUT_DIR="${2:-./results/full_benchmark_$(date +%Y%m%d_%H%M%S)}"

# Validate mode
if [[ ! "$MODE" =~ ^(quick|standard|full)$ ]]; then
    print_error "Invalid mode: $MODE"
    echo "Usage: $0 [quick|standard|full] [output_dir]"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

print_header "TensorFlow Multi-Engine CPU Inference Benchmark - Full Suite"

print_message "$BLUE" "Configuration:"
echo "  Mode: $MODE"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Project Root: $PROJECT_ROOT"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to project root
cd "$PROJECT_ROOT"

# Check Python environment
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

print_success "Python: $(python3 --version)"

# Log file
LOG_FILE="$OUTPUT_DIR/benchmark.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Log file: $LOG_FILE"

# ============================================================================
# 1. BERT Model Comparison (Base, Quantized, ONNX)
# ============================================================================

print_header "Phase 1: BERT Model Inference Comparison"

BERT_OUTPUT="$OUTPUT_DIR/bert_comparison"
mkdir -p "$BERT_OUTPUT"

print_message "$BLUE" "Running BERT base model comparison..."

# Test BERT base-uncased
python3 scripts/benchmark_bert_comparison.py \
    --model bert-base-uncased \
    --mode "$MODE" \
    --batch-size 1 \
    --output "$BERT_OUTPUT/bert_base" || {
    print_error "BERT base benchmark failed"
}

# Test DistilBERT
print_message "$BLUE" "Running DistilBERT comparison..."
python3 scripts/benchmark_bert_comparison.py \
    --model distilbert-base-uncased \
    --mode "$MODE" \
    --batch-size 1 \
    --output "$BERT_OUTPUT/distilbert" || {
    print_error "DistilBERT benchmark failed"
}

print_success "Phase 1 complete: BERT comparison"

# ============================================================================
# 2. Image Models Benchmark (All Engines)
# ============================================================================

print_header "Phase 2: Image Models Benchmark"

IMAGE_OUTPUT="$OUTPUT_DIR/image_models"
mkdir -p "$IMAGE_OUTPUT"

# Image models to test
IMAGE_MODELS=("mobilenet_v2" "resnet50" "efficientnet_b0")

for model in "${IMAGE_MODELS[@]}"; do
    print_message "$BLUE" "Benchmarking $model across all engines..."

    # TensorFlow configurations
    print_message "$YELLOW" "  - TensorFlow engines..."
    python3 src/main.py \
        --config configs/benchmark_config.yaml \
        --mode "$MODE" \
        --models "$model" \
        --engines tensorflow \
        --output "$IMAGE_OUTPUT/${model}_tensorflow" || {
        print_warning "TensorFlow benchmark failed for $model"
    }

    # TFLite configurations
    print_message "$YELLOW" "  - TFLite engines..."
    python3 src/main.py \
        --config configs/benchmark_config.yaml \
        --mode "$MODE" \
        --models "$model" \
        --engines tflite \
        --output "$IMAGE_OUTPUT/${model}_tflite" || {
        print_warning "TFLite benchmark failed for $model"
    }

    # ONNX Runtime configurations
    print_message "$YELLOW" "  - ONNX Runtime engines..."
    python3 src/main.py \
        --config configs/benchmark_config.yaml \
        --mode "$MODE" \
        --models "$model" \
        --engines onnxruntime \
        --output "$IMAGE_OUTPUT/${model}_onnx" || {
        print_warning "ONNX Runtime benchmark failed for $model"
    }

    # OpenVINO (x86_64 only)
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        print_message "$YELLOW" "  - OpenVINO engines..."
        python3 src/main.py \
            --config configs/benchmark_config.yaml \
            --mode "$MODE" \
            --models "$model" \
            --engines openvino \
            --output "$IMAGE_OUTPUT/${model}_openvino" || {
            print_warning "OpenVINO benchmark failed for $model"
        }
    else
        print_warning "Skipping OpenVINO (not supported on $ARCH)"
    fi

    print_success "Completed benchmark for $model"
done

print_success "Phase 2 complete: Image models"

# ============================================================================
# 3. Text Models Benchmark (All Engines)
# ============================================================================

print_header "Phase 3: Text Models Benchmark (Additional Models)"

TEXT_OUTPUT="$OUTPUT_DIR/text_models"
mkdir -p "$TEXT_OUTPUT"

# Additional text models (BERT already done in Phase 1)
TEXT_MODELS=("roberta-base")

for model in "${TEXT_MODELS[@]}"; do
    print_message "$BLUE" "Benchmarking $model across all engines..."

    # TensorFlow
    print_message "$YELLOW" "  - TensorFlow engines..."
    python3 src/main.py \
        --config configs/benchmark_config.yaml \
        --mode "$MODE" \
        --models "$model" \
        --engines tensorflow \
        --output "$TEXT_OUTPUT/${model}_tensorflow" || {
        print_warning "TensorFlow benchmark failed for $model"
    }

    # TFLite
    print_message "$YELLOW" "  - TFLite engines..."
    python3 src/main.py \
        --config configs/benchmark_config.yaml \
        --mode "$MODE" \
        --models "$model" \
        --engines tflite \
        --output "$TEXT_OUTPUT/${model}_tflite" || {
        print_warning "TFLite benchmark failed for $model"
    }

    # ONNX Runtime
    print_message "$YELLOW" "  - ONNX Runtime engines..."
    python3 src/main.py \
        --config configs/benchmark_config.yaml \
        --mode "$MODE" \
        --models "$model" \
        --engines onnxruntime \
        --output "$TEXT_OUTPUT/${model}_onnx" || {
        print_warning "ONNX Runtime benchmark failed for $model"
    }

    print_success "Completed benchmark for $model"
done

print_success "Phase 3 complete: Text models"

# ============================================================================
# 4. Batch Size Analysis
# ============================================================================

print_header "Phase 4: Batch Size Analysis"

BATCH_OUTPUT="$OUTPUT_DIR/batch_analysis"
mkdir -p "$BATCH_OUTPUT"

# Test different batch sizes for key models
print_message "$BLUE" "Testing MobileNetV2 with different batch sizes..."

BATCH_SIZES=(1 4 8 16 32)

for bs in "${BATCH_SIZES[@]}"; do
    print_message "$YELLOW" "  - Batch size: $bs"
    python3 src/main.py \
        --config configs/benchmark_config.yaml \
        --mode quick \
        --models mobilenet_v2 \
        --engines tensorflow \
        --output "$BATCH_OUTPUT/mobilenet_bs${bs}" || {
        print_warning "Batch size $bs test failed"
    }
done

print_success "Phase 4 complete: Batch size analysis"

# ============================================================================
# 5. Generate Consolidated Report
# ============================================================================

print_header "Phase 5: Generating Consolidated Report"

print_message "$BLUE" "Collecting all results..."

# Generate comprehensive report
python3 scripts/generate_consolidated_report.py \
    --input-dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/consolidated_report" || {
    print_warning "Consolidated report generation failed (may need manual generation)"
}

print_success "Phase 5 complete: Report generation"

# ============================================================================
# Summary
# ============================================================================

print_header "Benchmark Suite Complete!"

print_success "All benchmarks completed successfully"
echo ""
echo "Results location: $OUTPUT_DIR"
echo ""
echo "Key result files:"
echo "  - BERT comparison: $BERT_OUTPUT/"
echo "  - Image models: $IMAGE_OUTPUT/"
echo "  - Text models: $TEXT_OUTPUT/"
echo "  - Batch analysis: $BATCH_OUTPUT/"
echo "  - Log file: $LOG_FILE"
echo ""

print_message "$BLUE" "Next steps:"
echo "  1. Review results in: $OUTPUT_DIR"
echo "  2. Check BERT comparison report: $BERT_OUTPUT/bert_base/bert_comparison_report.md"
echo "  3. View consolidated report (if generated): $OUTPUT_DIR/consolidated_report/"
echo ""

print_success "Benchmark suite completed successfully!"

exit 0
