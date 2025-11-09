#!/bin/bash

# TensorFlow Multi-Engine CPU Inference Benchmark
# Docker Image Build Script
#
# This script automatically builds Docker images for the current architecture
# and optionally builds OpenVINO-specific images for x86_64.

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

# Detect architecture
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)
        DOCKER_ARCH="amd64"
        ;;
    aarch64|arm64)
        DOCKER_ARCH="arm64"
        ;;
    *)
        print_error "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
DOCKER_DIR="$PROJECT_ROOT/docker"

print_header "TensorFlow Multi-Engine CPU Inference Benchmark - Build Script"

print_message "$BLUE" "System Information:"
echo "  Architecture: $ARCH (Docker: $DOCKER_ARCH)"
echo "  Project Root: $PROJECT_ROOT"
echo "  Docker Dir:   $DOCKER_DIR"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

print_success "Docker is installed: $(docker --version)"

# Change to project root
cd "$PROJECT_ROOT"

# Optional overrides
BASE_PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
BASE_ENABLE_OPENVINO="${ENABLE_OPENVINO:-false}"

if [ -z "${BUILD_OPENVINO_VARIANT+x}" ]; then
    if [ "$BASE_ENABLE_OPENVINO" = "true" ]; then
        BUILD_OPENVINO_VARIANT="false"
    else
        BUILD_OPENVINO_VARIANT="true"
    fi
fi

# Build main image
print_header "Building Main Image (tf-cpu-benchmark:${DOCKER_ARCH})"

if docker build \
    --build-arg TARGETARCH="$DOCKER_ARCH" \
    --build-arg PYTHON_VERSION="$BASE_PYTHON_VERSION" \
    --build-arg ENABLE_OPENVINO="$BASE_ENABLE_OPENVINO" \
    -t "tf-cpu-benchmark:${DOCKER_ARCH}" \
    -t "tf-cpu-benchmark:latest" \
    -f "$DOCKER_DIR/Dockerfile" \
    .; then
    print_success "Successfully built tf-cpu-benchmark:${DOCKER_ARCH}"
else
    print_error "Failed to build main image"
    exit 1
fi

# Build OpenVINO image (x86_64 only)
if [ "$DOCKER_ARCH" = "amd64" ]; then
    if [ "$BUILD_OPENVINO_VARIANT" = "true" ]; then
        print_header "Building OpenVINO Image (x86_64 only)"

        if docker build \
            --build-arg TARGETARCH="$DOCKER_ARCH" \
            --build-arg PYTHON_VERSION="$BASE_PYTHON_VERSION" \
            --build-arg ENABLE_OPENVINO=true \
            -t "tf-cpu-benchmark-openvino:${DOCKER_ARCH}" \
            -t "tf-cpu-benchmark-openvino:latest" \
            -f "$DOCKER_DIR/Dockerfile" \
            .; then
            print_success "Successfully built tf-cpu-benchmark-openvino:${DOCKER_ARCH}"
        else
            print_warning "Failed to build OpenVINO image (non-critical)"
        fi
    else
        print_warning "Skipping OpenVINO image (disabled via BUILD_OPENVINO_VARIANT=false)"
    fi
else
    print_warning "Skipping OpenVINO image (only supported on x86_64)"
fi

# Display built images
print_header "Build Summary"

echo "Built images:"
docker images | grep "tf-cpu-benchmark" | while read -r line; do
    print_success "$line"
done

# Test the image
print_header "Testing Image"

print_message "$BLUE" "Running basic verification..."

if docker run --rm "tf-cpu-benchmark:${DOCKER_ARCH}" \
    -c "import sys; import tensorflow as tf; import onnxruntime as ort; \
        print(f'Python: {sys.version}'); \
        print(f'TensorFlow: {tf.__version__}'); \
        print(f'ONNX Runtime: {ort.__version__}'); \
        print('Basic verification passed!')"; then
    print_success "Image verification passed!"
else
    print_error "Image verification failed!"
    exit 1
fi

# Verify OpenVINO (x86_64 only)
if [ "$DOCKER_ARCH" = "amd64" ]; then
    if docker images | grep -q "tf-cpu-benchmark-openvino"; then
        print_message "$BLUE" "Verifying OpenVINO image..."
        if docker run --rm "tf-cpu-benchmark-openvino:${DOCKER_ARCH}" \
            -c "from openvino.runtime import Core; print('OpenVINO verification passed!')"; then
            print_success "OpenVINO image verification passed!"
        else
            print_warning "OpenVINO image verification failed (non-critical)"
        fi
    fi
fi

# Print usage instructions
print_header "Usage Instructions"

echo "Run benchmark with Docker:"
echo ""
echo "  # Quick test"
echo "  docker run --rm -v \$(pwd)/results:/app/results \\"
echo "    tf-cpu-benchmark:${DOCKER_ARCH} \\"
echo "    src/main.py --mode quick"
echo ""
echo "  # Standard benchmark"
echo "  docker run --rm -v \$(pwd)/results:/app/results \\"
echo "    tf-cpu-benchmark:${DOCKER_ARCH} \\"
echo "    src/main.py --mode standard"
echo ""
echo "  # Custom configuration"
echo "  docker run --rm \\"
echo "    -v \$(pwd)/results:/app/results \\"
echo "    -v \$(pwd)/configs:/app/configs \\"
echo "    tf-cpu-benchmark:${DOCKER_ARCH} \\"
echo "    src/main.py --config /app/configs/benchmark_config.yaml"
echo ""

if [ "$DOCKER_ARCH" = "amd64" ]; then
    echo "Run with OpenVINO:"
    echo ""
    echo "  docker run --rm -v \$(pwd)/results:/app/results \\"
    echo "    tf-cpu-benchmark-openvino:${DOCKER_ARCH} \\"
    echo "    src/main.py --engines openvino --mode standard"
    echo ""
fi

print_success "Build completed successfully!"

exit 0
