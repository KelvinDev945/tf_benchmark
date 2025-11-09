# syntax=docker/dockerfile:1.6

# TensorFlow Multi-Engine CPU Inference Benchmark
# Multi-architecture Dockerfile (x86_64 and ARM64)

ARG PYTHON_VERSION=3.10
ARG ENABLE_OPENVINO=false

FROM python:${PYTHON_VERSION}-slim AS builder

ARG TARGETARCH=amd64
ARG ENABLE_OPENVINO=false

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Install build-time tools and headers required for dependency resolution
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libhdf5-dev \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv "$VIRTUAL_ENV" \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /deps

# Copy dependency manifest
COPY requirements.txt .

# Resolve and install Python dependencies into the virtual environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "$VIRTUAL_ENV/bin/python" --no-cache-dir -r requirements.txt

# Optional: add OpenVINO for x86_64 builds when ENABLE_OPENVINO=true
RUN if [ "$TARGETARCH" = "amd64" ] && [ "$ENABLE_OPENVINO" = "true" ]; then \
        uv pip install --python "$VIRTUAL_ENV/bin/python" --no-cache-dir openvino==2023.2.0 openvino-dev==2023.2.0; \
    else \
        echo "Skipping OpenVINO (arch: $TARGETARCH)"; \
    fi

# Attempt to install TensorFlow Lite runtime (best-effort)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "$VIRTUAL_ENV/bin/python" --no-cache-dir \
        --extra-index-url https://google-coral.github.io/py-repo/ \
        tflite_runtime 2>/dev/null || \
    echo "Warning: tflite_runtime not available for Python ${PYTHON_VERSION}, falling back to TensorFlow Lite bundled runtime"

# Clean up caches and bytecode inside the virtual environment
RUN find "$VIRTUAL_ENV" -type d -name "__pycache__" -exec rm -rf {} + && \
    rm -rf /root/.cache/uv

# --------------------------------------------------------------------------- #

FROM python:${PYTHON_VERSION}-slim AS runtime

ARG TARGETARCH=amd64
ARG ENABLE_OPENVINO=false

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Install runtime shared libraries only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-310 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pre-built virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Prepare application directories
RUN mkdir -p /app/src /app/configs /app/results /app/models /app/data && \
    chmod -R 755 /app

# Copy source code and configs
COPY src/ /app/src/
COPY configs/ /app/configs/

# Verify installations
RUN python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" && \
    python -c "import onnxruntime as ort; print(f'ONNX Runtime version: {ort.__version__}')"

# Verify OpenVINO (x86_64 only, if enabled)
RUN if [ "$TARGETARCH" = "amd64" ] && [ "$ENABLE_OPENVINO" = "true" ]; then \
        python -c "from openvino.runtime import Core; print(f'OpenVINO available: True')"; \
    else \
        echo "OpenVINO verification skipped"; \
    fi

# Display system information
RUN python -c "import platform; import psutil; \
    print(f'Platform: {platform.platform()}'); \
    print(f'Architecture: {platform.machine()}'); \
    print(f'CPU count: {psutil.cpu_count()}'); \
    print(f'Total memory: {psutil.virtual_memory().total / (1024**3):.2f} GB')"

# Entrypoint executes inside the virtual environment
ENTRYPOINT ["python"]

# Default command
CMD ["src/main.py", "--help"]
