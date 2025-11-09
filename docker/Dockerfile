# TensorFlow Multi-Engine CPU Inference Benchmark
# Multi-architecture Dockerfile (x86_64 and ARM64)

# Build argument for target architecture
ARG TARGETARCH=amd64

FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config \
    libhdf5-dev \
    libssl-dev \
    libffi-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Update pip, setuptools, and wheel
RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set python3.11 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install architecture-specific packages
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        echo "Installing OpenVINO for x86_64..." && \
        pip3 install --no-cache-dir --no-deps openvino==2023.2.0 && \
        pip3 install --no-cache-dir --no-deps openvino-dev==2023.2.0 && \
        pip3 install --no-cache-dir openvino==2023.2.0 openvino-dev==2023.2.0; \
    else \
        echo "Skipping OpenVINO (ARM64 not supported)"; \
    fi

# Install TFLite runtime (without upgrading dependencies)
RUN pip3 install --no-cache-dir --no-deps --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime || \
    pip3 install --no-cache-dir --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

# Create necessary directories
RUN mkdir -p /app/src /app/configs /app/results /app/models /app/data

# Copy application code
COPY src/ /app/src/
COPY configs/ /app/configs/

# Set permissions
RUN chmod -R 755 /app

# Verify installations
RUN python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" && \
    python3 -c "import onnxruntime as ort; print(f'ONNX Runtime version: {ort.__version__}')"

# Verify OpenVINO (x86_64 only)
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        python3 -c "from openvino.runtime import Core; print(f'OpenVINO available: True')"; \
    fi

# Display system information
RUN python3 -c "import platform; import psutil; \
    print(f'Platform: {platform.platform()}'); \
    print(f'Architecture: {platform.machine()}'); \
    print(f'CPU count: {psutil.cpu_count()}'); \
    print(f'Total memory: {psutil.virtual_memory().total / (1024**3):.2f} GB')"

# Set entrypoint
ENTRYPOINT ["python3"]

# Default command
CMD ["src/main.py", "--help"]
