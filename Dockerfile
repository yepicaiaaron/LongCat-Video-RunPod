# LongCat-Video RunPod Serverless Dockerfile
# Base image with CUDA 12.1 support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/workspace/weights/LongCat-Video
ENV HF_HOME=/workspace/huggingface
ENV CONTEXT_PARALLEL_SIZE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 support
RUN pip install --no-cache-dir torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install FlashAttention 2
RUN pip install --no-cache-dir ninja psutil packaging
RUN pip install --no-cache-dir flash_attn==2.7.4.post1 --no-build-isolation

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Install RunPod SDK
RUN pip install --no-cache-dir runpod

# Copy application code
COPY . .# ARG HF_TOKEN
# RUN pip install --no-cache-dir "huggingface_hub[cli]" && \
#     huggingface-cli login --token $HF_TOKEN && \
#     huggingface-cli download meituan-longcat/LongCat-Video --local-dir ./weights/LongCat-Video

# Expose port for RunPod
EXPOSE 8000

# Run the handler
CMD ["python", "handler.py"]
