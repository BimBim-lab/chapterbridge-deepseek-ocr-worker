# Dockerfile for DeepSeek OCR Worker
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables (override these in cloud platform)
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

# Run worker
CMD ["python3", "workers/ocr/main.py", "--poll-seconds", "3"]
