FROM runpod/pytorch:2.7.0-py3.12-cuda12.8.0-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install LTX-2 from source
RUN git clone https://github.com/Lightricks/LTX-2.git /app/ltx2 && \
    cd /app/ltx2 && \
    pip install --no-cache-dir -e .

# Copy handler
COPY handler.py .

# Download models at build time (baked into image for fast cold starts)
# Comment this out if using RunPod network volume or model caching instead
COPY download_models.py .
RUN python download_models.py

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV MODEL_DIR=/app/models

CMD ["python", "-u", "/app/handler.py"]
