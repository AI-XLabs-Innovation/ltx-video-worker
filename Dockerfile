FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone LTX-2 and install via uv sync (handles circular deps in workspace)
RUN git clone https://github.com/Lightricks/LTX-2.git /app/ltx2 && \
    cd /app/ltx2 && \
    uv sync --frozen && \
    # Make the venv packages accessible to the system python
    cp -r .venv/lib/python3.11/site-packages/ltx_core /usr/local/lib/python3.11/dist-packages/ && \
    cp -r .venv/lib/python3.11/site-packages/ltx_pipelines /usr/local/lib/python3.11/dist-packages/ && \
    # Copy any other installed deps not already present
    cp -rn .venv/lib/python3.11/site-packages/* /usr/local/lib/python3.11/dist-packages/ 2>/dev/null || true

# Copy handler and model downloader
COPY handler.py .
COPY download_models.py .
COPY start.sh .
RUN chmod +x start.sh

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV MODEL_DIR=/runpod-volume/models
ENV DOWNLOAD_MODELS_ON_START=true

CMD ["/app/start.sh"]
