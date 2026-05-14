FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Python dependencies (into system python)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone LTX-2 and install via uv sync
RUN git clone https://github.com/Lightricks/LTX-2.git /app/ltx2 && \
    cd /app/ltx2 && \
    uv sync --frozen

# Fix circular import: move top-level import in fuse_loras.py to lazy imports
# fp8_cast.py imports from loader (triggers __init__) which imports fuse_loras
# which imports back from fp8_cast before it's fully initialized
RUN F=/app/ltx2/packages/ltx-core/src/ltx_core/loader/fuse_loras.py && \
    sed -i '/^from ltx_core.quantization.fp8_cast import _fused_add_round_launch/d' $F && \
    sed -i '/^from ltx_core.quantization.fp8_scaled_mm import quantize_weight_to_fp8_per_tensor/d' $F && \
    sed -i 's/        _fused_add_round_launch(deltas, weight, seed=0)/        from ltx_core.quantization.fp8_cast import _fused_add_round_launch; _fused_add_round_launch(deltas, weight, seed=0)/' $F && \
    sed -i 's/    new_fp8_weight, new_weight_scale = quantize_weight_to_fp8_per_tensor(new_weight)/    from ltx_core.quantization.fp8_scaled_mm import quantize_weight_to_fp8_per_tensor; new_fp8_weight, new_weight_scale = quantize_weight_to_fp8_per_tensor(new_weight)/' $F

# Install handler dependencies into the LTX-2 venv
RUN cd /app/ltx2 && uv pip install runpod boto3 requests huggingface_hub safetensors accelerate sentencepiece fastapi uvicorn

# Copy handler and model downloader
COPY handler.py .
COPY download_models.py .
COPY start.sh .
RUN chmod +x start.sh

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV MODEL_DIR=/runpod-volume/models
ENV DOWNLOAD_MODELS_ON_START=true
ENV RUNPOD_MODE=serverless

CMD ["/app/start.sh"]
