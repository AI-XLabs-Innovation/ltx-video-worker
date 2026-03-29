#!/bin/bash
set -e

# Use the LTX-2 venv which has ltx_core and ltx_pipelines installed
export VIRTUAL_ENV=/app/ltx2/.venv
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Install runpod + boto3 into the venv if not already present
pip install --quiet runpod boto3 2>/dev/null || true

# Download models on first start if not already present
if [ "$DOWNLOAD_MODELS_ON_START" = "true" ]; then
    CHECKPOINT="$MODEL_DIR/ltx-2.3-22b-dev-fp8.safetensors"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "📥 Models not found at $MODEL_DIR — downloading..."
        python /app/download_models.py
    else
        echo "✅ Models already present at $MODEL_DIR — skipping download"
    fi
fi

echo "🚀 Starting RunPod handler..."
exec python -u /app/handler.py
