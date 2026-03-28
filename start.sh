#!/bin/bash
set -e

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
