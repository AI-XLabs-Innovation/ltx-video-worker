#!/bin/bash
set -e

# Use the LTX-2 venv which has ltx_core, ltx_pipelines, and runpod installed
export VIRTUAL_ENV=/app/ltx2/.venv
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Verify runpod is importable
python -c "import runpod; print(f'✅ runpod {runpod.__version__} available')" || {
    echo "⚠️ runpod not found in venv, installing..."
    cd /app/ltx2 && uv pip install runpod boto3 requests fastapi uvicorn
}

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

echo "🚀 Starting LTX worker (mode: ${RUNPOD_MODE:-serverless})..."
exec python -u /app/handler.py
