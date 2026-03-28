"""
Download LTX Video 2.3 model files from HuggingFace at Docker build time.
This bakes models into the image for faster cold starts.
"""

import os
from huggingface_hub import hf_hub_download, snapshot_download

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")
os.makedirs(MODEL_DIR, exist_ok=True)

REPO_ID = "Lightricks/LTX-2.3"

# FP8 checkpoint (~20GB)
print("📥 Downloading LTX-2.3 FP8 checkpoint...")
hf_hub_download(
    repo_id=REPO_ID,
    filename="ltx-2.3-22b-dev-fp8.safetensors",
    local_dir=MODEL_DIR,
)

# Distilled LoRA
print("📥 Downloading distilled LoRA...")
hf_hub_download(
    repo_id=REPO_ID,
    filename="ltx-2.3-22b-distilled-lora-384.safetensors",
    local_dir=MODEL_DIR,
)

# Spatial upscaler
print("📥 Downloading spatial upscaler...")
hf_hub_download(
    repo_id=REPO_ID,
    filename="ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    local_dir=MODEL_DIR,
)

# Gemma text encoder
print("📥 Downloading Gemma text encoder...")
snapshot_download(
    repo_id="google/gemma-3-4b-pt",
    local_dir=os.path.join(MODEL_DIR, "gemma"),
    ignore_patterns=["*.gguf"],
)

print("✅ All models downloaded successfully!")
