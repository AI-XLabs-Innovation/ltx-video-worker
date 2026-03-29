"""
Download LTX Video 2.3 model files from HuggingFace.
FP8 checkpoint is in a separate repo (Lightricks/LTX-2.3-fp8).
LoRA, upscaler are in the main repo (Lightricks/LTX-2.3).
Gemma text encoder requires HF_TOKEN (gated model).
"""

import os
from huggingface_hub import hf_hub_download, snapshot_download

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
os.makedirs(MODEL_DIR, exist_ok=True)

# FP8 checkpoint (~29GB) — separate repo
print("📥 Downloading LTX-2.3 FP8 checkpoint...")
hf_hub_download(
    repo_id="Lightricks/LTX-2.3-fp8",
    filename="ltx-2.3-22b-dev-fp8.safetensors",
    local_dir=MODEL_DIR,
    token=HF_TOKEN,
)

# Distilled LoRA (~7.6GB) — main repo
print("📥 Downloading distilled LoRA...")
hf_hub_download(
    repo_id="Lightricks/LTX-2.3",
    filename="ltx-2.3-22b-distilled-lora-384.safetensors",
    local_dir=MODEL_DIR,
    token=HF_TOKEN,
)

# Spatial upscaler (~1GB) — main repo
print("📥 Downloading spatial upscaler...")
hf_hub_download(
    repo_id="Lightricks/LTX-2.3",
    filename="ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    local_dir=MODEL_DIR,
    token=HF_TOKEN,
)

# Gemma 3 text encoder (gated — requires HF_TOKEN)
print("📥 Downloading Gemma text encoder...")
if not HF_TOKEN:
    print("⚠️  HF_TOKEN not set — Gemma is a gated model, download may fail")
snapshot_download(
    repo_id="google/gemma-3-4b-pt",
    local_dir=os.path.join(MODEL_DIR, "gemma"),
    ignore_patterns=["*.gguf"],
    token=HF_TOKEN,
)

print("✅ All models downloaded successfully!")
