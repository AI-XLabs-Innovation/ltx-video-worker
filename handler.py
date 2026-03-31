"""
RunPod Serverless Handler for LTX Video 2.3 (FP8)
Generates videos from text/image prompts using Lightricks LTX-2.3
Supports: text-to-video, image-to-video, first+last frame interpolation
"""

import os
import uuid
import base64
import tempfile
import time
import requests
import runpod
import torch
import boto3
from botocore.config import Config as BotoConfig

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "")  # For R2 compatibility
S3_REGION = os.environ.get("S3_REGION", "auto")
S3_PUBLIC_URL = os.environ.get("S3_PUBLIC_URL", "")  # CDN/public URL prefix
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

# ============================================================
# MODEL LOADING (runs once at worker startup)
# ============================================================

print("🔄 Loading LTX Video 2.3 pipeline...")
start_time = time.time()

from ltx_core.quantization import QuantizationPolicy
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_core.components.guiders import MultiModalGuiderParams

# Distilled LoRA for faster inference
distilled_lora = [
    LoraPathStrengthAndSDOps(
        os.path.join(MODEL_DIR, "ltx-2.3-22b-distilled-lora-384.safetensors"),
        0.6,
        LTXV_LORA_COMFY_RENAMING_MAP,
    ),
]

pipeline = TI2VidTwoStagesPipeline(
    checkpoint_path=os.path.join(MODEL_DIR, "ltx-2.3-22b-dev-fp8.safetensors"),
    distilled_lora=distilled_lora,
    spatial_upsampler_path=os.path.join(MODEL_DIR, "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
    gemma_root=os.path.join(MODEL_DIR, "gemma"),
    loras=[],
    quantization=QuantizationPolicy.fp8_cast(),
)

load_time = time.time() - start_time
print(f"✅ Pipeline loaded in {load_time:.1f}s")

# ============================================================
# S3/R2 UPLOAD
# ============================================================

s3_client = None
if S3_BUCKET and AWS_ACCESS_KEY_ID:
    s3_kwargs = {
        "aws_access_key_id": AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
        "region_name": S3_REGION,
        "config": BotoConfig(signature_version="s3v4"),
    }
    if S3_ENDPOINT:
        s3_kwargs["endpoint_url"] = S3_ENDPOINT
    s3_client = boto3.client("s3", **s3_kwargs)
    print(f"✅ S3 client configured (bucket: {S3_BUCKET})")


def upload_to_storage(file_path: str) -> str:
    """Upload video to S3/R2 and return the public URL."""
    key = f"ltx-videos/{uuid.uuid4()}.mp4"
    s3_client.upload_file(
        file_path,
        S3_BUCKET,
        key,
        ExtraArgs={"ContentType": "video/mp4"},
    )
    if S3_PUBLIC_URL:
        return f"{S3_PUBLIC_URL.rstrip('/')}/{key}"
    if S3_ENDPOINT:
        return f"{S3_ENDPOINT}/{S3_BUCKET}/{key}"
    return f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"


# ============================================================
# IMAGE DOWNLOAD HELPER
# ============================================================

def download_image(url: str) -> str:
    """Download an image URL to a temp file and return the path."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    ext = ".jpg"
    ct = resp.headers.get("content-type", "")
    if "png" in ct:
        ext = ".png"
    elif "webp" in ct:
        ext = ".webp"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(resp.content)
    tmp.close()
    return tmp.name


# ============================================================
# VALIDATION
# ============================================================

def validate_input(job_input: dict) -> str | None:
    """Validate input parameters, return error message or None."""
    width = job_input.get("width", 768)
    height = job_input.get("height", 512)
    num_frames = job_input.get("num_frames", 121)

    if width % 32 != 0:
        return f"width must be divisible by 32, got {width}"
    if height % 32 != 0:
        return f"height must be divisible by 32, got {height}"
    if (num_frames - 1) % 8 != 0:
        return f"num_frames must be 8*N+1 (e.g. 25, 57, 121), got {num_frames}"
    if not job_input.get("prompt"):
        return "prompt is required"
    return None


# ============================================================
# HANDLER
# ============================================================

def handler(job):
    """Process a video generation request.

    Supports three modes based on input fields:
      - Text-to-video: only 'prompt' provided
      - Image-to-video: 'image' URL provided (conditioned on first frame)
      - First+last frame interpolation: 'image' + 'last_image' URLs provided
    """
    job_input = job["input"]

    # Validate
    error = validate_input(job_input)
    if error:
        return {"error": error}

    # Extract parameters with defaults
    prompt = job_input["prompt"]
    negative_prompt = job_input.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted")
    height = job_input.get("height", 512)
    width = job_input.get("width", 768)
    num_frames = job_input.get("num_frames", 121)  # 8*15+1 = ~5s at 25fps
    seed = job_input.get("seed", -1)
    num_inference_steps = job_input.get("num_inference_steps", 40)
    frame_rate = job_input.get("frame_rate", 25.0)
    guidance_scale = job_input.get("guidance_scale", 3.0)
    stg_scale = job_input.get("stg_scale", 1.0)
    image_strength = job_input.get("image_strength", 1.0)

    # Image conditioning URLs (optional)
    image_url = job_input.get("image")           # first frame / I2V
    last_image_url = job_input.get("last_image")  # last frame for interpolation

    # Random seed if -1
    if seed == -1:
        seed = torch.randint(0, 2**32, (1,)).item()

    # Guider params
    video_guider_params = MultiModalGuiderParams(
        cfg_scale=guidance_scale,
        stg_scale=stg_scale,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[29],
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[29],
    )

    # Build image conditioning list
    image_conditions = []
    downloaded_files = []

    try:
        if image_url:
            print(f"📥 Downloading first-frame image: {image_url[:80]}...")
            first_path = download_image(image_url)
            downloaded_files.append(first_path)
            image_conditions.append(
                ImageConditioningInput(
                    path=first_path,
                    frame_idx=0,
                    strength=image_strength,
                )
            )

        if last_image_url:
            print(f"📥 Downloading last-frame image: {last_image_url[:80]}...")
            last_path = download_image(last_image_url)
            downloaded_files.append(last_path)
            last_frame_idx = num_frames - 1
            image_conditions.append(
                ImageConditioningInput(
                    path=last_path,
                    frame_idx=last_frame_idx,
                    strength=image_strength,
                )
            )

        mode = "t2v"
        if image_url and last_image_url:
            mode = "keyframe"
        elif image_url:
            mode = "i2v"
        print(f"🎬 Mode: {mode} | {width}x{height} | {num_frames} frames @ {frame_rate}fps")

        # Generate
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name

        gen_start = time.time()

        pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_path=output_path,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            images=image_conditions,
        )

        gen_time = time.time() - gen_start

        # Return result
        result = {
            "seed": seed,
            "generation_time_seconds": round(gen_time, 2),
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
            "mode": mode,
        }

        # Upload to S3/R2 if configured
        if s3_client:
            url = upload_to_storage(output_path)
            result["video_url"] = url
        else:
            # Fallback to base64
            with open(output_path, "rb") as f:
                result["video_base64"] = base64.b64encode(f.read()).decode("utf-8")

        return result

    finally:
        # Clean up downloaded images and output
        for f in downloaded_files:
            if os.path.exists(f):
                os.unlink(f)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.unlink(output_path)


runpod.serverless.start({"handler": handler})
