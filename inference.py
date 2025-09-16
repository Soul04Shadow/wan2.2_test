import os
import json
import torch
import imageio
import time
from diffusers import DiffusionPipeline
from huggingface_hub import login

CONFIG_FILE = "wan_config.json"
with open(CONFIG_FILE, "r") as f:
    CONFIG = json.load(f)

HF_TOKEN = os.environ.get("HF_TOKEN", None)
MODEL_ID = os.environ.get("WAN_MODEL", CONFIG.get("default_model"))

# Where to cache models on instance disk (avoid repeated downloads)
os.environ["HF_HOME"] = "/cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface/transformers"
os.makedirs("/cache/huggingface", exist_ok=True)

def load_pipeline():
    print("[INFO] Logging into Hugging Face (if token provided)...")
    if HF_TOKEN:
        login(HF_TOKEN)

    print(f"[INFO] Loading pipeline: {MODEL_ID} (fp16)...")
    # Load with safetensors if available and use fp16 for A100
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=True,
        revision="main",
        # use_auth_token=HF_TOKEN  # not required if login() used
    )

    # Send model to GPU
    pipe = pipe.to("cuda")
    # Enable memory optimizations (works with accelerate)
    try:
        pipe.enable_model_cpu_offload()  # helps when memory pressure exists
    except Exception:
        pass

    # Optional: enable xformers attention (if installed)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    return pipe

# Lazy-load pipeline object once
PIPE = None

def ensure_pipe():
    global PIPE
    if PIPE is None:
        PIPE = load_pipeline()
    return PIPE

def map_resolution(res_str):
    if res_str.lower() in ("720p","720"):
        return (1280, 720)
    if res_str.lower() in ("1080p","1080"):
        return (1920, 1080)
    return (1280, 720)

def generate_video(prompt: str, duration: int, fps: int, resolution: str, output_file: str):
    start_time = time.time()
    pipe = ensure_pipe()

    width, height = map_resolution(resolution)
    num_frames = max(1, int(duration * fps))

    print(f"[INFO] Generating: {duration}s ({num_frames} frames) at {width}x{height} @ {fps}fps")
    # call pipeline - note: exact kwargs depend on the Wan2 pipeline interface
    # Wan2 pipeline often expects: prompt, num_frames, width, height
    result = pipe(prompt=prompt, num_frames=num_frames, width=width, height=height)

    # `result.frames` should be list/np array of frames (PIL images or numpy arrays)
    frames = getattr(result, "frames", None)
    if frames is None:
        # Try other attribute names
        frames = result if isinstance(result, list) else []

    if not frames:
        raise RuntimeError("No frames returned from pipeline. Check pipeline API and model compatibility.")

    # Write MP4 with imageio (libx264)
    writer = imageio.get_writer(output_file, fps=fps, codec="libx264", quality=8)
    for i, frame in enumerate(frames):
        # frame may be PIL.Image; convert to numpy if needed
        if hasattr(frame, "convert"):
            frame = frame.convert("RGB")
            frame = np.array(frame)
        writer.append_data(frame)
    writer.close()

    elapsed = time.time() - start_time
    print(f"[INFO] Done. Saved to {output_file} in {elapsed:.1f}s")
    return output_file
