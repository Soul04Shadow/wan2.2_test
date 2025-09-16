import os
import json
import torch
import imageio

from diffusers import DiffusionPipeline

CONFIG_FILE = "wan_config.json"

with open(CONFIG_FILE, "r") as f:
    CONFIG = json.load(f)

# Load Wan2.2 once at startup
print("[INFO] Loading Wan2.2 model...")
pipe = DiffusionPipeline.from_pretrained(
    "TencentARC/Wan2.2-T2V",  # official Hugging Face repo
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")
pipe.enable_model_cpu_offload()  # helps with VRAM efficiency

def generate_video(prompt: str, duration: int, fps: int, resolution: str, output_file: str):
    num_frames = duration * fps
    print(f"[INFO] Generating video → Prompt: '{prompt}', {duration}s, {fps}fps, {resolution}")

    # Map resolution string to tuple
    if resolution == "720p":
        res = (1280, 720)
    elif resolution == "1080p":
        res = (1920, 1080)
    else:
        res = (1280, 720)  # fallback

    # Run generation
    video_frames = pipe(
        prompt=prompt,
        num_frames=num_frames,
        width=res[0],
        height=res[1]
    ).frames  # list of PIL images

    # Save as mp4
    with imageio.get_writer(output_file, fps=fps, codec="libx264") as writer:
        for frame in video_frames:
            writer.append_data(frame)

    return output_file
