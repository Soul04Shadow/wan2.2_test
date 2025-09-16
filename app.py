from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
import uuid
import os

from inference import generate_video

app = FastAPI()

@app.post("/generate")
async def generate(
    prompt: str = Body(..., embed=True),
    duration: int = Body(10),
    fps: int = Body(24),
    resolution: str = Body("720p")
):
    output_file = f"/tmp/{uuid.uuid4()}.mp4"
    generate_video(prompt, duration, fps, resolution, output_file)
    return FileResponse(output_file, media_type="video/mp4", filename="result.mp4")

@app.get("/")
def home():
    return {"status": "ok", "message": "Wan2 Koyeb API running!"}
