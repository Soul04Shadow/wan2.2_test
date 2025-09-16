from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import FileResponse
import uuid, os, shutil, time
from inference import generate_video

TMP_DIR = "/tmp/wan_outputs"
os.makedirs(TMP_DIR, exist_ok=True)

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "message": "Wan2 Koyeb API running (A100)!"}

@app.post("/generate")
async def generate(
    prompt: str = Body(..., embed=True),
    duration: int = Body(10),
    fps: int = Body(24),
    resolution: str = Body("720p")
):
    out_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}.mp4")
    try:
        output_file = generate_video(prompt, duration, fps, resolution, out_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return FileResponse(output_file, media_type="video/mp4", filename=os.path.basename(output_file))
