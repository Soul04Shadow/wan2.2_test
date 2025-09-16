# Wan2 Video Generator on Koyeb

Deploy Wan2 / Wan2.2 on Koyeb GPU and generate short AI videos.

## 🚀 Deploy on Koyeb
1. Fork this repo.
2. On [Koyeb](https://www.koyeb.com), click **Deploy App → GitHub**.
3. Select this repo, set instance type:
   - `gpu-l40s` (48GB VRAM) → good for 720p 10s videos.
   - `gpu-a100` (80GB VRAM) → for 1080p or longer videos.
4. Deploy. Koyeb will build and start the API.

## 📡 Usage
Send a POST request to `/generate` with JSON:

```bash
curl -X POST https://<your-koyeb-url>/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "animation style monk meditating on hilltop near waterfall, cinematic orbit", "duration": 10, "fps": 24, "resolution": "720p"}' \
  --output result.mp4
