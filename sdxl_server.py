"""
sdxl_server.py — Drop this in the same folder as SDXL_worker.py and run:

    python sdxl_server.py

It loads SDXL once, keeps it in memory, and exposes a simple API on port 5050
that Loom (or anything else) can call to generate images.

Endpoints:
  POST /generate   { "prompt": "...", "width": 1024, "height": 1024,
                     "steps": 30, "guidance": 7.5, "seed": -1 }
                   → { "image": "<base64 png>", "prompt": "...", "seed": 123 }

  GET  /health     → { "status": "ready" }
  GET  /           → simple status page
"""

import base64
import io
import random
import traceback

from flask import Flask, jsonify, request
from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline

app = Flask(__name__)

# ── Load SDXL once at startup ─────────────────────────────────────────────────
print("Loading SDXL — this takes ~30s the first time...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
pipe.enable_model_cpu_offload()
print("✅ SDXL ready on port 5050")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return "<h2>SDXL Server running ✅</h2><p>POST to /generate to create images.</p>"


@app.route("/health")
def health():
    return jsonify({"status": "ready", "model": "stable-diffusion-xl-base-1.0"})


@app.route("/generate", methods=["POST"])
def generate():
    data    = request.get_json(force=True) or {}
    prompt          = data.get("prompt", "")
    negative_prompt = data.get("negative_prompt", "")
    width           = int(data.get("width",    512))
    height          = int(data.get("height",   512))
    steps           = int(data.get("steps",    40))
    guidance        = float(data.get("guidance", 7.0))
    seed            = int(data.get("seed", -1))

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    # Clamp dimensions to multiples of 8 (required by diffusers)
    width  = max(512, min(1024, (width  // 8) * 8))
    height = max(512, min(1024, (height // 8) * 8))

    # Seed handling
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    try:
        print(f"Generating: '{prompt[:80]}' | neg: '{negative_prompt[:40]}' | {width}x{height} | steps={steps} | seed={seed}")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]

        # Resize for display if larger than 900px on either side
        image.thumbnail((900, 900), Image.LANCZOS)

        # Encode to base64 PNG
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({
            "image":  b64,
            "prompt": prompt,
            "seed":   seed,
            "width":  image.width,
            "height": image.height,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # threaded=False is important — diffusers is not thread-safe
    app.run(host="0.0.0.0", port=5050, threaded=False)

