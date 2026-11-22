"""CPU-first, local Stable Diffusion image-to-image engine for AFG AI.
Model weights are downloaded once from Hugging Face and then loaded from local cache.
No image-editing API is called. CPU mode is slow and memory-intensive.
"""
import os
import threading
import io
from PIL import Image


_MODEL_ID = os.getenv("AFG_LOCAL_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
_pipe = None
_lock = threading.Lock()


def _get_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe
    with _lock:
        if _pipe is not None:
            return _pipe
        try:
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline
        except ImportError as exc:
            raise RuntimeError("Local editor dependencies are missing. Install requirements-local.txt.") from exc
        # CPU-only, low-memory configuration. This does not make SD 1.5 lightweight;
        # host RAM and disk must still be sufficient.
        torch.set_num_threads(max(1, int(os.getenv("AFG_CPU_THREADS", "2"))))
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            _MODEL_ID,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=True,
        )
        pipe.to("cpu")
        pipe.enable_attention_slicing()
        _pipe = pipe
        return _pipe



def edit_image_local(raw: bytes, prompt: str) -> bytes:
    if not raw:
        raise ValueError("No input image was provided.")
    if len(raw) > 10 * 1024 * 1024:
        raise ValueError("Input image must be 10 MB or smaller.")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            image = im.convert("RGB")
    except Exception as exc:
        raise ValueError("Could not decode the uploaded image.") from exc
    # Keep aspect ratio and limit CPU memory use. SD 1.5 is generally best at multiples of 8.
    max_side = int(os.getenv("AFG_LOCAL_MAX_SIDE", "512"))
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    w, h = image.size
    w, h = max(64, (w // 8) * 8), max(64, (h // 8) * 8)
    image = image.resize((w, h), Image.Resampling.LANCZOS)
    pipe = _get_pipe()
    steps = max(5, min(30, int(os.getenv("AFG_LOCAL_STEPS", "15"))))
    strength = float(os.getenv("AFG_LOCAL_STRENGTH", "0.55"))
    strength = max(0.15, min(0.85, strength))
    guidance = float(os.getenv("AFG_LOCAL_GUIDANCE", "7.0"))
    with _lock:
        result = pipe(
            prompt=prompt[:4000], image=image, strength=strength,
            num_inference_steps=steps, guidance_scale=guidance,
        ).images[0]
    out = io.BytesIO()
    result.save(out, format="PNG", optimize=False)
    return out.getvalue()
