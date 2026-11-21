"""AFG AI backend - Groq chat, Pollinations text-to-image, Pixazo image editing."""
import asyncio
import base64
import io
import json
import logging
import os
import time
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from groq import Groq

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("afg_ai")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")
PIXAZO_API_KEY = os.getenv("PIXAZO_API_KEY") or os.getenv("PIXAZO_KEY")
IMAGE_MODEL = os.getenv("POLLINATIONS_IMAGE_MODEL", "black-forest-labs/flux.1-schnell")
IMAGE_EDIT_MODEL = os.getenv("POLLINATIONS_EDIT_MODEL", "black-forest-labs/flux.1-kontext-pro")
IMAGE_API_BASE = "https://gen.pollinations.ai"
PIXAZO_EDIT_URL = os.getenv("PIXAZO_EDIT_URL", "https://gateway.pixazo.ai/gpt-image-2-5-sunburst/v1/image-to-image/editing")
PIXAZO_STATUS_BASE = "https://gateway.pixazo.ai/v2/requests/status/"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_RESPONSE_BYTES = 50 * 1024 * 1024


app = FastAPI(
    title="AFG AI", version="1.2.0",
    description="Multilingual AI assistant with Groq chat, Pollinations text-to-image, and Pixazo image editing.",
)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12_000)
    mode: str = Field(default="chat", max_length=32)
    history: list[dict] = Field(default_factory=list, max_length=20)


PROMPTS = {
    "chat": "You are AFG AI, a helpful, accurate, and multilingual assistant.",
    "coding": "You are an expert programming assistant. Explain clearly and provide safe, practical code.",
    "translator": "You are a professional translator. Preserve meaning, tone, and context.",
    "writer": "You are a professional writing assistant. Produce polished, natural writing.",
    "analysis": "You are a careful analytical assistant. Separate facts, assumptions, and uncertainty.",
}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "chat_configured": groq_client is not None,
        "chat_model": CHAT_MODEL,
        "image_configured": bool(POLLINATIONS_API_KEY),
        "image_provider": "pollinations",
        "image_model": IMAGE_MODEL,
        "image_edit_provider": "pixazo",
        "pixazo_edit_configured": bool(PIXAZO_API_KEY),
        "pollinations_edit_model": IMAGE_EDIT_MODEL,
    }


@app.post("/api/chat")
def chat(data: ChatRequest):
    if groq_client is None:
        raise HTTPException(status_code=503, detail="Chat service is not configured. Add GROQ_API_KEY in Render Environment.")
    system_prompt = PROMPTS.get(data.mode.lower(), PROMPTS["chat"])
    messages = [{"role": "system", "content": system_prompt + "\nReply in the language used by the user."}]
    for item in data.history[-20:]:
        role = item.get("role")
        if role not in ("user", "assistant"):
            continue
        text = str(item.get("text", "")).strip()[:6000]
        if text:
            messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": data.message.strip()})
    try:
        result = groq_client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.7, max_completion_tokens=4096)
        answer = result.choices[0].message.content if result.choices else None
        if not answer:
            return {"answer": "The AI returned an empty response. Please try again."}
        return {"answer": answer}
    except Exception as exc:
        logger.exception("Groq chat request failed (model=%s).", CHAT_MODEL)
        raise HTTPException(status_code=502, detail="Chat request failed. Check Render logs for details.") from exc



def prepare_edit_output(raw: bytes) -> tuple[bytes, str, str]:
    """Decode image and re-encode as PNG without resizing/cropping."""
    if not raw:
        raise ValueError("The image provider returned an empty image.")
    if len(raw) > MAX_IMAGE_RESPONSE_BYTES:
        raise ValueError("The edited image is larger than the 50 MB response limit.")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for PNG output. Add 'Pillow' to requirements.txt and redeploy.") from exc
    try:
        with Image.open(io.BytesIO(raw)) as source:
            source.load()
            image = source.copy()
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGBA" if "transparency" in image.info else "RGB")
            output = io.BytesIO()
            image.save(output, format="PNG", optimize=False, compress_level=6)
            png_data = output.getvalue()
    except Exception as exc:
        raise ValueError("The image provider returned data that could not be decoded as an image.") from exc
    if len(png_data) > MAX_IMAGE_RESPONSE_BYTES:
        raise ValueError("The PNG image is larger than the 50 MB response limit.")
    return png_data, "image/png", "png"


def _pixazo_find_image(obj):
    """Extract an image URL or base64 payload from common Pixazo response shapes."""
    if isinstance(obj, dict):
        for key in ("b64_json", "image_base64", "base64", "image_b64"):
            value = obj.get(key)
            if isinstance(value, str) and value:
                try:
                    return base64.b64decode(value, validate=True)
                except Exception:
                    pass
        for key in ("image_url", "url", "output_url", "file_url"):
            value = obj.get(key)
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                return value
        for value in obj.values():
            found = _pixazo_find_image(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _pixazo_find_image(value)
            if found:
                return found
    return None



def _download_pixazo_image(url: str) -> bytes:
    req = Request(url, headers={"Accept": "image/*", "User-Agent": "AFG-AI/1.4"})
    with urlopen(req, timeout=120) as resp:
        data = resp.read(MAX_IMAGE_RESPONSE_BYTES + 1)
    if len(data) > MAX_IMAGE_RESPONSE_BYTES:
        raise ValueError("Pixazo image output exceeded 50 MB.")
    return data



def _pixazo_edit_image(raw: bytes, mime_type: str, prompt: str) -> bytes:
    """Submit a base64 data-URI image edit to Pixazo and poll its async job."""
    if not PIXAZO_API_KEY:
        raise RuntimeError("Pixazo is not configured. Add PIXAZO_API_KEY in Render Environment.")
    data_uri = f"data:{mime_type};base64,{base64.b64encode(raw).decode('ascii')}"
    payload = json.dumps({"prompt": prompt, "image": [data_uri], "size": "2048x2048", "quality": "high", "format": "png"}).encode()
    headers = {
        "Ocp-Apim-Subscription-Key": PIXAZO_API_KEY,
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "Accept": "application/json",
        "User-Agent": "AFG-AI/1.4",
    }
    req = Request(PIXAZO_EDIT_URL, data=payload, headers=headers, method="POST")
    with urlopen(req, timeout=180) as resp:
        response_payload = json.loads(resp.read(2 * 1024 * 1024).decode("utf-8"))
    # Some endpoints return the image immediately; others return an async request ID/polling URL.
    image = _pixazo_find_image(response_payload)
    if isinstance(image, bytes):
        return image
    if isinstance(image, str):
        return _download_pixazo_image(image)
    poll_url = response_payload.get("polling_url") or response_payload.get("status_url")
    request_id = response_payload.get("request_id")
    if not poll_url and request_id:
        poll_url = PIXAZO_STATUS_BASE + quote(str(request_id), safe="")
    if not poll_url:
        raise ValueError("Pixazo response contained no image or polling URL: " + json.dumps(response_payload)[:1500])
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        poll_req = Request(poll_url, headers={"Ocp-Apim-Subscription-Key": PIXAZO_API_KEY, "Accept": "application/json", "User-Agent": "AFG-AI/1.4"})
        with urlopen(poll_req, timeout=60) as poll_resp:
            poll_data = poll_resp.read(2 * 1024 * 1024)
            ctype = poll_resp.headers.get("Content-Type", "application/json").split(";", 1)[0]
            if ctype.startswith("image/"):
                return poll_data
        try:
            status_payload = json.loads(poll_data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("Pixazo status endpoint returned an unrecognized response.")
        image = _pixazo_find_image(status_payload)
        if isinstance(image, bytes):
            return image
        if isinstance(image, str):
            return _download_pixazo_image(image)
        status = str(status_payload.get("status", status_payload.get("state", ""))).lower()
        if status in ("failed", "failure", "error", "cancelled", "canceled"):
            raise RuntimeError("Pixazo image-edit job failed: " + json.dumps(status_payload)[:1000])
        time.sleep(3)
    raise TimeoutError("Pixazo image editing timed out after 4 minutes.")


@app.post("/api/image")
async def image(prompt: str = Form(..., min_length=1, max_length=4000), image_file: Optional[UploadFile] = File(default=None)):
    """Text prompts use Pollinations. Uploaded photos are edited with Pixazo."""
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Please enter an image description or editing instruction.")
    try:
        if image_file is not None:
            raw = await image_file.read(MAX_IMAGE_BYTES + 1)
            if len(raw) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
            mime_type = image_file.content_type or "application/octet-stream"
            if mime_type not in ("image/jpeg", "image/png", "image/webp"):
                raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")
            if not PIXAZO_API_KEY:
                raise HTTPException(status_code=503, detail="Photo editing is not configured. Add PIXAZO_API_KEY in Render Environment.")
            edited = await asyncio.to_thread(_pixazo_edit_image, raw, mime_type, prompt)
            body, content_type, extension = prepare_edit_output(edited)
        else:
            if not POLLINATIONS_API_KEY:
                raise HTTPException(status_code=503, detail="Image generation is not configured. Add POLLINATIONS_API_KEY in Render Environment.")
            if POLLINATIONS_API_KEY.startswith("pk_"):
                raise HTTPException(status_code=503, detail="Use a secret Pollinations API key beginning with sk_. Keep the key private.")
            query = urlencode({"model": IMAGE_MODEL, "width": 1024, "height": 1024})
            url = f"{IMAGE_API_BASE}/image/{quote(prompt, safe='')}?{query}"
            request = Request(url, headers={"Authorization": f"Bearer {POLLINATIONS_API_KEY}", "Accept": "image/*", "User-Agent": "AFG-AI/1.4"}, method="GET")
            def fetch_image():
                with urlopen(request, timeout=180) as response:
                    ctype = response.headers.get("Content-Type", "image/jpeg").split(";", 1)[0]
                    result = response.read(MAX_IMAGE_RESPONSE_BYTES + 1)
                if len(result) > MAX_IMAGE_RESPONSE_BYTES:
                    raise ValueError("Generated image is larger than the 50 MB limit.")
                return result, ctype
            body, content_type = await asyncio.to_thread(fetch_image)
            if not content_type.startswith("image/"):
                raise HTTPException(status_code=502, detail="Pollinations did not return an image. Check key/model permissions.")
            extension = "png" if "png" in content_type else "webp" if "webp" in content_type else "jpg"
        return Response(content=body, media_type=content_type, headers={"Content-Disposition": f'inline; filename="afg-ai-result.{extension}"'})
    except HTTPException:
        raise
    except HTTPError as exc:
        error_body = exc.read(2000).decode("utf-8", errors="replace")
        logger.warning("Image API returned HTTP %s: %s", exc.code, error_body)
        if image_file is not None:
            if exc.code in (401, 403):
                detail = "Pixazo API key is invalid or has no access to this model. Check PIXAZO_API_KEY and model permissions."
            elif exc.code == 429:
                detail = "Pixazo rate limit reached. Wait and try again."
            elif exc.code == 400:
                detail = "Pixazo rejected the edit request or input image. Check Render logs."
            else:
                detail = f"Pixazo image-edit request failed (HTTP {exc.code}). Check Render logs."
        else:
            detail = f"Pollinations image generation failed (HTTP {exc.code}). Check Render logs."
        raise HTTPException(status_code=502, detail=detail) from exc
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        provider = "Pixazo" if image_file is not None else "Pollinations"
        logger.exception("%s image request failed.", provider)
        raise HTTPException(status_code=502, detail=f"{provider} could not process the image. Check Render logs and try again.") from exc
    except Exception as exc:
        provider = "Pixazo" if image_file is not None else "Pollinations"
        logger.exception("Unexpected %s image error.", provider)
        raise HTTPException(status_code=502, detail="Image request failed. Check Render logs.") from exc


@app.get("/")
def index():
    return FileResponse("frontend/index.html")


@app.get("/style.css")
def css():
    return FileResponse("frontend/style.css", media_type="text/css")

@app.get("/app.js")
def js():
    return FileResponse("frontend/app.js", media_type="application/javascript")
