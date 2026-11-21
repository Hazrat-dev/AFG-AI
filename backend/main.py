"""AFG AI backend - Groq for chat and Pollinations for image generation."""
import asyncio
import base64
import io
import json
import logging
import os
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
IMAGE_MODEL = os.getenv("POLLINATIONS_IMAGE_MODEL", "black-forest-labs/flux.1-schnell")
IMAGE_EDIT_MODEL = os.getenv("POLLINATIONS_EDIT_MODEL", "black-forest-labs/flux.1-kontext-pro")
IMAGE_API_BASE = "https://gen.pollinations.ai"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_RESPONSE_BYTES = 50 * 1024 * 1024


app = FastAPI(
    title="AFG AI",
    version="1.1.0",
    description="Multilingual AI assistant with Groq chat and Pollinations image generation and editing.",
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
        "image_edit_model": IMAGE_EDIT_MODEL,
    }


@app.post("/api/chat")
def chat(data: ChatRequest):
    if groq_client is None:
        raise HTTPException(status_code=503, detail="Chat service is not configured. Add GROQ_API_KEY in Render Environment.")
    system_prompt = PROMPTS.get(data.mode.lower(), PROMPTS["chat"])
    messages = [
        {"role": "system", "content": system_prompt + "\nReply in the language used by the user."}
    ]
    for item in data.history[-20:]:
        role = item.get("role")
        if role not in ("user", "assistant"):
            continue
        text = str(item.get("text", "")).strip()[:6000]
        if text:
            messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": data.message.strip()})
    try:
        result = groq_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.7,
            max_completion_tokens=4096,
        )
        answer = result.choices[0].message.content if result.choices else None
        if not answer:
            return {"answer": "The AI returned an empty response. Please try again."}
        return {"answer": answer}
    except Exception as exc:
        logger.exception("Groq chat request failed (model=%s).", CHAT_MODEL)
        raise HTTPException(status_code=502, detail="Chat request failed. Check Render logs for details.") from exc


def prepare_edit_output(raw: bytes) -> tuple[bytes, str, str]:
    """Detect the actual image format and losslessly re-encode edited output as PNG."""
    if not raw:
        raise ValueError("The image provider returned an empty image.")

    if len(raw) > MAX_IMAGE_RESPONSE_BYTES:
        raise ValueError("The edited image is larger than the 50 MB response limit.")

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for PNG output. Add 'Pillow' to requirements.txt and redeploy."
        ) from exc

    try:
        with Image.open(io.BytesIO(raw)) as source:
            source.load()
            # Keep the provider's original pixel dimensions; do not resize or crop.
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


@app.post("/api/image")
async def image(
    prompt: str = Form(..., min_length=1, max_length=4_000),
    image_file: Optional[UploadFile] = File(default=None),
):
    """Generate an image from text or edit an uploaded image using Pollinations."""
    if not POLLINATIONS_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Image service is not configured. Add POLLINATIONS_API_KEY in Render Environment.",
        )
    if POLLINATIONS_API_KEY.startswith("pk_"):
        raise HTTPException(
            status_code=503,
            detail="Use a secret Pollinations API key beginning with sk_. Keep the key private.",
        )

    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Please enter an image description or editing instruction.")

    try:
        if image_file is not None:
            # Only the uploaded-image editing path is post-processed.
            raw = await image_file.read(MAX_IMAGE_BYTES + 1)
            if len(raw) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
            mime_type = image_file.content_type or "application/octet-stream"
            if not mime_type.startswith("image/"):
                raise HTTPException(status_code=415, detail="Please upload a valid image file.")

            boundary = "----AFGAIImageEditBoundary7MA4YWxkTrZu0gW"
            filename = (image_file.filename or "upload.png").replace('"', '')
            fields = [
                ("prompt", prompt),
                ("model", IMAGE_EDIT_MODEL),
                ("size", "auto"),
                ("quality", "high"),
                ("response_format", "b64_json"),
            ]
            chunks = []
            for name, value in fields:
                chunks.append(
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode("utf-8")
                )
            chunks.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\nContent-Type: {mime_type}\r\n\r\n".encode("utf-8")
            )
            chunks.append(raw)
            chunks.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
            edit_request = Request(
                f"{IMAGE_API_BASE}/v1/images/edits",
                data=b"".join(chunks),
                headers={
                    "Authorization": f"Bearer {POLLINATIONS_API_KEY}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Accept": "application/json",
                    "User-Agent": "AFG-AI/1.3",
                },
                method="POST",
            )

            def fetch_edit():
                with urlopen(edit_request, timeout=240) as response:
                    response_bytes = response.read(MAX_IMAGE_RESPONSE_BYTES * 2 + 1)
                if len(response_bytes) > MAX_IMAGE_RESPONSE_BYTES * 2:
                    raise ValueError("Image provider response is too large.")
                payload = json.loads(response_bytes.decode("utf-8"))
                item = (payload.get("data") or [{}])[0]
                encoded = item.get("b64_json")
                if encoded:
                    return base64.b64decode(encoded, validate=True)
                image_url = item.get("url")
                if image_url:
                    image_request = Request(image_url, headers={"Accept": "image/*"}, method="GET")
                    with urlopen(image_request, timeout=120) as image_response:
                        result = image_response.read(MAX_IMAGE_RESPONSE_BYTES + 1)
                    if len(result) > MAX_IMAGE_RESPONSE_BYTES:
                        raise ValueError("Edited image is larger than the 50 MB limit.")
                    return result
                raise ValueError("Pollinations edit response did not contain an image.")

            raw_result = await asyncio.to_thread(fetch_edit)
            body, content_type, extension = prepare_edit_output(raw_result)
        else:
            # Text-to-image generation remains unchanged.
            query = urlencode({"model": IMAGE_MODEL, "width": 1024, "height": 1024})
            url = f"{IMAGE_API_BASE}/image/{quote(prompt, safe='')}?{query}"
            request = Request(
                url,
                headers={
                    "Authorization": f"Bearer {POLLINATIONS_API_KEY}",
                    "Accept": "image/*",
                    "User-Agent": "AFG-AI/1.3",
                },
                method="GET",
            )

            def fetch_image():
                with urlopen(request, timeout=180) as response:
                    content_type = response.headers.get("Content-Type", "image/jpeg").split(";", 1)[0]
                    body = response.read(MAX_IMAGE_RESPONSE_BYTES + 1)
                    if len(body) > MAX_IMAGE_RESPONSE_BYTES:
                        raise ValueError("Generated image is larger than the 50 MB limit.")
                    return body, content_type

            body, content_type = await asyncio.to_thread(fetch_image)
            if not content_type.startswith("image/"):
                logger.error("Pollinations returned a non-image response: %s", content_type)
                raise HTTPException(status_code=502, detail="Image provider did not return an image. Check the API key and model permissions.")
            extension = "png" if "png" in content_type else "webp" if "webp" in content_type else "jpg"

        return Response(
            content=body,
            media_type=content_type,
            headers={"Content-Disposition": f'inline; filename="afg-ai-result.{extension}"'},
        )
    except HTTPException:
        raise
    except HTTPError as exc:
        error_body = exc.read(2000).decode("utf-8", errors="replace")
        logger.warning("Pollinations API returned HTTP %s: %s", exc.code, error_body)
        if exc.code == 401:
            detail = "Pollinations API key is invalid. Check that POLLINATIONS_API_KEY is a valid secret key beginning with sk_."
        elif exc.code == 402:
            detail = "Pollinations account/key has insufficient balance or budget for this image model. Check the account balance and key budget."
        elif exc.code == 429:
            detail = "Pollinations rate limit reached. Please wait and try again."
        elif exc.code == 400:
            detail = "The selected image model may not support editing, or the image request is invalid. Check Render logs."
        else:
            detail = f"Pollinations image request failed (HTTP {exc.code}). Check Render logs."
        raise HTTPException(status_code=502, detail=detail) from exc
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        logger.exception("Pollinations image request failed (generate_model=%s, edit_model=%s).", IMAGE_MODEL, IMAGE_EDIT_MODEL)
        raise HTTPException(status_code=502, detail="Could not process the image. Check Render logs and ensure Pillow is installed.") from exc
    except Exception as exc:
        logger.exception("Unexpected Pollinations image error (generate_model=%s, edit_model=%s).", IMAGE_MODEL, IMAGE_EDIT_MODEL)
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
