"""AFG AI backend - Groq for chat and Pollinations for image generation."""
import asyncio
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
IMAGE_API_BASE = "https://gen.pollinations.ai"
MAX_IMAGE_BYTES = 10 * 1024 * 1024


app = FastAPI(
    title="AFG AI",
    version="1.1.0",
    description="Multilingual AI assistant with Groq chat and Pollinations image generation.",
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


@app.post("/api/image")
async def image(
    prompt: str = Form(..., min_length=1, max_length=4_000),
    image_file: Optional[UploadFile] = File(default=None),
):
    """Generate an image with Pollinations. Uploaded-image editing is not enabled in this version."""
    if not POLLINATIONS_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Image service is not configured. Add POLLINATIONS_API_KEY in Render Environment.",
        )
    if POLLINATIONS_API_KEY.startswith("pk_"):
        raise HTTPException(
            status_code=503,
            detail="This endpoint needs a secret Pollinations key beginning with sk_, not an app key beginning with pk_. Keep the key private.",
        )
    if image_file is not None:
        raise HTTPException(
            status_code=501,
            detail="Text-to-image is enabled. Image editing will be added separately; remove the uploaded image and try a text prompt.",
        )

    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Please enter an image description.")

    query = urlencode({"model": IMAGE_MODEL, "width": 1024, "height": 1024})
    url = f"{IMAGE_API_BASE}/image/{quote(prompt, safe='')}?{query}"
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {POLLINATIONS_API_KEY}",
            "Accept": "image/*",
            "User-Agent": "AFG-AI/1.2",
        },
        method="GET",
    )

    def fetch_image():
        with urlopen(request, timeout=180) as response:
            content_type = response.headers.get("Content-Type", "image/jpeg").split(";", 1)[0]
            body = response.read(MAX_IMAGE_BYTES * 5 + 1)
            if len(body) > MAX_IMAGE_BYTES * 5:
                raise ValueError("Generated image is larger than the 50 MB limit.")
            return body, content_type

    try:
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
            detail = "Pollinations account/key has no remaining image-generation budget. Check your account balance and key budget."
        elif exc.code == 429:
            detail = "Pollinations rate limit reached. Please wait and try again."
        else:
            detail = f"Pollinations image request failed (HTTP {exc.code}). Check Render logs."
        raise HTTPException(status_code=502, detail=detail) from exc
    except (URLError, TimeoutError, ValueError) as exc:
        logger.exception("Pollinations image request failed (model=%s).", IMAGE_MODEL)
        raise HTTPException(status_code=502, detail="Could not generate the image. Check Render logs and try again.") from exc
    except Exception as exc:
        logger.exception("Unexpected Pollinations image error (model=%s).", IMAGE_MODEL)
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
