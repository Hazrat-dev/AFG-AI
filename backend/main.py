"""AFG AI backend - Groq for chat, Google Gemini for image generation/editing."""
import logging
import os
from typing import Optional


from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from groq import Groq


load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("afg_ai")


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
MAX_IMAGE_BYTES = 10 * 1024 * 1024


app = FastAPI(
    title="AFG AI",
    version="1.1.0",
    description="Multilingual AI assistant with Groq chat and Gemini image generation/editing.",
)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


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
        "image_configured": gemini_client is not None,
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
    """Generate an image or edit an uploaded image with Gemini."""
    if gemini_client is None:
        raise HTTPException(status_code=503, detail="Image service is not configured. Add GEMINI_API_KEY in Render Environment.")
    prompt = prompt.strip()
    parts = [types.Part(text=prompt)]
    if image_file is not None:
        raw = await image_file.read(MAX_IMAGE_BYTES + 1)
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
        mime_type = image_file.content_type or "application/octet-stream"
        if not mime_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="Please upload a valid image file.")
        parts.insert(0, types.Part(inline_data=types.Blob(mime_type=mime_type, data=raw)))
    try:
        result = gemini_client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
        for candidate in result.candidates or []:
            candidate_content = getattr(candidate, "content", None)
            for part in getattr(candidate_content, "parts", []) or []:
                inline_data = getattr(part, "inline_data", None)
                if inline_data and inline_data.data:
                    return Response(
                        content=inline_data.data,
                        media_type=inline_data.mime_type or "image/png",
                        headers={"Content-Disposition": 'inline; filename="afg-ai-result.png"'},
                    )
        logger.warning("Gemini image model returned no image (model=%s).", IMAGE_MODEL)
        raise HTTPException(status_code=502, detail="The model returned no image. Try another prompt.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Gemini image request failed (model=%s).", IMAGE_MODEL)
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
