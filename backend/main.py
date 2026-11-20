import os
import base64
from io import BytesIO
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

app = FastAPI(title="AFG AI", version="0.1.0",
              description="Multilingual AI assistant with image generation and editing.")

# Same-origin deployment: frontend and API are served by this FastAPI app.
if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    client = None

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    mode: str = "chat"
    history: list[dict] = Field(default_factory=list, max_length=20)

@app.get("/api/health")
def health():
    return {"status": "ok", "configured": client is not None}

@app.post("/api/chat")
def chat(data: ChatRequest):
    if client is None:
        raise HTTPException(503, "AI service is not configured. Add GEMINI_API_KEY.")
    modes = {
        "chat": "You are AFG AI, a helpful multilingual assistant.",
        "coding": "You are an expert programming assistant. Give safe, clear explanations.",
        "translator": "You are a professional translator. Preserve meaning and tone.",
        "writer": "You are a professional writing assistant.",
        "analysis": "You are a careful analytical assistant. Distinguish facts from uncertainty."
    }
    system = modes.get(data.mode, modes["chat"])
    contents = []
    for item in data.history[-20:]:
        role = "model" if item.get("role") == "assistant" else "user"
        text = str(item.get("text", ""))[:6000]
        if text:
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    contents.append(types.Content(role="user", parts=[
        types.Part(text=system + "\nReply in the language used by the user.\n\n" + data.message)
    ]))
    try:
        result = client.models.generate_content(model=TEXT_MODEL, contents=contents)
        return {"answer": result.text or "The model returned no text."}
    except Exception as exc:
        raise HTTPException(502, "Text AI request failed. Check model availability and API quota.") from exc

@app.post("/api/image")
async def image(
    prompt: str = Form(..., min_length=1, max_length=4000),
    image_file: Optional[UploadFile] = File(default=None)
):
    """Generate an image from text, or edit an uploaded image using a prompt."""
    if client is None:
        raise HTTPException(503, "AI service is not configured. Add GEMINI_API_KEY.")
    parts = [types.Part(text=prompt)]
    if image_file is not None:
        raw = await image_file.read()
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(413, "Image must be 10 MB or smaller.")
        mime = image_file.content_type or "application/octet-stream"
        if not mime.startswith("image/"):
            raise HTTPException(415, "Please upload an image file.")
        parts.insert(0, types.Part(inline_data=types.Blob(mime_type=mime, data=raw)))
    try:
        result = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
        )
        for candidate in (result.candidates or []):
            for part in (candidate.content.parts or []):
                if part.inline_data and part.inline_data.data:
                    mime = part.inline_data.mime_type or "image/png"
                    return Response(
                        content=part.inline_data.data,
                        media_type=mime,
                        headers={"Content-Disposition": 'inline; filename="afg-ai-result.png"'}
                    )
        raise HTTPException(502, "No image was returned. Try a different prompt or model.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, "Image request failed. Check image-model access and API quota.") from exc

@app.get("/")
def index():
    return FileResponse("frontend/index.html")

@app.get("/style.css")
def css():
    return FileResponse("frontend/style.css", media_type="text/css")

@app.get("/app.js")
def js():
    return FileResponse("frontend/app.js", media_type="application/javascript")
