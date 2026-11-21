
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

# ==============================
# AI MODEL CONFIGURATION
# ==============================

API_KEY = os.getenv("GEMINI_API_KEY")

TEXT_MODEL = os.getenv(
    "GEMINI_TEXT_MODEL",
    "gemini-3.8-flash"
)

IMAGE_MODEL = os.getenv(
    "GEMINI_IMAGE_MODEL",
    "gemini-2.5-flash-image"
)

# ==============================
# FASTAPI APPLICATION
# ==============================

app = FastAPI(
    title="AFG AI",
    version="0.2.0",
    description="Advanced multilingual AI assistant with text and image capabilities."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=API_KEY) if API_KEY else None


# ==============================
# REQUEST MODELS
# ==============================

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    mode: str = "chat"
    history: list[dict] = Field(default_factory=list, max_length=20)


# ==============================
# HEALTH CHECK
# ==============================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "configured": client is not None,
        "text_model": TEXT_MODEL,
        "image_model": IMAGE_MODEL,
        "version": "0.2.0"
    }


# ==============================
# CHAT API
# ==============================

@app.post("/api/chat")
def chat(data: ChatRequest):

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="AI service is not configured. Add GEMINI_API_KEY."
        )

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
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=text)]
                )
            )

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text=system
                    + "\nReply in the language used by the user."
                    + "\n\n"
                    + data.message
                )
            ]
        )
    )

    try:

        result = client.models.generate_content(
            model=TEXT_MODEL,
            contents=contents
        )

        answer = result.text

        return {
            "answer": answer or "The model returned no text.",
            "model": TEXT_MODEL
        }

    except Exception as exc:

        print(f"Gemini chat error ({TEXT_MODEL}): {exc}")

        raise HTTPException(
            status_code=502,
            detail=f"Text AI request failed using model {TEXT_MODEL}. Check model availability and API quota."
        ) from exc


# ==============================
# IMAGE GENERATION & EDITING
# ==============================

@app.post("/api/image")
async def image(
    prompt: str = Form(..., min_length=1, max_length=4000),
    image_file: Optional[UploadFile] = File(default=None)
):

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="AI service is not configured. Add GEMINI_API_KEY."
        )

    parts = [types.Part(text=prompt)]

    # Image editing if an image is uploaded
    if image_file is not None:

        raw = await image_file.read()

        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be 10 MB or smaller."
            )

        mime = image_file.content_type or "application/octet-stream"

        if not mime.startswith("image/"):
            raise HTTPException(
                status_code=415,
                detail="Please upload an image file."
            )

        parts.insert(
            0,
            types.Part(
                inline_data=types.Blob(
                    mime_type=mime,
                    data=raw
                )
            )
        )

    try:

        result = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=parts
                )
            ],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"]
            )
        )

        for candidate in (result.candidates or []):

            candidate_content = candidate.content

            if not candidate_content:
                continue

            for part in (candidate_content.parts or []):

                if part.inline_data and part.inline_data.data:

                    mime = part.inline_data.mime_type or "image/png"

                    return Response(
                        content=part.inline_data.data,
                        media_type=mime,
                        headers={
                            "Content-Disposition": 'inline; filename="afg-ai-result.png"'
                        }
                    )

        raise HTTPException(
            status_code=502,
            detail="No image was returned. Check image model access or try another prompt."
        )

    except HTTPException:
        raise

    except Exception as exc:

        print(f"Gemini image error ({IMAGE_MODEL}): {exc}")

        raise HTTPException(
            status_code=502,
            detail=f"Image request failed using {IMAGE_MODEL}. Check image-model access and API quota."
        ) from exc


# ==============================
# FRONTEND ROUTES
# ==============================

@app.get("/")
def index():
    return FileResponse("frontend/index.html")


@app.get("/style.css")
def css():
    return FileResponse(
        "frontend/style.css",
        media_type="text/css"
    )


@app.get("/app.js")
def js():
    return FileResponse(
        "frontend/app.js",
        media_type="application/javascript"
    )
