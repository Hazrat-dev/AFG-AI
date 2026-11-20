# AFG AI — starter web app

A mobile-friendly, English-first AI workspace with multilingual chat and an image studio for text-to-image generation and photo editing.

## Included in this starter

- Responsive web interface for phones and computers
- Gemini text chat with general, coding, translation, writing, and analysis modes
- Image generation from a text prompt
- Image editing by uploading a photo and describing the changes
- FastAPI backend; the API key stays on the server
- Render deployment blueprint

## Important limitations

- This is a university-project starter, not a finished production service.
- Image generation/editing requires access to the configured Gemini image model and available API quota. Free access and limits depend on Google's current account, region, and model policies.
- The Documents page is currently a placeholder; PDF question-answering is not implemented yet.
- Chat history currently stays in the browser session and is not saved to a database.
- Do not upload private, sensitive, or identifying documents/photos to a public demo.
- Render's free web services sleep after 15 minutes without traffic, and local files are ephemeral. See https://render.com/docs/free.

## Deploy using only an Android phone

1. Download and extract `AFG-AI-starter.zip` on your phone.
2. In GitHub's mobile website, create a new repository named `AFG-AI`.
3. Upload the extracted project files and folders to the repository. Keep the folder structure:
   - `backend/main.py`
   - `frontend/index.html`
   - `frontend/style.css`
   - `frontend/app.js`
   - `requirements.txt`
   - `render.yaml`
4. Create/sign in to a Render account and connect GitHub.
5. In Render, choose **New → Blueprint** if available, or create a **Web Service** from the repository.
6. For a Web Service, use:
   - Runtime: Python 3
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - Plan: Free
7. In the Render service's **Environment** settings, add:
   - `GEMINI_API_KEY` = your own API key
   - `GEMINI_TEXT_MODEL` = `gemini-2.5-flash`
   - `GEMINI_IMAGE_MODEL` = `gemini-2.5-flash-image`
8. Deploy. Open the generated `onrender.com` URL.

Never put your API key in HTML, JavaScript, GitHub, screenshots, or messages. If it is accidentally published, revoke it and create a new one.

## Local development

Use Python 3.10+:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux/Termux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your API key
uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000 and http://127.0.0.1:8000/docs.

## Model configuration

The model IDs are environment variables, so they can be changed in Render without editing code. Confirm that the selected models are available to your Google AI Studio project and region before deployment.
