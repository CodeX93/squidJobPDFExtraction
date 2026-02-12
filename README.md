PDF to JSON Extraction API
==========================

Quick notes for deploying to Render

- Start Command (Render web service): use the `Procfile` or set Start Command to:

  `uvicorn script:app --host 0.0.0.0 --port $PORT`

- Recommended (Procfile):

  `web: gunicorn -k uvicorn.workers.UvicornWorker script:app --bind 0.0.0.0:$PORT --workers 4`

- Requirements: see `requirements.txt`.

- Environment variables:
  - `GEMINI_API_KEY` — set this as a secret in Render for Gemini access.

- Local run for testing:

```bash
pip install -r requirements.txt
uvicorn script:app --reload --host 0.0.0.0 --port 8000
```
