"""Entry point: ``python main.py`` starts the FastAPI backend with uvicorn."""

from __future__ import annotations

import uvicorn

from app.api.app import create_app
from app.core.config import get_settings

app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, log_config=None)
