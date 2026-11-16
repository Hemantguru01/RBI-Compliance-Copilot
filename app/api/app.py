"""FastAPI application factory."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    IndexCorruptedError,
    IndexNotReadyError,
    LLMConfigurationError,
    LLMGenerationError,
    PDFExtractionError,
)
from app.core.logging import get_logger, setup_logging
from app.services.container import AppContainer

logger = get_logger(__name__)

DESCRIPTION = """
Retrieval-augmented question answering over Reserve Bank of India directions and circulars.

Every answer is grounded in retrieved passages and returns verifiable citations
(document title, circular number, page number and supporting excerpt). When the
indexed documents do not support an answer, the API says so instead of guessing.
"""


def create_app(settings: Settings | None = None, container: AppContainer | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.log_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if container is None:
            app.state.container = AppContainer(settings)
        else:
            app.state.container = container
        app.state.container.startup()
        logger.info("API ready", extra={"documents": len(app.state.container.store.documents)})
        yield

    app = FastAPI(title="RBI Compliance Copilot API", version="1.0.0", description=DESCRIPTION, lifespan=lifespan)

    @app.middleware("http")
    async def timing(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
        logger.info("request", extra={"method": request.method, "path": request.url.path,
                                      "status": response.status_code, "ms": round(elapsed, 1)})
        return response

    def _handler(status_code: int):
        async def handle(request: Request, exc: Exception) -> JSONResponse:
            logger.warning("Request failed", extra={"path": request.url.path, "error": str(exc), "type": type(exc).__name__})
            return JSONResponse(status_code=status_code, content={"detail": str(exc), "error_type": type(exc).__name__})
        return handle

    app.add_exception_handler(IndexNotReadyError, _handler(503))
    app.add_exception_handler(IndexCorruptedError, _handler(503))
    app.add_exception_handler(LLMConfigurationError, _handler(503))
    app.add_exception_handler(LLMGenerationError, _handler(502))
    app.add_exception_handler(PDFExtractionError, _handler(422))

    app.include_router(router)
    return app
