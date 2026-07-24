"""Main FastAPI Application initialization."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uvicorn

from api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Trading Bot API Server...")
    yield
    # Shutdown
    logger.info("Shutting down Trading Bot API Server...")

import os

app = FastAPI(title="HMM Trading Bot API", version="1.0.0", lifespan=lifespan)

# Allow dashboard cross-origin requests
dashboard_origin = os.environ.get("DASHBOARD_URL", "http://localhost:8501")
allowed_origins = [dashboard_origin] if os.environ.get("ENV") != "development" else ["*"]

from starlette.middleware.base import BaseHTTPMiddleware
from security.api_security import rate_limit_middleware

app.add_middleware(BaseHTTPMiddleware, dispatch=rate_limit_middleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

def start_api(host: str = "127.0.0.1", port: int = 8000):
    """Entry point to run the API programmatically."""
    logger.info(f"Launching Uvicorn API on {host}:{port}")
    uvicorn.run("api.main:app", host=host, port=port, log_level="warning")

if __name__ == "__main__":
    start_api()
