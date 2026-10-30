"""Application entry point and lifecycle management."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import router
from app.models import init_db
from app.services import initialize_services, shutdown_services

# Prepare shared resources once when the app starts, then clean them up when it stops.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Make sure the jobs table exists before any request tries to use it.
    init_db()
    await initialize_services()
    try:
        yield
    finally:
        await shutdown_services()

# Keeping app creation here gives Uvicorn one clear place to import the application from.
app = FastAPI(
    title="Audio Transcription API",
    version="1.0.0",
    lifespan=lifespan,
)
app.openapi_version = "3.0.3"
app.include_router(router)
