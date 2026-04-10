"""Health check endpoint."""

import logging

from fastapi import APIRouter

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health():
    """Return container health status."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "log_level": settings.log_level,
    }
