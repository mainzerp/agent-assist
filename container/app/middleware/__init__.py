"""Middleware and FastAPI dependency wrappers for authentication."""

from app.middleware.auth import apply_auth_dependencies

__all__ = ["apply_auth_dependencies"]
