import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import RedirectResponse as StarletteRedirect

from app.db.repository import SetupStateRepository

logger = logging.getLogger(__name__)


async def _safe_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning("HTTP %d: %s %s", exc.status_code, request.method, request.url.path)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=getattr(exc, "headers", None))


async def _safe_generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s %s", request.method, request.url.path, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def apply_auth_dependencies(app: FastAPI) -> None:
    app.add_exception_handler(HTTPException, _safe_http_exception_handler)
    app.add_exception_handler(Exception, _safe_generic_exception_handler)


class SetupRedirectMiddleware(BaseHTTPMiddleware):
    """Redirect all routes to /setup/ if setup is not complete."""

    ALLOWED_PREFIXES = ("/setup", "/api/health", "/static")

    def __init__(self, app, *args, **kwargs):
        super().__init__(app, *args, **kwargs)
        self._setup_complete: bool | None = None

    async def dispatch(self, request: StarletteRequest, call_next):
        # Cache the completion state -- once complete, never check again
        if self._setup_complete is None or not self._setup_complete:
            self._setup_complete = await SetupStateRepository.is_complete()

        if not self._setup_complete:
            path = request.url.path
            if not any(path.startswith(p) for p in self.ALLOWED_PREFIXES):
                return StarletteRedirect(url="/setup/", status_code=302)

        return await call_next(request)
