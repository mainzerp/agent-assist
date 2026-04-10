"""Admin dashboard routes."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.security.auth import (
    authenticate_admin,
    create_session_cookie,
    require_admin_session_redirect,
    SESSION_COOKIE_NAME,
)

logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/", response_class=HTMLResponse)
async def dashboard_index(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Dashboard home page."""
    return templates.TemplateResponse(request, "overview.html")


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str | None = None):
    """Login page."""
    return templates.TemplateResponse(request, "login.html", context={"title": "Login", "error": error})


@router.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    username: str = Form("admin"),
    password: str = Form(...),
):
    """Handle login form submission."""
    session_data = await authenticate_admin(username, password)
    if session_data is None:
        return templates.TemplateResponse(
            request,
            "login.html",
            context={"title": "Login", "error": "Invalid credentials"},
        )
    cookie_value = create_session_cookie(session_data)
    response = RedirectResponse(url="/dashboard/", status_code=303)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        cookie_value,
        httponly=True,
        samesite="lax",
        max_age=86400,
        # secure=True should be added in production behind HTTPS
    )
    return response


@router.get("/logout")
async def logout():
    """Clear session and redirect to login."""
    response = RedirectResponse(url="/dashboard/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response


@router.get("/agents", response_class=HTMLResponse)
async def agents_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Agent configuration page."""
    return templates.TemplateResponse(request, "agents.html")


@router.get("/system-health", response_class=HTMLResponse)
async def system_health_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """System health monitoring page."""
    return templates.TemplateResponse(request, "system_health.html")


@router.get("/rewrite", response_class=HTMLResponse)
async def rewrite_config_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Rewrite agent configuration page."""
    return templates.TemplateResponse(request, "rewrite_config.html")


@router.get("/health", response_class=RedirectResponse)
async def health_redirect():
    """Redirect to API health endpoint."""
    return RedirectResponse(url="/api/health")


@router.get("/conversations", response_class=HTMLResponse)
async def conversations_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Conversation history page."""
    return templates.TemplateResponse(request, "conversations.html")


@router.get("/cache", response_class=HTMLResponse)
async def cache_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Cache management page."""
    return templates.TemplateResponse(request, "cache.html")


@router.get("/entity-index", response_class=HTMLResponse)
async def entity_index_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Entity index page."""
    return templates.TemplateResponse(request, "entity_index.html")


@router.get("/analytics", response_class=HTMLResponse)
async def analytics_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Analytics dashboard page."""
    return templates.TemplateResponse(request, "analytics.html")


@router.get("/traces", response_class=HTMLResponse)
async def traces_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Request traces page."""
    return templates.TemplateResponse(request, "traces.html")


@router.get("/mcp-servers", response_class=HTMLResponse)
async def mcp_servers_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """MCP server management page."""
    return templates.TemplateResponse(request, "mcp_servers.html")


@router.get("/custom-agents", response_class=HTMLResponse)
async def custom_agents_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Custom agents builder page."""
    return templates.TemplateResponse(request, "custom_agents.html")


@router.get("/entity-visibility", response_class=HTMLResponse)
async def entity_visibility_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Entity visibility management page."""
    return templates.TemplateResponse(request, "entity_visibility.html")


@router.get("/presence", response_class=HTMLResponse)
async def presence_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Presence detection status page."""
    return templates.TemplateResponse(request, "presence.html")


@router.get("/plugins", response_class=HTMLResponse)
async def plugins_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Plugin management page."""
    return templates.TemplateResponse(request, "plugins.html")


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Unified settings page for all advanced configuration."""
    return templates.TemplateResponse(request, "settings.html")
