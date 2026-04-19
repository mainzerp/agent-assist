"""Admin dashboard routes."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.security.auth import (
    authenticate_admin,
    create_session_cookie,
    ensure_csrf_token,
    require_admin_session_redirect,
    set_csrf_cookie,
    verify_csrf,
    SESSION_COOKIE_NAME,
)
from app.config import settings as app_settings

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
    token = ensure_csrf_token(request)
    response = templates.TemplateResponse(
        request,
        "login.html",
        context={"title": "Login", "error": error, "csrf_token": token},
    )
    set_csrf_cookie(response, token)
    return response


@router.post(
    "/login",
    response_class=HTMLResponse,
    dependencies=[Depends(verify_csrf)],
)
async def login_submit(
    request: Request,
    username: str = Form("admin"),
    password: str = Form(...),
):
    """Handle login form submission."""
    session_data = await authenticate_admin(username, password)
    if session_data is None:
        token = ensure_csrf_token(request)
        response = templates.TemplateResponse(
            request,
            "login.html",
            context={"title": "Login", "error": "Invalid credentials", "csrf_token": token},
        )
        set_csrf_cookie(response, token)
        return response
    cookie_value = create_session_cookie(session_data)
    response = RedirectResponse(url="/dashboard/", status_code=303)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        cookie_value,
        httponly=True,
        samesite="lax",
        max_age=86400,
        secure=app_settings.cookie_secure,
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


@router.get("/health", response_class=RedirectResponse)
async def health_redirect():
    """Redirect to API health endpoint."""
    return RedirectResponse(url="/api/health")


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Chat test interface."""
    return templates.TemplateResponse(request, "chat.html")

@router.get("/personality", response_class=HTMLResponse)
async def personality_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Personality configuration page."""
    return templates.TemplateResponse(request, "personality.html")

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
    """Redirect to entity index (entity visibility merged into entity index)."""
    from starlette.responses import RedirectResponse
    agent = request.query_params.get("agent", "")
    url = "/dashboard/entity-index"
    if agent:
        url += f"?agent={agent}"
    return RedirectResponse(url=url, status_code=301)


@router.get("/presence", response_class=HTMLResponse)
async def presence_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Presence detection status page."""
    return templates.TemplateResponse(request, "presence.html")


@router.get("/timers", response_class=HTMLResponse)
async def timers_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Timers & alarms dashboard page."""
    return templates.TemplateResponse(request, "timers.html")


@router.get("/plugins", response_class=HTMLResponse)
async def plugins_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Plugin management page."""
    return templates.TemplateResponse(request, "plugins.html")


@router.get("/send-devices", response_class=HTMLResponse)
async def send_devices_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Send device mappings management page."""
    return templates.TemplateResponse(request, "send_devices.html")


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(
    request: Request,
    _session: dict = Depends(require_admin_session_redirect),
):
    """Unified settings page for all advanced configuration."""
    return templates.TemplateResponse(request, "settings.html")
