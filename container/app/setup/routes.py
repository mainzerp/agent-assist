"""Setup wizard routes."""

from __future__ import annotations

import logging
import secrets
from pathlib import Path

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.db.repository import (
    AdminAccountRepository,
    SetupStateRepository,
    SettingsRepository,
)
from app.security.encryption import store_secret, retrieve_secret
from app.security.hashing import hash_password
from app.ha_client.rest import test_ha_connection

logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

router = APIRouter(prefix="/setup", tags=["setup"])

STEP_ORDER = [
    "admin_password",
    "ha_connection",
    "container_api_key",
    "llm_providers",
    "review_complete",
]


@router.get("/", response_class=HTMLResponse)
async def setup_index(request: Request):
    """Redirect to the first incomplete step."""
    steps = await SetupStateRepository.get_all_steps()
    step_map = {s["step"]: s["completed"] for s in steps}
    for i, step_name in enumerate(STEP_ORDER):
        if not step_map.get(step_name, False):
            return RedirectResponse(url=f"/setup/step/{i + 1}", status_code=302)
    return RedirectResponse(url="/dashboard/", status_code=302)


@router.get("/step/{step_num}", response_class=HTMLResponse)
async def render_step(request: Request, step_num: int):
    """Render the appropriate step template."""
    steps = await SetupStateRepository.get_all_steps()
    step_map = {s["step"]: s["completed"] for s in steps}
    context = {
        "step_num": step_num,
        "total_steps": len(STEP_ORDER),
        "steps": step_map,
    }
    return templates.TemplateResponse(request, f"step{step_num}.html", context=context)


@router.post("/step/1", response_class=HTMLResponse)
async def save_admin_password(
    request: Request,
    username: str = Form("admin"),
    password: str = Form(...),
):
    """Step 1: Create admin account with bcrypt-hashed password."""
    hashed = hash_password(password)
    await AdminAccountRepository.create(username, hashed)
    await SetupStateRepository.set_step_completed("admin_password")
    return RedirectResponse(url="/setup/step/2", status_code=303)


@router.post("/step/2", response_class=HTMLResponse)
async def save_ha_connection(
    request: Request,
    ha_url: str = Form(...),
    ha_token: str = Form(...),
):
    """Step 2: Save HA URL and token (Fernet-encrypted)."""
    await SettingsRepository.set("ha_url", ha_url, "string", "ha", "Home Assistant URL")
    from app.ha_client.auth import set_ha_token
    await set_ha_token(ha_token)
    await SetupStateRepository.set_step_completed("ha_connection")
    return RedirectResponse(url="/setup/step/3", status_code=303)


@router.post("/step/3", response_class=HTMLResponse)
async def generate_api_key(request: Request):
    """Step 3: Auto-generate container API key, store encrypted, show once."""
    api_key = secrets.token_urlsafe(32)
    await store_secret("container_api_key", api_key)
    await SetupStateRepository.set_step_completed("container_api_key")
    steps = await SetupStateRepository.get_all_steps()
    step_map = {s["step"]: s["completed"] for s in steps}
    return templates.TemplateResponse(request, "step3.html", context={
        "step_num": 3,
        "total_steps": len(STEP_ORDER),
        "steps": step_map,
        "api_key": api_key,
        "generated": True,
    })


@router.post("/step/4", response_class=HTMLResponse)
async def save_llm_keys(
    request: Request,
    openrouter_key: str = Form(""),
    groq_key: str = Form(""),
    ollama_url: str = Form(""),
):
    """Step 4: Save LLM provider keys (Fernet-encrypted)."""
    if openrouter_key:
        await store_secret("openrouter_api_key", openrouter_key)
    if groq_key:
        await store_secret("groq_api_key", groq_key)
    if ollama_url:
        await SettingsRepository.set(
            "ollama_base_url", ollama_url, "string", "llm", "Ollama API URL"
        )
    await SetupStateRepository.set_step_completed("llm_providers")
    return RedirectResponse(url="/setup/step/5", status_code=303)


@router.post("/step/5", response_class=HTMLResponse)
async def complete_setup(request: Request):
    """Step 5: Mark setup complete and trigger post-setup initialization."""
    await SetupStateRepository.set_step_completed("review_complete")
    logger.info("Setup wizard completed, triggering post-setup initialization")
    return RedirectResponse(url="/dashboard/", status_code=303)


@router.post("/test/ha")
async def test_ha_endpoint(ha_url: str = Form(...), ha_token: str = Form(...)):
    """Test HA connection with provided URL and token."""
    success = await test_ha_connection(ha_url, ha_token)
    return {"success": success}


@router.post("/test/llm")
async def test_llm_endpoint(provider: str = Form(...), api_key: str = Form(...)):
    """Test LLM provider with a small completion request."""
    try:
        import litellm
        if provider == "groq":
            model = "groq/llama-3.1-8b-instant"
        elif provider == "openrouter":
            model = "openrouter/openai/gpt-4o-mini"
        elif provider == "ollama":
            model = "ollama/llama3"
        else:
            return {"success": False, "error": f"Unknown provider: {provider}"}

        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key=api_key,
            max_tokens=10,
        )
        return {"success": True, "response": response.choices[0].message.content}
    except Exception as e:
        return {"success": False, "error": str(e)}
