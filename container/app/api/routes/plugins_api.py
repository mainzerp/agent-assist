"""Plugin management API endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from app.db.repository import PluginRepository
from app.security.auth import require_admin_session

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin/plugins",
    tags=["admin-plugins"],
    dependencies=[Depends(require_admin_session)],
)


@router.get("")
async def list_plugins(request: Request) -> list[dict[str, Any]]:
    """List all installed plugins with loaded status."""
    db_plugins = await PluginRepository.list_all()
    plugin_loader = getattr(request.app.state, "plugin_loader", None)
    loaded_names = set()
    if plugin_loader:
        loaded_names = set(plugin_loader.loaded_plugins.keys())

    for plugin in db_plugins:
        plugin["loaded"] = plugin["name"] in loaded_names
    return db_plugins


@router.post("/{name}/enable")
async def enable_plugin(request: Request, name: str) -> dict[str, Any]:
    """Enable a plugin."""
    plugin_loader = getattr(request.app.state, "plugin_loader", None)
    if not plugin_loader:
        raise HTTPException(status_code=503, detail="Plugin system not initialized")

    success = await plugin_loader.enable_plugin(name)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to enable plugin")
    return {"name": name, "enabled": True}


@router.post("/{name}/disable")
async def disable_plugin(request: Request, name: str) -> dict[str, Any]:
    """Disable a plugin."""
    plugin_loader = getattr(request.app.state, "plugin_loader", None)
    if not plugin_loader:
        raise HTTPException(status_code=503, detail="Plugin system not initialized")

    success = await plugin_loader.disable_plugin(name)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to disable plugin")
    return {"name": name, "enabled": False}
