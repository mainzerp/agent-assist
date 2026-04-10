"""Admin REST API endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.security.auth import require_admin_session
from app.db.repository import AgentConfigRepository, SettingsRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin_session)])

# The registry is set by main.py during startup
_registry = None


def set_registry(reg) -> None:
    """Called by main.py to inject the A2A registry."""
    global _registry
    _registry = reg


@router.get("/agents")
async def list_agents():
    """List all registered agents."""
    agents = await _registry.list_agents()
    result = []
    for a in agents:
        card = a.model_dump()
        config = await AgentConfigRepository.get(a.agent_id)
        if config:
            card.update(config)
        result.append(card)
    return {"agents": result}


@router.get("/settings")
async def get_settings():
    """Get all settings grouped by category."""
    rows = await SettingsRepository.get_all()
    grouped: dict[str, list] = {}
    for row in rows:
        cat = row.get("category", "general")
        grouped.setdefault(cat, []).append(row)
    return {"settings": grouped}


@router.put("/settings")
async def update_settings(payload: dict):
    """Update multiple settings. Payload: {key: value, ...}."""
    items = payload.get("items", payload)
    if isinstance(items, dict):
        for key, value in items.items():
            if key == "items":
                continue
            await SettingsRepository.set(key, str(value))
    return {"status": "ok"}


@router.put("/settings/{key}")
async def update_single_setting(key: str, payload: dict):
    """Update a single setting by key."""
    value = payload.get("value")
    if value is None:
        return {"status": "error", "detail": "Missing value"}
    value_type = payload.get("value_type", "string")
    category = payload.get("category", "general")
    description = payload.get("description")
    await SettingsRepository.set(key, str(value), value_type, category, description)
    return {"status": "ok", "key": key}
