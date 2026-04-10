"""Custom agents CRUD API endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.security.auth import require_admin_session
from app.db.repository import CustomAgentRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin/custom-agents",
    tags=["admin-custom-agents"],
    dependencies=[Depends(require_admin_session)],
)


class CustomAgentCreate(BaseModel):
    name: str
    description: str = ""
    system_prompt: str
    model_override: str | None = None
    mcp_tools: list[dict[str, str]] | None = None
    entity_visibility: list[dict[str, str]] | None = None
    intent_patterns: list[str] | None = None


class CustomAgentUpdate(BaseModel):
    description: str | None = None
    system_prompt: str | None = None
    model_override: str | None = None
    mcp_tools: list[dict[str, str]] | None = None
    entity_visibility: list[dict[str, str]] | None = None
    intent_patterns: list[str] | None = None
    enabled: bool | None = None


@router.get("")
async def list_custom_agents() -> list[dict[str, Any]]:
    """List all custom agents."""
    return await CustomAgentRepository.list_all()


@router.post("", status_code=201)
async def create_custom_agent(
    request: Request, body: CustomAgentCreate
) -> dict[str, Any]:
    """Create a new custom agent."""
    existing = await CustomAgentRepository.get(body.name)
    if existing:
        raise HTTPException(status_code=409, detail="Agent with this name already exists")

    await CustomAgentRepository.create(
        name=body.name,
        system_prompt=body.system_prompt,
        description=body.description,
        model_override=body.model_override,
        mcp_tools=body.mcp_tools,
        entity_visibility=body.entity_visibility,
        intent_patterns=body.intent_patterns,
    )
    custom_loader = request.app.state.custom_loader
    await custom_loader.reload()
    return await CustomAgentRepository.get(body.name)


@router.get("/{name}")
async def get_custom_agent(name: str) -> dict[str, Any]:
    """Get a single custom agent."""
    agent = await CustomAgentRepository.get(name)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.put("/{name}")
async def update_custom_agent(
    request: Request, name: str, body: CustomAgentUpdate
) -> dict[str, Any]:
    """Update a custom agent."""
    existing = await CustomAgentRepository.get(name)
    if not existing:
        raise HTTPException(status_code=404, detail="Agent not found")

    update_data = body.model_dump(exclude_none=True)
    if update_data:
        await CustomAgentRepository.update(name, **update_data)

    custom_loader = request.app.state.custom_loader
    await custom_loader.reload()
    return await CustomAgentRepository.get(name)


@router.delete("/{name}")
async def delete_custom_agent(request: Request, name: str) -> dict[str, str]:
    """Delete a custom agent."""
    existing = await CustomAgentRepository.get(name)
    if not existing:
        raise HTTPException(status_code=404, detail="Agent not found")

    await CustomAgentRepository.delete(name)
    custom_loader = request.app.state.custom_loader
    await custom_loader.reload()
    return {"status": "deleted", "name": name}
