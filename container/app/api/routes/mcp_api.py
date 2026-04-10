"""MCP server management API endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.security.auth import require_admin_session
from app.db.repository import McpServerRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin/mcp-servers",
    tags=["admin-mcp"],
    dependencies=[Depends(require_admin_session)],
)


class McpServerCreate(BaseModel):
    name: str
    transport: str
    command_or_url: str
    env_vars: dict[str, str] | None = None
    timeout: int = 30


@router.get("")
async def list_mcp_servers(request: Request) -> list[dict[str, Any]]:
    """List all MCP servers with connection status."""
    db_servers = await McpServerRepository.list_all()
    mcp_registry = request.app.state.mcp_registry
    live_status = {s["name"]: s["connected"] for s in mcp_registry.list_servers()}

    result = []
    for server in db_servers:
        server["connected"] = live_status.get(server["name"], False)
        result.append(server)
    return result


@router.post("", status_code=201)
async def add_mcp_server(request: Request, body: McpServerCreate) -> dict[str, Any]:
    """Add a new MCP server."""
    existing = await McpServerRepository.get(body.name)
    if existing:
        raise HTTPException(status_code=409, detail="Server with this name already exists")

    mcp_registry = request.app.state.mcp_registry
    connected = await mcp_registry.add_server(
        name=body.name,
        transport=body.transport,
        command_or_url=body.command_or_url,
        env_vars=body.env_vars,
        timeout=body.timeout,
    )
    return {"name": body.name, "connected": connected}


@router.delete("/{name}")
async def remove_mcp_server(request: Request, name: str) -> dict[str, str]:
    """Remove an MCP server."""
    existing = await McpServerRepository.get(name)
    if not existing:
        raise HTTPException(status_code=404, detail="Server not found")

    mcp_registry = request.app.state.mcp_registry
    await mcp_registry.remove_server(name)
    return {"status": "deleted", "name": name}


@router.get("/{name}/tools")
async def list_server_tools(request: Request, name: str) -> list[dict[str, Any]]:
    """List discovered tools for a specific MCP server."""
    mcp_tool_manager = request.app.state.mcp_tool_manager
    all_tools = await mcp_tool_manager.discover_tools()
    server_tools = all_tools.get(name)
    if server_tools is None:
        existing = await McpServerRepository.get(name)
        if not existing:
            raise HTTPException(status_code=404, detail="Server not found")
        return []
    return server_tools
