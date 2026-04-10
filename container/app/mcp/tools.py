"""MCP tool discovery and agent assignment."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.db.repository import CustomAgentRepository
from app.mcp.registry import MCPServerRegistry

logger = logging.getLogger(__name__)


class MCPToolManager:
    """Discovers MCP tools and maps them to agents."""

    def __init__(self, mcp_registry: MCPServerRegistry) -> None:
        self._registry = mcp_registry

    async def discover_tools(self) -> dict[str, list[dict[str, Any]]]:
        """For each connected server, list available tools.

        Returns:
            Dict mapping server name to list of tool descriptors.
        """
        result: dict[str, list[dict[str, Any]]] = {}
        for info in self._registry.list_servers():
            name = info["name"]
            if not info["connected"]:
                continue
            client = self._registry.get_client(name)
            if client is None:
                continue
            try:
                tools = await client.list_tools()
                result[name] = tools
            except Exception:
                logger.error("Failed to discover tools for server '%s'", name, exc_info=True)
                result[name] = []
        return result

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict | None = None
    ) -> Any:
        """Invoke a tool on a specific MCP server."""
        client = self._registry.get_client(server_name)
        if client is None:
            raise ValueError(f"MCP server '{server_name}' not found")
        if not client.connected:
            raise ConnectionError(f"MCP server '{server_name}' is not connected")
        try:
            return await asyncio.wait_for(
                client.call_tool(tool_name, arguments),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.error("MCP tool '%s.%s' timed out after 30s", server_name, tool_name)
            raise

    async def get_tools_for_agent(self, agent_id: str) -> list[dict[str, Any]]:
        """Look up tool assignments for a given agent.

        Reads the mcp_tools JSON field from custom_agents table.
        Returns tool descriptors for tools assigned to the agent.
        """
        # Extract custom agent name from agent_id (custom-<name>)
        if agent_id.startswith("custom-"):
            name = agent_id[len("custom-"):]
        else:
            return []

        row = await CustomAgentRepository.get(name)
        if not row:
            return []

        mcp_tools = row.get("mcp_tools") or []
        if not mcp_tools:
            return []

        # mcp_tools is a list of {"server": str, "tool": str} dicts
        assigned: list[dict[str, Any]] = []
        for entry in mcp_tools:
            server_name = entry.get("server", "")
            tool_name = entry.get("tool", "")
            client = self._registry.get_client(server_name)
            if client is None or not client.connected:
                continue
            try:
                tools = await client.list_tools()
                for tool in tools:
                    if tool["name"] == tool_name:
                        assigned.append(tool)
                        break
            except Exception:
                logger.warning(
                    "Could not fetch tool '%s' from server '%s'",
                    tool_name, server_name, exc_info=True,
                )
        return assigned
