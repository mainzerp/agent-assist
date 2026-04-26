"""MCP tool discovery and agent assignment."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from app.mcp.registry import MCPServerRegistry

logger = logging.getLogger(__name__)


@dataclass
class _ToolDescriptorCacheEntry:
    tools: list[dict[str, Any]]
    connected: bool
    refreshed_at: float


class MCPToolManager:
    """Discovers MCP tools and maps them to agents."""

    def __init__(self, mcp_registry: MCPServerRegistry) -> None:
        self._registry = mcp_registry
        self._descriptor_cache: dict[str, _ToolDescriptorCacheEntry] = {}
        self._server_locks: dict[str, asyncio.Lock] = {}
        register_listener = getattr(mcp_registry, "add_change_listener", None)
        if callable(register_listener):
            register_listener(self._handle_registry_change)

    def _handle_registry_change(self, server_name: str | None) -> None:
        if server_name is None:
            self.invalidate_all()
        else:
            self.invalidate_server(server_name)

    def _lock_for(self, server_name: str) -> asyncio.Lock:
        lock = self._server_locks.get(server_name)
        if lock is None:
            lock = asyncio.Lock()
            self._server_locks[server_name] = lock
        return lock

    @staticmethod
    def _copy_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [dict(tool) for tool in tools]

    def _mark_disconnected(self, server_name: str) -> None:
        self._descriptor_cache[server_name] = _ToolDescriptorCacheEntry(
            tools=[],
            connected=False,
            refreshed_at=time.monotonic(),
        )

    def invalidate_server(self, server_name: str) -> None:
        """Discard cached tool descriptors for one MCP server."""
        self._descriptor_cache.pop(server_name, None)

    def invalidate_all(self) -> None:
        """Discard all cached MCP tool descriptors."""
        self._descriptor_cache.clear()

    async def refresh_server(self, server_name: str) -> list[dict[str, Any]]:
        """Force-refresh descriptors for one MCP server."""
        return await self._get_server_tools(server_name, force_refresh=True)

    async def _get_server_tools(self, server_name: str, *, force_refresh: bool = False) -> list[dict[str, Any]]:
        client = self._registry.get_client(server_name)
        if client is None or not client.connected:
            self._mark_disconnected(server_name)
            return []

        cached = self._descriptor_cache.get(server_name)
        if cached and cached.connected and not force_refresh:
            return self._copy_tools(cached.tools)

        async with self._lock_for(server_name):
            client = self._registry.get_client(server_name)
            if client is None or not client.connected:
                self._mark_disconnected(server_name)
                return []

            cached = self._descriptor_cache.get(server_name)
            if cached and cached.connected and not force_refresh:
                return self._copy_tools(cached.tools)

            try:
                tools = await client.list_tools()
            except Exception:
                self._descriptor_cache.pop(server_name, None)
                logger.warning("Failed to refresh MCP tools for server '%s'", server_name, exc_info=True)
                return []

            copied_tools = self._copy_tools(tools)
            self._descriptor_cache[server_name] = _ToolDescriptorCacheEntry(
                tools=copied_tools,
                connected=True,
                refreshed_at=time.monotonic(),
            )
            return self._copy_tools(copied_tools)

    async def discover_tools(self) -> dict[str, list[dict[str, Any]]]:
        """For each connected server, list available tools.

        Returns:
            Dict mapping server name to list of tool descriptors.
        """
        result: dict[str, list[dict[str, Any]]] = {}
        for info in self._registry.list_servers():
            name = info["name"]
            if not info["connected"]:
                self._mark_disconnected(name)
                continue
            result[name] = await self.refresh_server(name)
        return result

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict | None = None) -> Any:
        """Invoke a tool on a specific MCP server."""
        client = self._registry.get_client(server_name)
        if client is None:
            raise ValueError(f"MCP server '{server_name}' not found")
        if not client.connected:
            raise ConnectionError(f"MCP server '{server_name}' is not connected")
        timeout = float(client.timeout)
        try:
            return await asyncio.wait_for(
                client.call_tool(tool_name, arguments),
                timeout=timeout,
            )
        except TimeoutError:
            logger.error(
                "MCP tool '%s.%s' timed out after %ds",
                server_name,
                tool_name,
                int(timeout),
            )
            raise

    async def get_tools_for_agent(self, agent_id: str) -> list[dict[str, Any]]:
        """Look up tool assignments for a given agent.

        Uses the unified agent_mcp_tools table for built-in and custom agents.
        """
        from app.db.repository import AgentMcpToolsRepository

        assignments = await AgentMcpToolsRepository.get_tools(agent_id)
        if not assignments:
            return []

        server_names: list[str] = []
        for entry in assignments:
            server_name = entry.get("server_name") or entry.get("server", "")
            if server_name and server_name not in server_names:
                server_names.append(server_name)

        async def _load_server(server_name: str) -> tuple[str, list[dict[str, Any]]]:
            try:
                return server_name, await self._get_server_tools(server_name)
            except Exception:
                logger.warning("Could not fetch tools from server '%s'", server_name, exc_info=True)
                return server_name, []

        loaded = await asyncio.gather(*(_load_server(server_name) for server_name in server_names))
        tools_by_server = {server_name: tools for server_name, tools in loaded}

        assigned: list[dict[str, Any]] = []
        for entry in assignments:
            server_name = entry.get("server_name") or entry.get("server", "")
            tool_name = entry.get("tool_name") or entry.get("tool", "")
            tools = tools_by_server.get(server_name) or []
            for tool in tools:
                if tool.get("name") == tool_name:
                    tool_with_server = dict(tool)
                    tool_with_server["_server_name"] = server_name
                    assigned.append(tool_with_server)
                    break
        return assigned
