"""Tests for app.mcp -- client, registry, and tool manager."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.mcp.client import MCPClient
from app.mcp.registry import MCPServerRegistry
from app.mcp.tools import MCPToolManager


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------

class TestMCPClient:

    def test_initial_state_not_connected(self):
        client = MCPClient(name="test", transport="stdio", command_or_url="echo hello")
        assert client.connected is False
        assert client.name == "test"

    async def test_connect_unknown_transport_returns_false(self):
        client = MCPClient(name="test", transport="grpc", command_or_url="localhost:50051")
        result = await client.connect()
        assert result is False
        assert client.connected is False

    async def test_list_tools_returns_empty_when_not_connected(self):
        client = MCPClient(name="test", transport="stdio", command_or_url="echo")
        tools = await client.list_tools()
        assert tools == []

    async def test_call_tool_raises_when_not_connected(self):
        client = MCPClient(name="test", transport="stdio", command_or_url="echo")
        with pytest.raises(ConnectionError, match="Not connected"):
            await client.call_tool("some_tool", {"arg": 1})

    async def test_disconnect_clears_state(self):
        client = MCPClient(name="test", transport="stdio", command_or_url="echo")
        client._connected = True
        client._session = MagicMock()
        client._session_cm = None
        client._transport_cm = None
        await client.disconnect()
        assert client.connected is False
        assert client._session is None

    @patch("app.mcp.client.MCPClient._connect_stdio", new_callable=AsyncMock, return_value=True)
    async def test_connect_stdio_transport(self, mock_connect):
        client = MCPClient(name="test", transport="stdio", command_or_url="echo hello")
        result = await client.connect()
        assert result is True

    @patch("app.mcp.client.MCPClient._connect_sse", new_callable=AsyncMock, return_value=True)
    async def test_connect_sse_transport(self, mock_connect):
        client = MCPClient(name="test", transport="sse", command_or_url="http://localhost:3000")
        result = await client.connect()
        assert result is True

    async def test_list_tools_parses_result(self):
        client = MCPClient(name="test", transport="stdio", command_or_url="echo")
        client._connected = True
        tool_mock = MagicMock()
        tool_mock.name = "get_weather"
        tool_mock.description = "Get weather info"
        tool_mock.inputSchema = {"type": "object"}
        session = AsyncMock()
        session.list_tools = AsyncMock(return_value=MagicMock(tools=[tool_mock]))
        client._session = session

        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "get_weather"

    async def test_call_tool_invokes_session(self):
        client = MCPClient(name="test", transport="stdio", command_or_url="echo")
        client._connected = True
        session = AsyncMock()
        session.call_tool = AsyncMock(return_value={"result": "ok"})
        client._session = session

        result = await client.call_tool("my_tool", {"arg": "val"})
        assert result == {"result": "ok"}
        session.call_tool.assert_awaited_once_with("my_tool", arguments={"arg": "val"})

    def test_timeout_property_returns_configured_value(self):
        client = MCPClient(name="test", transport="stdio", command_or_url="echo", timeout=60)
        assert client.timeout == 60

    async def test_connect_http_transport_returns_false(self):
        """http transport is no longer supported; only stdio and sse are valid."""
        client = MCPClient(name="test", transport="http", command_or_url="http://localhost")
        result = await client.connect()
        assert result is False
        assert client.connected is False

    @patch("app.mcp.client.MCPClient._connect_stdio", new_callable=AsyncMock)
    async def test_connect_timeout_returns_false(self, mock_stdio):
        """If connection takes longer than timeout, connect() returns False."""
        async def slow_connect():
            await asyncio.sleep(10)
            return True

        mock_stdio.side_effect = slow_connect
        client = MCPClient(name="test", transport="stdio", command_or_url="echo", timeout=1)
        result = await client.connect()
        assert result is False
        assert client.connected is False


# ---------------------------------------------------------------------------
# MCPServerRegistry
# ---------------------------------------------------------------------------

class TestMCPServerRegistry:

    def test_list_servers_empty_on_init(self):
        registry = MCPServerRegistry()
        assert registry.list_servers() == []

    def test_get_client_returns_none_for_unknown(self):
        registry = MCPServerRegistry()
        assert registry.get_client("nonexistent") is None

    @patch("app.mcp.registry.McpServerRepository")
    @patch("app.mcp.registry.MCPClient")
    async def test_add_server_registers_and_connects(self, MockClient, mock_repo):
        mock_repo.upsert = AsyncMock()
        client_instance = AsyncMock()
        client_instance.connect = AsyncMock(return_value=True)
        client_instance.connected = True
        MockClient.return_value = client_instance

        registry = MCPServerRegistry()
        result = await registry.add_server("test-server", "stdio", "echo hello")
        assert result is True
        assert len(registry.list_servers()) == 1

    @patch("app.mcp.registry.McpServerRepository")
    async def test_remove_server_disconnects_and_removes(self, mock_repo):
        mock_repo.delete = AsyncMock()
        registry = MCPServerRegistry()
        client = AsyncMock()
        client.connected = True
        registry._clients["test-server"] = client

        await registry.remove_server("test-server")
        client.disconnect.assert_awaited_once()
        assert registry.get_client("test-server") is None

    async def test_disconnect_all_clears_clients(self):
        registry = MCPServerRegistry()
        client1 = AsyncMock()
        client2 = AsyncMock()
        registry._clients = {"a": client1, "b": client2}
        await registry.disconnect_all()
        assert len(registry._clients) == 0
        client1.disconnect.assert_awaited_once()
        client2.disconnect.assert_awaited_once()


# ---------------------------------------------------------------------------
# MCPToolManager
# ---------------------------------------------------------------------------

class TestMCPToolManager:

    async def test_discover_tools_returns_tools_for_connected_servers(self):
        registry = MagicMock(spec=MCPServerRegistry)
        registry.list_servers.return_value = [
            {"name": "server1", "connected": True},
            {"name": "server2", "connected": False},
        ]
        client1 = AsyncMock()
        client1.list_tools = AsyncMock(return_value=[{"name": "tool1", "description": "d", "input_schema": {}}])
        registry.get_client.side_effect = lambda n: client1 if n == "server1" else None

        manager = MCPToolManager(registry)
        tools = await manager.discover_tools()
        assert "server1" in tools
        assert "server2" not in tools
        assert len(tools["server1"]) == 1

    async def test_call_tool_raises_on_unknown_server(self):
        registry = MagicMock(spec=MCPServerRegistry)
        registry.get_client.return_value = None
        manager = MCPToolManager(registry)
        with pytest.raises(ValueError, match="not found"):
            await manager.call_tool("unknown", "tool1")

    async def test_call_tool_raises_on_disconnected_server(self):
        registry = MagicMock(spec=MCPServerRegistry)
        client = MagicMock()
        client.connected = False
        registry.get_client.return_value = client
        manager = MCPToolManager(registry)
        with pytest.raises(ConnectionError, match="not connected"):
            await manager.call_tool("server1", "tool1")

    async def test_call_tool_uses_per_server_timeout(self):
        """call_tool should use the client's configured timeout, not a hardcoded value."""
        registry = MagicMock(spec=MCPServerRegistry)
        client = MagicMock()
        client.connected = True
        client.timeout = 10
        client.call_tool = AsyncMock(return_value={"result": "ok"})
        registry.get_client.return_value = client

        manager = MCPToolManager(registry)
        result = await manager.call_tool("server1", "tool1", {"arg": "val"})
        assert result == {"result": "ok"}

    async def test_call_tool_raises_timeout_error(self):
        """call_tool should raise TimeoutError when tool execution exceeds server timeout."""
        registry = MagicMock(spec=MCPServerRegistry)
        client = MagicMock()
        client.connected = True
        client.timeout = 1

        async def slow_tool(*args, **kwargs):
            await asyncio.sleep(10)
            return {}

        client.call_tool = AsyncMock(side_effect=slow_tool)
        registry.get_client.return_value = client

        manager = MCPToolManager(registry)
        with pytest.raises(asyncio.TimeoutError):
            await manager.call_tool("server1", "tool1")


# ---------------------------------------------------------------------------
# DuckDuckGo MCP Server
# ---------------------------------------------------------------------------

class TestDuckDuckGoServerTools:

    def test_server_module_importable(self):
        """The DuckDuckGo MCP server module can be imported."""
        pytest.importorskip("mcp")
        pytest.importorskip("duckduckgo_search")
        from app.mcp.servers import duckduckgo_server
        assert hasattr(duckduckgo_server, "server")

    async def test_list_tools_returns_expected_tools(self):
        """Server exposes web_search and web_search_news tools."""
        pytest.importorskip("mcp")
        pytest.importorskip("duckduckgo_search")
        from app.mcp.servers.duckduckgo_server import list_tools
        tools = await list_tools()
        names = {t.name for t in tools}
        assert "web_search" in names
        assert "web_search_news" in names

    async def test_web_search_tool_has_query_param(self):
        """web_search tool requires a 'query' parameter."""
        pytest.importorskip("mcp")
        pytest.importorskip("duckduckgo_search")
        from app.mcp.servers.duckduckgo_server import list_tools
        tools = await list_tools()
        search_tool = next(t for t in tools if t.name == "web_search")
        assert "query" in search_tool.inputSchema["properties"]
        assert "query" in search_tool.inputSchema["required"]


# ---------------------------------------------------------------------------
# MCP Tool Assignment for Built-in Agents
# ---------------------------------------------------------------------------

class TestAgentMcpToolAssignment:

    async def test_get_tools_for_builtin_agent_returns_empty_by_default(self):
        """Built-in agent with no assignments returns empty list."""
        with patch("app.db.repository.AgentMcpToolsRepository.get_tools", new_callable=AsyncMock, return_value=[]):
            registry = MCPServerRegistry()
            manager = MCPToolManager(registry)
            tools = await manager.get_tools_for_agent("general-agent")
            assert tools == []

    async def test_get_tools_for_builtin_agent_returns_assigned_tools(self):
        """Built-in agent with assignments gets tool descriptors."""
        with patch("app.db.repository.AgentMcpToolsRepository.get_tools", new_callable=AsyncMock, return_value=[
            {"server": "duckduckgo-search", "tool": "web_search"}
        ]):
            registry = MCPServerRegistry()
            mock_client = MagicMock()
            mock_client.connected = True
            mock_client.list_tools = AsyncMock(return_value=[
                {"name": "web_search", "description": "Search", "input_schema": {}}
            ])
            registry._clients["duckduckgo-search"] = mock_client
            manager = MCPToolManager(registry)
            tools = await manager.get_tools_for_agent("general-agent")
            assert len(tools) == 1
            assert tools[0]["name"] == "web_search"
            assert tools[0]["_server_name"] == "duckduckgo-search"
