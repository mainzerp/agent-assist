"""Integration tests for API endpoints.

Tests all API routes using httpx AsyncClient with ASGITransport against the
real FastAPI app with mocked dependencies.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from app.security.auth import (
    require_admin_session,
    require_admin_session_redirect,
    require_api_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_test_app(
    *,
    override_api_key: bool = True,
    override_admin_session: bool = True,
    mock_ha_rest_client=None,
):
    """Build a FastAPI test app with test lifespan and optional auth overrides."""
    from app.main import create_app
    from app.api.routes import conversation as conversation_routes
    from app.api.routes import admin as admin_routes

    app = create_app()

    # ---- auth overrides ----
    if override_api_key:
        app.dependency_overrides[require_api_key] = lambda: "test-api-key"
    if override_admin_session:
        app.dependency_overrides[require_admin_session] = lambda: {"username": "admin"}
        app.dependency_overrides[require_admin_session_redirect] = lambda: {"username": "admin"}

    # ---- mock registry ----
    mock_registry = MagicMock()
    mock_registry.list_agents = AsyncMock(return_value=[])
    admin_routes.set_registry(mock_registry)

    # ---- mock dispatcher ----
    mock_response = MagicMock()
    mock_response.error = None
    mock_response.result = {"speech": "Test response from agent"}

    mock_dispatcher = MagicMock()
    mock_dispatcher.dispatch = AsyncMock(return_value=mock_response)

    # Streaming mock
    async def _stream(req):
        chunk = MagicMock()
        chunk.result = {"token": "Hello"}
        chunk.done = False
        yield chunk
        final = MagicMock()
        final.result = {"token": ""}
        final.done = True
        yield final

    mock_dispatcher.dispatch_stream = _stream
    conversation_routes.set_dispatcher(mock_dispatcher)

    # ---- no-op lifespan (ASGITransport does not trigger lifespan events,
    #      but override it just in case to prevent the real one from running) ----
    @asynccontextmanager
    async def _noop_lifespan(a):
        yield

    app.router.lifespan_context = _noop_lifespan

    # ---- set state directly on the app object ----
    app.state.startup_time = 0
    app.state.registry = mock_registry
    app.state.dispatcher = mock_dispatcher
    app.state.ha_client = mock_ha_rest_client or MagicMock()
    app.state.entity_index = None
    app.state.cache_manager = None
    app.state.entity_matcher = None
    app.state.alias_resolver = None
    app.state.custom_loader = None
    mcp_reg = MagicMock()
    mcp_reg.list_servers.return_value = []
    app.state.mcp_registry = mcp_reg
    app.state.mcp_tool_manager = MagicMock()
    app.state.ws_client = None
    app.state.presence_detector = None
    plugin_ldr = MagicMock()
    plugin_ldr.loaded_plugins = {}
    app.state.plugin_loader = plugin_ldr
    return app


@pytest_asyncio.fixture()
async def authed_client(db_repository):
    """Async httpx client with all auth dependencies overridden."""
    app = _build_test_app()
    with patch(
        "app.db.repository.SetupStateRepository.is_complete",
        new_callable=AsyncMock,
        return_value=True,
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            yield client


@pytest_asyncio.fixture()
async def unauthed_client(db_repository):
    """Async httpx client with NO auth overrides (for 401 tests)."""
    app = _build_test_app(override_api_key=False, override_admin_session=False)
    with patch(
        "app.db.repository.SetupStateRepository.is_complete",
        new_callable=AsyncMock,
        return_value=True,
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            yield client


# ===================================================================
# Health
# ===================================================================


@pytest.mark.integration
class TestHealthEndpoint:

    async def test_health_returns_200(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/health")
        assert resp.status_code == 200

    async def test_health_returns_status_json(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    async def test_health_accessible_without_auth(self, unauthed_client: httpx.AsyncClient):
        resp = await unauthed_client.get("/api/health")
        assert resp.status_code == 200


# ===================================================================
# Conversation REST + SSE
# ===================================================================


@pytest.mark.integration
class TestConversationEndpoints:

    async def test_conversation_rest_returns_response(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.post(
            "/api/conversation",
            json={"text": "turn on the kitchen light"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "speech" in data

    async def test_conversation_rest_without_auth_returns_401(
        self, unauthed_client: httpx.AsyncClient
    ):
        resp = await unauthed_client.post(
            "/api/conversation",
            json={"text": "turn on the kitchen light"},
        )
        assert resp.status_code == 401

    async def test_conversation_rest_invalid_payload_returns_422(
        self, authed_client: httpx.AsyncClient
    ):
        resp = await authed_client.post("/api/conversation", json={})
        assert resp.status_code == 422

    async def test_conversation_sse_returns_event_stream(
        self, authed_client: httpx.AsyncClient
    ):
        resp = await authed_client.post(
            "/api/conversation/stream",
            json={"text": "turn on the kitchen light"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    async def test_conversation_sse_without_auth_returns_401(
        self, unauthed_client: httpx.AsyncClient
    ):
        resp = await unauthed_client.post(
            "/api/conversation/stream",
            json={"text": "hello"},
        )
        assert resp.status_code == 401


# ===================================================================
# Admin Settings
# ===================================================================


@pytest.mark.integration
class TestAdminSettingsEndpoints:

    async def test_get_settings_returns_200(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "settings" in data

    async def test_get_settings_without_auth_returns_401(
        self, unauthed_client: httpx.AsyncClient
    ):
        resp = await unauthed_client.get("/api/admin/settings")
        assert resp.status_code == 401

    async def test_put_settings_updates_value(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.put(
            "/api/admin/settings",
            json={"log_level": "DEBUG"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    async def test_put_single_setting(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.put(
            "/api/admin/settings/log_level",
            json={"value": "WARNING", "value_type": "string", "category": "general"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["key"] == "log_level"


# ===================================================================
# Admin Agents
# ===================================================================


@pytest.mark.integration
class TestAdminAgentsEndpoint:

    async def test_list_agents_returns_200(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data
        assert isinstance(data["agents"], list)


# ===================================================================
# Entity Index API
# ===================================================================


@pytest.mark.integration
class TestEntityIndexAPI:

    async def test_get_stats_returns_200(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/entity-index/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data

    async def test_get_stats_not_initialized(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/entity-index/stats")
        data = resp.json()
        # entity_index is None in test lifespan
        assert data["count"] == 0
        assert data.get("status") == "not_initialized"


# ===================================================================
# Cache API
# ===================================================================


@pytest.mark.integration
class TestCacheAPI:

    async def test_get_cache_stats_returns_200(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/cache/stats")
        assert resp.status_code == 200

    async def test_get_cache_stats_not_initialized(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/cache/stats")
        data = resp.json()
        # cache_manager is None in test lifespan
        assert data.get("status") == "not_initialized"

    async def test_flush_cache_not_initialized(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.post(
            "/api/admin/cache/flush",
            json={},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"


# ===================================================================
# MCP API
# ===================================================================


@pytest.mark.integration
class TestMcpAPI:

    async def test_list_mcp_servers_returns_200(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/mcp-servers")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    async def test_add_mcp_server_duplicate_returns_409(
        self, authed_client: httpx.AsyncClient
    ):
        body = {
            "name": "test-srv",
            "transport": "stdio",
            "command_or_url": "echo hello",
        }
        # Insert first via DB so the duplicate check triggers
        from app.db.repository import McpServerRepository

        await McpServerRepository.create(
            name="test-srv",
            transport="stdio",
            command_or_url="echo hello",
        )
        resp = await authed_client.post("/api/admin/mcp-servers", json=body)
        assert resp.status_code == 409


# ===================================================================
# Plugins API
# ===================================================================


@pytest.mark.integration
class TestPluginsAPI:

    async def test_list_plugins_returns_200(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/plugins")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


# ===================================================================
# Presence API
# ===================================================================


@pytest.mark.integration
class TestPresenceAPI:

    async def test_get_presence_status_returns_200(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/presence/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "enabled" in data
        assert "room_confidence" in data
        assert "sensors" in data

    async def test_update_presence_config_returns_200(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.put(
            "/api/admin/presence/config",
            json={"enabled": False, "decay_timeout": 600.0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "updated"


# ===================================================================
# Analytics API
# ===================================================================


@pytest.mark.integration
class TestAnalyticsAPI:

    async def test_analytics_overview_returns_200(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/analytics/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data
        assert "avg_latency_ms" in data
        assert "cache_hit_rate" in data

    async def test_analytics_requests_returns_200(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/analytics/requests")
        assert resp.status_code == 200


# ===================================================================
# Traces API
# ===================================================================


@pytest.mark.integration
class TestTracesAPI:

    async def test_list_traces_returns_200(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert "traces" in data
        assert "total" in data

    async def test_get_trace_detail_not_found(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/traces/nonexistent-trace-id")
        assert resp.status_code == 404


# ===================================================================
# Conversations API
# ===================================================================


@pytest.mark.integration
class TestConversationsAPI:

    async def test_list_conversations_returns_200(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert "conversations" in data
        assert "total" in data
