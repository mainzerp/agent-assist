"""Integration tests for dashboard routes.

Tests login-required behavior, page accessibility with session, and basic
template rendering.
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


def _build_dashboard_app(*, override_session: bool = True):
    """Build a FastAPI app for dashboard integration tests."""
    from app.main import create_app

    app = create_app()

    if override_session:
        app.dependency_overrides[require_admin_session] = lambda: {"username": "admin"}
        app.dependency_overrides[require_admin_session_redirect] = lambda: {"username": "admin"}
        app.dependency_overrides[require_api_key] = lambda: "test-api-key"

    @asynccontextmanager
    async def _noop_lifespan(a):
        yield

    app.router.lifespan_context = _noop_lifespan

    # Set state directly
    app.state.startup_time = 0
    app.state.registry = MagicMock()
    app.state.registry.list_agents = AsyncMock(return_value=[])
    app.state.dispatcher = MagicMock()
    app.state.ha_client = MagicMock()
    app.state.entity_index = None
    app.state.cache_manager = None
    app.state.entity_matcher = None
    app.state.alias_resolver = None
    app.state.custom_loader = None
    app.state.mcp_registry = MagicMock()
    app.state.mcp_registry.list_servers.return_value = []
    app.state.mcp_tool_manager = MagicMock()
    app.state.ws_client = None
    app.state.presence_detector = None
    app.state.plugin_loader = MagicMock()
    app.state.plugin_loader.loaded_plugins = {}
    app.state.setup_runtime_initialized = True
    return app


@pytest_asyncio.fixture()
async def dashboard_client(db_repository):
    """Client with admin session authentication overridden."""
    app = _build_dashboard_app(override_session=True)
    with patch(
        "app.db.repository.SetupStateRepository.is_complete",
        new_callable=AsyncMock,
        return_value=True,
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            follow_redirects=False,
        ) as client:
            yield client


@pytest_asyncio.fixture()
async def no_session_client(db_repository):
    """Client WITHOUT session auth overrides (for login-required tests)."""
    app = _build_dashboard_app(override_session=False)
    with patch(
        "app.db.repository.SetupStateRepository.is_complete",
        new_callable=AsyncMock,
        return_value=True,
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            follow_redirects=False,
        ) as client:
            yield client


# ===================================================================
# Login required (redirect)
# ===================================================================


@pytest.mark.integration
class TestDashboardLoginRequired:
    async def test_dashboard_index_requires_auth(self, no_session_client: httpx.AsyncClient):
        resp = await no_session_client.get("/dashboard/")
        # The require_admin_session_redirect raises HTTPException with 303
        # which gets handled by the exception handler as a JSON response
        # with Location header
        assert resp.status_code == 303
        assert "/dashboard/login" in resp.headers.get("location", "")

    async def test_agents_page_requires_auth(self, no_session_client: httpx.AsyncClient):
        resp = await no_session_client.get("/dashboard/agents")
        assert resp.status_code == 303

    async def test_login_page_accessible_without_session(self, no_session_client: httpx.AsyncClient):
        resp = await no_session_client.get("/dashboard/login")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


# ===================================================================
# Page accessibility with session
# ===================================================================


@pytest.mark.integration
class TestDashboardPageAccessibility:
    async def test_dashboard_index(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.get("/dashboard/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    async def test_agents_page(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.get("/dashboard/agents")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    async def test_system_health_page(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.get("/dashboard/system-health")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    async def test_mcp_servers_page(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.get("/dashboard/mcp-servers")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    async def test_entity_visibility_redirects(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.get("/dashboard/entity-visibility", follow_redirects=False)
        assert resp.status_code == 301
        assert "/dashboard/entity-index" in resp.headers.get("location", "")

    async def test_entity_visibility_redirect_preserves_agent(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.get("/dashboard/entity-visibility?agent=light-agent", follow_redirects=False)
        assert resp.status_code == 301
        assert "agent=light-agent" in resp.headers.get("location", "")

    async def test_analytics_page(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.get("/dashboard/analytics")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    async def test_plugins_page(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.get("/dashboard/plugins")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    async def test_presence_page(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.get("/dashboard/presence")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


# ===================================================================
# Template rendering content checks
# ===================================================================


@pytest.mark.integration
class TestDashboardTemplateRendering:
    async def test_login_page_contains_form(self, no_session_client: httpx.AsyncClient):
        resp = await no_session_client.get("/dashboard/login")
        html = resp.text
        assert "<form" in html.lower() or "form" in html.lower()

    async def test_logout_clears_session(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.get("/dashboard/logout")
        assert resp.status_code == 303
        assert "/dashboard/login" in resp.headers.get("location", "")


# ===================================================================
# Personality page
# ===================================================================


@pytest.mark.integration
class TestPersonalityPage:
    async def test_personality_page_accessible(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.get("/dashboard/personality")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    async def test_personality_page_requires_auth(self, no_session_client: httpx.AsyncClient):
        resp = await no_session_client.get("/dashboard/personality")
        assert resp.status_code == 303

    async def test_get_personality_config(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.get("/api/admin/personality/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "prompt" in data

    async def test_put_personality_config(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.put(
            "/api/admin/personality/config",
            json={"prompt": "You are Lucia, a friendly assistant."},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        # Verify it persisted
        resp2 = await dashboard_client.get("/api/admin/personality/config")
        assert resp2.json()["prompt"] == "You are Lucia, a friendly assistant."

    async def test_put_personality_config_empty(self, dashboard_client: httpx.AsyncClient):
        # Set then clear
        await dashboard_client.put(
            "/api/admin/personality/config",
            json={"prompt": "Something"},
        )
        resp = await dashboard_client.put(
            "/api/admin/personality/config",
            json={"prompt": ""},
        )
        assert resp.status_code == 200
        resp2 = await dashboard_client.get("/api/admin/personality/config")
        assert resp2.json()["prompt"] == ""


# ===================================================================
# Overview extended endpoint
# ===================================================================


@pytest.mark.integration
class TestOverviewExtended:
    async def test_overview_extended_returns_all_fields(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.get("/api/admin/overview/extended")
        assert resp.status_code == 200
        data = resp.json()

        expected_keys = {
            "recent_requests",
            "cache_hit_rate",
            "agent_count",
            "entity_count",
            "mcp_server_count",
            "presence_rooms",
            "avg_latency_ms",
            "total_conversations",
            "agent_distribution",
            "cache_tier",
            "request_trend",
            "recent_traces",
            "warnings",
        }
        assert expected_keys.issubset(data.keys())

        assert isinstance(data["agent_distribution"], list)
        assert isinstance(data["cache_tier"], dict)
        assert "routing_hits" in data["cache_tier"]
        assert "response_hits" in data["cache_tier"]
        assert "misses" in data["cache_tier"]
        assert isinstance(data["request_trend"], dict)
        assert "labels" in data["request_trend"]
        assert "data" in data["request_trend"]
        assert isinstance(data["recent_traces"], list)
        assert isinstance(data["warnings"], dict)
        assert "agent_timeouts" in data["warnings"]
        assert "rewrite_failures" in data["warnings"]

    async def test_overview_extended_numeric_types(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.get("/api/admin/overview/extended")
        assert resp.status_code == 200
        data = resp.json()

        assert isinstance(data["recent_requests"], int)
        assert isinstance(data["cache_hit_rate"], (int, float))
        assert isinstance(data["agent_count"], int)
        assert isinstance(data["entity_count"], int)
        assert isinstance(data["avg_latency_ms"], (int, float))
        assert isinstance(data["total_conversations"], int)

    async def test_overview_extended_attempts_runtime_bootstrap(self, dashboard_client: httpx.AsyncClient):
        with patch(
            "app.api.routes.dashboard_api.ensure_setup_runtime_initialized",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_init:
            resp = await dashboard_client.get("/api/admin/overview/extended")
            assert resp.status_code == 200
            mock_init.assert_awaited_once()

    async def test_extended_health_reports_entity_index_building_as_warning(self, dashboard_client: httpx.AsyncClient):
        app = dashboard_client._transport.app
        app.state.ha_client.get_states = AsyncMock(return_value=[])
        entity_index = MagicMock()
        entity_index.get_stats.return_value = {
            "count": 0,
            "embedding_status": {
                "state": "building",
                "progress": 25,
                "processed": 500,
                "total": 2000,
                "error": None,
            },
        }
        app.state.entity_index = entity_index
        app.state.cache_manager = MagicMock()
        app.state.cache_manager.get_stats.return_value = {"routing": {}, "response": {}}

        resp = await dashboard_client.get("/api/admin/health/extended")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_index"]["status"] == "warning"
        assert data["entity_index"]["progress"] == 25


# ===================================================================
# Send devices API
# ===================================================================


@pytest.mark.integration
class TestSendDevicesAPI:
    async def test_list_empty(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.get("/api/admin/send-devices")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_create_and_list(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.post(
            "/api/admin/send-devices",
            json={
                "display_name": "Laura Handy",
                "device_type": "notify",
                "ha_service_target": "mobile_app_lauras_iphone",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data

        resp = await dashboard_client.get("/api/admin/send-devices")
        assert resp.status_code == 200
        mappings = resp.json()
        assert len(mappings) == 1
        assert mappings[0]["display_name"] == "Laura Handy"

    async def test_create_duplicate_rejected(self, dashboard_client: httpx.AsyncClient):
        await dashboard_client.post(
            "/api/admin/send-devices",
            json={
                "display_name": "Laura Handy",
                "device_type": "notify",
                "ha_service_target": "svc_a",
            },
        )
        resp = await dashboard_client.post(
            "/api/admin/send-devices",
            json={
                "display_name": "Laura Handy",
                "device_type": "notify",
                "ha_service_target": "svc_b",
            },
        )
        assert resp.status_code == 409

    async def test_create_invalid_type(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.post(
            "/api/admin/send-devices",
            json={
                "display_name": "Test",
                "device_type": "invalid",
                "ha_service_target": "svc",
            },
        )
        assert resp.status_code == 400

    async def test_delete(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.post(
            "/api/admin/send-devices",
            json={
                "display_name": "To Delete",
                "device_type": "notify",
                "ha_service_target": "svc_del",
            },
        )
        mapping_id = resp.json()["id"]
        resp = await dashboard_client.delete(f"/api/admin/send-devices/{mapping_id}")
        assert resp.status_code == 200

        resp = await dashboard_client.get("/api/admin/send-devices")
        assert resp.status_code == 200
        assert len(resp.json()) == 0

    async def test_send_devices_page(self, dashboard_client: httpx.AsyncClient):
        resp = await dashboard_client.get("/dashboard/send-devices")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
