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
            json={"items": {"cache.routing.threshold": "0.90"}},
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

    async def test_list_traces_with_search(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/traces?search=light")
        assert resp.status_code == 200

    async def test_list_traces_with_agent_filter(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/traces?agent=light-agent")
        assert resp.status_code == 200

    async def test_export_traces_csv(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/traces/export")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers.get("content-type", "")

    async def test_list_labels(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/traces/labels")
        assert resp.status_code == 200
        assert "labels" in resp.json()

    async def test_update_label_not_found(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.put(
            "/api/admin/traces/nonexistent/label",
            json={"label": "test"},
        )
        assert resp.status_code == 404

    async def test_trace_detail_returns_three_communication_entries(self, authed_client: httpx.AsyncClient):
        """Trace detail should build 3 agent_communication entries for the full round-trip."""
        summary = {
            "trace_id": "t-comm-3",
            "conversation_id": "conv-1",
            "created_at": "2024-01-01T00:00:00",
            "total_duration_ms": 100,
            "user_input": "turn on the light",
            "final_response": "Done, light is on.",
            "routing_agent": "light-agent",
            "routing_confidence": 0.9,
            "routing_duration_ms": 20,
            "routing_reasoning": None,
            "agent_instructions": None,
            "label": None,
            "source": "api",
        }
        spans = [
            {
                "span_name": "classify",
                "agent_id": "orchestrator",
                "start_time": "2024-01-01T00:00:00",
                "duration_ms": 20,
                "status": "ok",
                "metadata": {
                    "target_agent": "light-agent",
                    "condensed_task": "Turn on the light",
                    "confidence": 0.9,
                    "routing_cached": False,
                },
            },
            {
                "span_name": "dispatch",
                "agent_id": "light-agent",
                "start_time": "2024-01-01T00:00:01",
                "duration_ms": 80,
                "status": "ok",
                "metadata": {"agent_response": "Light turned on."},
            },
        ]
        with patch("app.api.routes.traces_api.TraceSummaryRepository") as mock_summary, \
             patch("app.api.routes.traces_api.TraceSpanRepository") as mock_spans:
            mock_summary.get = AsyncMock(return_value=summary)
            mock_spans.get_trace_spans = AsyncMock(return_value=spans)
            resp = await authed_client.get("/api/admin/traces/t-comm-3")
        assert resp.status_code == 200
        data = resp.json()
        comms = data["agent_communication"]
        assert len(comms) == 3
        assert comms[0]["from_agent"] == "user"
        assert comms[0]["to_agent"] == "orchestrator"
        assert comms[0]["task"] == "turn on the light"
        assert comms[1]["from_agent"] == "orchestrator"
        assert comms[1]["to_agent"] == "light-agent"
        assert comms[1]["task"] == "Turn on the light"
        assert comms[1]["response"] == "Light turned on."
        assert comms[2]["from_agent"] == "light-agent"
        assert comms[2]["to_agent"] == "orchestrator"
        assert comms[2]["task"] == ""
        assert comms[2]["response"] == "Done, light is on."
        assert comms[2]["response_unchanged"] is False

    async def test_trace_communication_task_pass_through(self, authed_client: httpx.AsyncClient):
        """When condensed_task == user_input, step 2 should have task_pass_through=True."""
        summary = {
            "trace_id": "t-pass",
            "conversation_id": "conv-1",
            "created_at": "2024-01-01T00:00:00",
            "total_duration_ms": 100,
            "user_input": "turn on the light",
            "final_response": "Done.",
            "routing_agent": "light-agent",
            "routing_confidence": 0.9,
            "routing_duration_ms": 20,
            "routing_reasoning": None,
            "agent_instructions": None,
            "label": None,
            "source": "api",
        }
        spans = [
            {
                "span_name": "classify",
                "agent_id": "orchestrator",
                "start_time": "2024-01-01T00:00:00",
                "duration_ms": 20,
                "status": "ok",
                "metadata": {
                    "target_agent": "light-agent",
                    "condensed_task": "turn on the light",
                    "confidence": 0.9,
                    "routing_cached": False,
                },
            },
            {
                "span_name": "dispatch",
                "agent_id": "light-agent",
                "start_time": "2024-01-01T00:00:01",
                "duration_ms": 80,
                "status": "ok",
                "metadata": {"agent_response": "Done."},
            },
            {
                "span_name": "return",
                "agent_id": "orchestrator",
                "start_time": "2024-01-01T00:00:02",
                "duration_ms": 5,
                "status": "ok",
                "metadata": {"from_agent": "light-agent", "final_response": "Done.", "mediated": False},
            },
        ]
        with patch("app.api.routes.traces_api.TraceSummaryRepository") as mock_summary, \
             patch("app.api.routes.traces_api.TraceSpanRepository") as mock_spans:
            mock_summary.get = AsyncMock(return_value=summary)
            mock_spans.get_trace_spans = AsyncMock(return_value=spans)
            resp = await authed_client.get("/api/admin/traces/t-pass")
        assert resp.status_code == 200
        data = resp.json()
        comms = data["agent_communication"]
        assert comms[1]["task_pass_through"] is True
        assert comms[2]["response_unchanged"] is True


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


# ===================================================================
# LLM Provider API
# ===================================================================


@pytest.mark.integration
class TestLLMProviderAPI:

    async def test_get_llm_providers_returns_200(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/llm-providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        assert "openrouter" in data["providers"]
        assert "groq" in data["providers"]
        assert "anthropic" in data["providers"]
        assert "ollama" in data["providers"]

    async def test_get_llm_providers_none_configured(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/llm-providers")
        data = resp.json()
        assert data["providers"]["openrouter"]["configured"] is False
        assert data["providers"]["groq"]["configured"] is False
        assert data["providers"]["anthropic"]["configured"] is False

    async def test_put_llm_provider_key(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.put(
            "/api/admin/llm-providers",
            json={"provider": "groq", "api_key": "gsk_test_key_12345678"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["provider"] == "groq"

    async def test_put_llm_provider_key_then_get_shows_configured(
        self, authed_client: httpx.AsyncClient
    ):
        await authed_client.put(
            "/api/admin/llm-providers",
            json={"provider": "groq", "api_key": "gsk_test_key_12345678"},
        )
        resp = await authed_client.get("/api/admin/llm-providers")
        data = resp.json()
        assert data["providers"]["groq"]["configured"] is True
        assert data["providers"]["groq"]["masked_key"] == "5678"

    async def test_put_llm_provider_unknown_returns_400(
        self, authed_client: httpx.AsyncClient
    ):
        resp = await authed_client.put(
            "/api/admin/llm-providers",
            json={"provider": "unknown_provider", "api_key": "key"},
        )
        assert resp.status_code == 400

    async def test_put_ollama_url(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.put(
            "/api/admin/llm-providers/ollama",
            json={"url": "http://myhost:11434"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    async def test_put_ollama_url_then_get_shows_configured(
        self, authed_client: httpx.AsyncClient
    ):
        await authed_client.put(
            "/api/admin/llm-providers/ollama",
            json={"url": "http://myhost:11434"},
        )
        resp = await authed_client.get("/api/admin/llm-providers")
        data = resp.json()
        assert data["providers"]["ollama"]["configured"] is True
        assert data["providers"]["ollama"]["url"] == "http://myhost:11434"

    async def test_delete_llm_provider_key(self, authed_client: httpx.AsyncClient):
        # Store a key first
        await authed_client.put(
            "/api/admin/llm-providers",
            json={"provider": "openrouter", "api_key": "sk-or-test1234"},
        )
        # Delete it
        resp = await authed_client.delete("/api/admin/llm-providers/openrouter")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        # Verify it's gone
        resp = await authed_client.get("/api/admin/llm-providers")
        data = resp.json()
        assert data["providers"]["openrouter"]["configured"] is False

    async def test_delete_llm_provider_unknown_returns_400(
        self, authed_client: httpx.AsyncClient
    ):
        resp = await authed_client.delete("/api/admin/llm-providers/unknown_prov")
        assert resp.status_code == 400

    async def test_test_llm_provider_no_key_returns_error(
        self, authed_client: httpx.AsyncClient
    ):
        resp = await authed_client.post(
            "/api/admin/llm-providers/test",
            json={"provider": "groq"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "No API key" in data["detail"]

    async def test_test_llm_provider_unknown_returns_error(
        self, authed_client: httpx.AsyncClient
    ):
        resp = await authed_client.post(
            "/api/admin/llm-providers/test",
            json={"provider": "unknown_prov"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"

    async def test_get_configured_providers(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/llm-providers/configured")
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        assert isinstance(data["providers"], list)
        # Ollama is always included
        assert "ollama" in data["providers"]

    async def test_get_configured_providers_after_storing_key(
        self, authed_client: httpx.AsyncClient
    ):
        await authed_client.put(
            "/api/admin/llm-providers",
            json={"provider": "groq", "api_key": "gsk_test_key_12345678"},
        )
        resp = await authed_client.get("/api/admin/llm-providers/configured")
        data = resp.json()
        assert "groq" in data["providers"]
        assert "ollama" in data["providers"]


# ===================================================================
# Entity Visibility Summary API
# ===================================================================


@pytest.mark.integration
class TestEntityVisibilitySummaryAPI:

    async def test_visibility_summary_empty(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.get("/api/admin/agents/visibility-summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert isinstance(data["summary"], dict)

    async def test_visibility_summary_with_rules(self, authed_client: httpx.AsyncClient):
        from app.db.repository import EntityVisibilityRepository

        await EntityVisibilityRepository.set_rules("light-agent", [
            {"rule_type": "domain_include", "rule_value": "light"},
            {"rule_type": "domain_include", "rule_value": "switch"},
            {"rule_type": "domain_exclude", "rule_value": "sensor"},
        ])
        resp = await authed_client.get("/api/admin/agents/visibility-summary")
        data = resp.json()
        summary = data["summary"]
        assert "light-agent" in summary
        assert summary["light-agent"]["has_rules"] is True
        assert "light" in summary["light-agent"]["domains"]
        assert "switch" in summary["light-agent"]["domains"]
        assert "sensor" in summary["light-agent"]["excluded_domains"]

    async def test_visibility_summary_includes_device_class_fields(self, authed_client: httpx.AsyncClient):
        from app.db.repository import EntityVisibilityRepository

        await EntityVisibilityRepository.set_rules("climate-agent", [
            {"rule_type": "domain_include", "rule_value": "climate"},
            {"rule_type": "domain_include", "rule_value": "sensor"},
            {"rule_type": "device_class_include", "rule_value": "temperature"},
            {"rule_type": "device_class_include", "rule_value": "humidity"},
        ])
        resp = await authed_client.get("/api/admin/agents/visibility-summary")
        data = resp.json()
        summary = data["summary"]
        assert "climate-agent" in summary
        assert "temperature" in summary["climate-agent"]["device_classes"]
        assert "humidity" in summary["climate-agent"]["device_classes"]
        assert summary["climate-agent"]["excluded_device_classes"] == []


@pytest.mark.integration
class TestEntityVisibilityRuleTypeValidation:

    async def test_put_invalid_rule_type_returns_422(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.put(
            "/api/admin/entity-visibility/light-agent",
            json={"rules": [{"rule_type": "bogus", "rule_value": "light"}]},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert "Invalid rule_type" in data["detail"]

    async def test_put_valid_rule_type_succeeds(self, authed_client: httpx.AsyncClient):
        resp = await authed_client.put(
            "/api/admin/entity-visibility/light-agent",
            json={"rules": [
                {"rule_type": "domain_include", "rule_value": "light"},
                {"rule_type": "entity_include", "rule_value": "switch.kitchen"},
            ]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["rules_count"] == 2
