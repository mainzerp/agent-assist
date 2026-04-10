"""Tests for app.ha_client -- REST client, auth, and WebSocket."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from app.ha_client.rest import HARestClient
from app.ha_client.rest import test_ha_connection as _test_ha_connection
from app.ha_client.auth import (
    build_auth_headers,
    get_auth_headers,
    get_ha_token,
    set_ha_token,
    HA_TOKEN_SECRET_KEY,
)
from app.ha_client.websocket import HAWebSocketClient


# ---------------------------------------------------------------------------
# REST Client
# ---------------------------------------------------------------------------

class TestHARestClient:

    @respx.mock
    async def test_get_states_makes_get_request(self):
        states = [{"entity_id": "light.test", "state": "on", "attributes": {}}]
        respx.get("http://ha.local/api/states").mock(return_value=httpx.Response(200, json=states))

        client = HARestClient()
        client._base_url = "http://ha.local"
        client._client = httpx.AsyncClient(base_url="http://ha.local", headers={"Authorization": "Bearer test"})

        result = await client.get_states()
        assert result == states
        await client.close()

    @respx.mock
    async def test_get_state_returns_single_entity(self):
        state = {"entity_id": "light.kitchen", "state": "on", "attributes": {}}
        respx.get("http://ha.local/api/states/light.kitchen").mock(return_value=httpx.Response(200, json=state))

        client = HARestClient()
        client._base_url = "http://ha.local"
        client._client = httpx.AsyncClient(base_url="http://ha.local", headers={})

        result = await client.get_state("light.kitchen")
        assert result["entity_id"] == "light.kitchen"
        await client.close()

    @respx.mock
    async def test_get_state_returns_none_on_404(self):
        respx.get("http://ha.local/api/states/light.missing").mock(return_value=httpx.Response(404))

        client = HARestClient()
        client._base_url = "http://ha.local"
        client._client = httpx.AsyncClient(base_url="http://ha.local", headers={})

        result = await client.get_state("light.missing")
        assert result is None
        await client.close()

    @respx.mock
    async def test_call_service_posts_correct_endpoint(self):
        respx.post("http://ha.local/api/services/light/turn_on").mock(return_value=httpx.Response(200, json=[]))

        client = HARestClient()
        client._base_url = "http://ha.local"
        client._client = httpx.AsyncClient(base_url="http://ha.local", headers={})

        result = await client.call_service("light", "turn_on", entity_id="light.kitchen")
        assert isinstance(result, list)
        await client.close()

    @respx.mock
    async def test_call_service_includes_service_data(self):
        route = respx.post("http://ha.local/api/services/light/turn_on").mock(return_value=httpx.Response(200, json=[]))

        client = HARestClient()
        client._base_url = "http://ha.local"
        client._client = httpx.AsyncClient(base_url="http://ha.local", headers={})

        await client.call_service("light", "turn_on", entity_id="light.kitchen", service_data={"brightness": 128})
        body = route.calls[0].request.content
        assert b"brightness" in body
        await client.close()

    @respx.mock
    async def test_fire_event_posts_correct_endpoint(self):
        respx.post("http://ha.local/api/events/test_event").mock(return_value=httpx.Response(200, json={"message": "ok"}))

        client = HARestClient()
        client._base_url = "http://ha.local"
        client._client = httpx.AsyncClient(base_url="http://ha.local", headers={})

        result = await client.fire_event("test_event", {"key": "value"})
        assert result["message"] == "ok"
        await client.close()

    @respx.mock
    async def test_test_connection_returns_true_on_200(self):
        respx.get("http://ha.local/api/").mock(return_value=httpx.Response(200, json={"message": "API running."}))

        client = HARestClient()
        client._base_url = "http://ha.local"
        client._client = httpx.AsyncClient(base_url="http://ha.local", headers={})

        result = await client.test_connection()
        assert result is True
        await client.close()

    @respx.mock
    async def test_test_connection_returns_false_on_error(self):
        respx.get("http://ha.local/api/").mock(side_effect=httpx.ConnectError("refused"))

        client = HARestClient()
        client._base_url = "http://ha.local"
        client._client = httpx.AsyncClient(base_url="http://ha.local", headers={})

        result = await client.test_connection()
        assert result is False
        await client.close()

    @respx.mock
    async def test_get_states_raises_on_server_error(self):
        respx.get("http://ha.local/api/states").mock(return_value=httpx.Response(500, text="Internal Server Error"))

        client = HARestClient()
        client._base_url = "http://ha.local"
        client._client = httpx.AsyncClient(base_url="http://ha.local", headers={})

        with pytest.raises(httpx.HTTPStatusError):
            await client.get_states()
        await client.close()

    @patch("app.ha_client.rest.SettingsRepository")
    @patch("app.ha_client.rest.get_auth_headers", new_callable=AsyncMock)
    async def test_initialize_creates_httpx_client(self, mock_auth, mock_settings):
        mock_settings.get_value = AsyncMock(return_value="http://ha.local:8123")
        mock_auth.return_value = {"Authorization": "Bearer test-token"}

        client = HARestClient()
        await client.initialize()

        assert client._base_url == "http://ha.local:8123"
        assert client._client is not None
        await client.close()


# ---------------------------------------------------------------------------
# Standalone test_ha_connection
# ---------------------------------------------------------------------------

class TestHaConnectionUtility:

    @respx.mock
    async def test_test_ha_connection_success(self):
        respx.get("http://ha.local:8123/api/").mock(return_value=httpx.Response(200))
        result = await _test_ha_connection("http://ha.local:8123", "test-token")
        assert result is True

    @respx.mock
    async def test_test_ha_connection_failure(self):
        respx.get("http://ha.local:8123/api/").mock(return_value=httpx.Response(401))
        result = await _test_ha_connection("http://ha.local:8123", "bad-token")
        assert result is False


# ---------------------------------------------------------------------------
# Auth module
# ---------------------------------------------------------------------------

class TestHAAuth:

    def test_build_auth_headers_format(self):
        headers = build_auth_headers("my-token-123")
        assert headers["Authorization"] == "Bearer my-token-123"
        assert headers["Content-Type"] == "application/json"

    @patch("app.ha_client.auth.retrieve_secret", new_callable=AsyncMock, return_value="stored-token")
    async def test_get_ha_token_returns_token(self, mock_retrieve):
        token = await get_ha_token()
        assert token == "stored-token"
        mock_retrieve.assert_awaited_once_with(HA_TOKEN_SECRET_KEY)

    @patch("app.ha_client.auth.retrieve_secret", new_callable=AsyncMock, return_value=None)
    async def test_get_ha_token_returns_none_when_not_set(self, mock_retrieve):
        token = await get_ha_token()
        assert token is None

    @patch("app.ha_client.auth.store_secret", new_callable=AsyncMock)
    async def test_set_ha_token_stores_token(self, mock_store):
        await set_ha_token("new-token")
        mock_store.assert_awaited_once_with(HA_TOKEN_SECRET_KEY, "new-token")

    @patch("app.ha_client.auth.retrieve_secret", new_callable=AsyncMock, return_value="tok")
    async def test_get_auth_headers_returns_dict(self, mock_retrieve):
        headers = await get_auth_headers()
        assert headers is not None
        assert headers["Authorization"] == "Bearer tok"

    @patch("app.ha_client.auth.retrieve_secret", new_callable=AsyncMock, return_value=None)
    async def test_get_auth_headers_returns_none_when_no_token(self, mock_retrieve):
        headers = await get_auth_headers()
        assert headers is None


# ---------------------------------------------------------------------------
# WebSocket client
# ---------------------------------------------------------------------------

class TestHAWebSocketClient:

    def test_initial_state_not_connected(self):
        ws = HAWebSocketClient()
        assert ws.is_connected() is False

    def test_next_id_increments(self):
        ws = HAWebSocketClient()
        id1 = ws._next_id()
        id2 = ws._next_id()
        assert id2 == id1 + 1

    def test_on_event_registers_callback(self):
        ws = HAWebSocketClient()
        callback = MagicMock()
        ws.on_event("state_changed", callback)
        assert "state_changed" in ws._listeners
        assert callback in ws._listeners["state_changed"]

    @patch("app.ha_client.websocket.SettingsRepository")
    @patch("app.ha_client.websocket.get_ha_token", new_callable=AsyncMock, return_value=None)
    async def test_connect_returns_false_when_no_token(self, mock_token, mock_settings):
        mock_settings.get_value = AsyncMock(return_value="http://ha.local")
        ws = HAWebSocketClient()
        result = await ws.connect()
        assert result is False

    @patch("app.ha_client.websocket.SettingsRepository")
    @patch("app.ha_client.websocket.get_ha_token", new_callable=AsyncMock, return_value="tok")
    async def test_connect_returns_false_when_no_url(self, mock_token, mock_settings):
        mock_settings.get_value = AsyncMock(return_value=None)
        ws = HAWebSocketClient()
        result = await ws.connect()
        assert result is False

    async def test_disconnect_sets_running_false(self):
        ws = HAWebSocketClient()
        ws._running = True
        await ws.disconnect()
        assert ws._running is False
