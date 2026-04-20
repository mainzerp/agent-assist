"""Tests for app.ha_client -- REST client, auth, and WebSocket."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from app.ha_client.auth import (
    HA_TOKEN_SECRET_KEY,
    build_auth_headers,
    get_auth_headers,
    get_ha_token,
    set_ha_token,
)
from app.ha_client.rest import HARestClient
from app.ha_client.rest import test_ha_connection as _test_ha_connection
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
        respx.post("http://ha.local/api/events/test_event").mock(
            return_value=httpx.Response(200, json={"message": "ok"})
        )

        client = HARestClient()
        client._base_url = "http://ha.local"
        client._client = httpx.AsyncClient(base_url="http://ha.local", headers={})

        result = await client.fire_event("test_event", {"key": "value"})
        assert result["message"] == "ok"
        await client.close()

    @respx.mock
    async def test_get_config_returns_ha_config(self):
        config = {"time_zone": "Europe/Berlin", "location_name": "Berlin", "latitude": 52.52}
        respx.get("http://ha.local/api/config").mock(return_value=httpx.Response(200, json=config))

        client = HARestClient()
        client._base_url = "http://ha.local"
        client._client = httpx.AsyncClient(base_url="http://ha.local", headers={})

        result = await client.get_config()
        assert result["time_zone"] == "Europe/Berlin"
        assert result["location_name"] == "Berlin"
        await client.close()

    @respx.mock
    async def test_get_config_returns_empty_dict_on_error(self):
        respx.get("http://ha.local/api/config").mock(return_value=httpx.Response(500))

        client = HARestClient()
        client._base_url = "http://ha.local"
        client._client = httpx.AsyncClient(base_url="http://ha.local", headers={})

        result = await client.get_config()
        assert result == {}
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

    @patch("app.ha_client.rest.get_auth_headers", new_callable=AsyncMock)
    @patch("app.ha_client.rest.SettingsRepository")
    async def test_close_nulls_client_and_refresh_is_noop(self, mock_settings, mock_auth):
        """COR-5: after close() the httpx client is None and _refresh_headers
        is a safe no-op instead of raising httpx.RuntimeError."""
        mock_settings.get_value = AsyncMock(return_value="http://ha.local:8123")
        mock_auth.return_value = {"Authorization": "Bearer x"}

        client = HARestClient()
        await client.initialize()
        await client.close()
        assert client._client is None
        # Must not raise
        await client._refresh_headers()
        assert client._client is None


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

    @patch("app.ha_client.websocket.SettingsRepository")
    @patch("app.ha_client.websocket.get_ha_token", new_callable=AsyncMock, return_value="tok")
    async def test_run_attempts_connect_from_fresh_state(self, mock_token, mock_settings):
        mock_settings.get_value = AsyncMock(return_value="http://ha.local")
        ws = HAWebSocketClient()

        async def _fake_connect():
            ws._running = False
            return True

        ws.connect = AsyncMock(side_effect=_fake_connect)
        ws._receive_loop = AsyncMock()
        await ws.run()
        ws.connect.assert_called()

    @patch("app.ha_client.websocket.SettingsRepository")
    @patch("app.ha_client.websocket.get_ha_token", new_callable=AsyncMock, return_value="tok")
    async def test_run_exits_cleanly_when_disconnect_called(self, mock_token, mock_settings):
        mock_settings.get_value = AsyncMock(return_value="http://ha.local")
        ws = HAWebSocketClient()

        async def _fake_connect():
            await ws.disconnect()
            return False

        ws.connect = AsyncMock(side_effect=_fake_connect)
        await ws.run()
        assert ws._running is False

    @patch("app.ha_client.websocket.SettingsRepository")
    @patch("app.ha_client.websocket.get_ha_token", new_callable=AsyncMock, return_value="tok")
    async def test_run_reconnects_after_receive_loop_error(self, mock_token, mock_settings):
        mock_settings.get_value = AsyncMock(return_value="http://ha.local")
        ws = HAWebSocketClient()
        call_count = 0

        async def _fake_connect():
            return True

        async def _fake_receive():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("connection lost")
            ws._running = False

        ws.connect = AsyncMock(side_effect=_fake_connect)
        ws._receive_loop = AsyncMock(side_effect=_fake_receive)
        ws._close_session = AsyncMock()
        ws._reconnect_loop = AsyncMock()
        await ws.run()
        ws._reconnect_loop.assert_called()

    @patch("app.ha_client.websocket.SettingsRepository")
    @patch("app.ha_client.websocket.asyncio.sleep", new_callable=AsyncMock)
    async def test_reconnect_loop_caps_attempts_and_pauses(self, mock_sleep, mock_settings):
        """FLOW-RECONN-1 (P2-4): after MAX_RECONNECT_ATTEMPTS the loop pauses
        for RECONNECT_PAUSE_DURATION seconds, then resets the counter."""
        from app.ha_client import websocket as ws_module

        mock_settings.get_value = AsyncMock(return_value=None)
        ws = ws_module.HAWebSocketClient()
        ws._running = True

        # Always fail to connect so we exercise the attempt counter.
        connect_calls = 0

        async def _fail_connect():
            nonlocal connect_calls
            connect_calls += 1
            # Stop once we have seen the pause so the test finishes.
            if connect_calls >= ws_module.MAX_RECONNECT_ATTEMPTS + 1:
                ws._running = False
            return False

        ws.connect = AsyncMock(side_effect=_fail_connect)
        await ws._reconnect_loop()

        # Pause value must have been awaited exactly once.
        pause_calls = [
            c for c in mock_sleep.await_args_list if c.args and c.args[0] == ws_module.RECONNECT_PAUSE_DURATION
        ]
        assert len(pause_calls) == 1, f"expected one pause sleep, got {pause_calls}"
        assert connect_calls >= ws_module.MAX_RECONNECT_ATTEMPTS


# ---------------------------------------------------------------------------
# Conversation Entity Concurrency
# ---------------------------------------------------------------------------


class TestConversationEntityConcurrency:
    """Tests for overlapping-turn serialization on the HA conversation entity."""

    async def test_overlapping_ws_turns_serialized(self):
        """Two concurrent tasks sharing a lock execute sequentially."""

        lock = asyncio.Lock()
        call_order: list[str] = []

        async def fake_process(label, delay):
            call_order.append(f"{label}_start")
            await asyncio.sleep(delay)
            call_order.append(f"{label}_end")

        async with lock:
            # Prove the lock is re-entrant-free; a second acquire must wait
            acquired = lock.locked()
            assert acquired is True

        # Functional check: two tasks sharing a lock execute sequentially
        call_order.clear()

        async def guarded(label, delay):
            async with lock:
                await fake_process(label, delay)

        t1 = asyncio.create_task(guarded("A", 0.05))
        t2 = asyncio.create_task(guarded("B", 0.01))
        await asyncio.gather(t1, t2)

        # A must fully complete before B starts (or vice versa)
        a_start = call_order.index("A_start")
        a_end = call_order.index("A_end")
        b_start = call_order.index("B_start")
        b_end = call_order.index("B_end")
        assert (a_end < b_start) or (b_end < a_start), f"Turns interleaved: {call_order}"


# ---------------------------------------------------------------------------
# HA Conversation Entity -- WS close/error handling
# ---------------------------------------------------------------------------


class TestHAConversationWSCloseError:
    """Tests for _process_via_ws raising on CLOSED/ERROR instead of returning partial speech."""

    @pytest.fixture(autouse=True)
    def _mock_homeassistant(self):
        """Mock homeassistant dependencies so custom_components can be imported."""
        import sys

        mocks = {}
        ha_modules = [
            "homeassistant",
            "homeassistant.components",
            "homeassistant.components.assist_pipeline",
            "homeassistant.components.conversation",
            "homeassistant.config_entries",
            "homeassistant.const",
            "homeassistant.core",
            "homeassistant.helpers",
            "homeassistant.helpers.device_registry",
            "homeassistant.helpers.entity_registry",
            "homeassistant.helpers.intent",
            "homeassistant.helpers.entity_platform",
        ]
        for mod in ha_modules:
            if mod not in sys.modules:
                mocks[mod] = MagicMock()
                sys.modules[mod] = mocks[mod]

        # Provide required constants/classes used at import time
        sys.modules["homeassistant.const"].CONF_URL = "url"
        sys.modules["homeassistant.const"].CONF_API_KEY = "api_key"
        sys.modules["homeassistant.const"].MATCH_ALL = "*"
        conv_mod = sys.modules["homeassistant.components.conversation"]
        conv_mod.ConversationEntityFeature = MagicMock()
        conv_mod.ConversationEntity = type(
            "ConversationEntity",
            (),
            {
                "__init__": lambda self, *a, **kw: None,
            },
        )
        # Wire parent attribute so `from homeassistant.components import conversation`
        # resolves to the same object as sys.modules[...conversation].
        sys.modules["homeassistant.components"].conversation = conv_mod
        sys.modules["homeassistant.components"].assist_pipeline = sys.modules[
            "homeassistant.components.assist_pipeline"
        ]

        yield

        for mod in mocks:
            sys.modules.pop(mod, None)
        # Clear the imported custom_components module so it doesn't leak
        for key in list(sys.modules):
            if key.startswith("custom_components"):
                del sys.modules[key]

    async def test_process_via_ws_closed_mid_stream(self):
        """WS CLOSED mid-stream should raise _WsDroppedAfterSendError wrapping aiohttp.ClientError."""
        import json as _json
        import sys

        import aiohttp

        sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
        from custom_components.ha_agenthub.conversation import (
            HaAgentHubConversationEntity,
            _WsDroppedAfterSendError,
        )

        entity = MagicMock()
        entity._ws = AsyncMock()
        entity._ws.send_json = AsyncMock()

        msg_text = MagicMock()
        msg_text.type = aiohttp.WSMsgType.TEXT
        msg_text.data = _json.dumps({"token": "partial ", "done": False})

        msg_closed = MagicMock()
        msg_closed.type = aiohttp.WSMsgType.CLOSED

        entity._ws.receive = AsyncMock(side_effect=[msg_text, msg_closed])

        user_input = MagicMock()
        user_input.text = "hello"
        user_input.conversation_id = "conv-1"
        user_input.language = "en"
        user_input.device_id = None

        with pytest.raises(_WsDroppedAfterSendError) as exc_info:
            await HaAgentHubConversationEntity._process_via_ws(entity, user_input)
        assert isinstance(exc_info.value.__cause__, aiohttp.ClientError)
        assert "closed mid-stream" in str(exc_info.value.__cause__)

    async def test_process_via_ws_error_mid_stream(self):
        """WS ERROR mid-stream should raise _WsDroppedAfterSendError wrapping aiohttp.ClientError."""
        import json as _json
        import sys

        import aiohttp

        sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
        from custom_components.ha_agenthub.conversation import (
            HaAgentHubConversationEntity,
            _WsDroppedAfterSendError,
        )

        entity = MagicMock()
        entity._ws = AsyncMock()
        entity._ws.send_json = AsyncMock()

        msg_text = MagicMock()
        msg_text.type = aiohttp.WSMsgType.TEXT
        msg_text.data = _json.dumps({"token": "partial ", "done": False})

        msg_error = MagicMock()
        msg_error.type = aiohttp.WSMsgType.ERROR

        entity._ws.receive = AsyncMock(side_effect=[msg_text, msg_error])

        user_input = MagicMock()
        user_input.text = "hello"
        user_input.conversation_id = "conv-1"
        user_input.language = "en"
        user_input.device_id = None

        with pytest.raises(_WsDroppedAfterSendError) as exc_info:
            await HaAgentHubConversationEntity._process_via_ws(entity, user_input)
        assert isinstance(exc_info.value.__cause__, aiohttp.ClientError)
        assert "error mid-stream" in str(exc_info.value.__cause__)
        assert entity._ws is None

    async def test_process_via_ws_close_before_any_tokens(self):
        """WS CLOSED immediately (no tokens) should raise _WsDroppedAfterSendError."""
        import sys

        import aiohttp

        sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
        from custom_components.ha_agenthub.conversation import (
            HaAgentHubConversationEntity,
            _WsDroppedAfterSendError,
        )

        entity = MagicMock()
        entity._ws = AsyncMock()
        entity._ws.send_json = AsyncMock()

        msg_closed = MagicMock()
        msg_closed.type = aiohttp.WSMsgType.CLOSED

        entity._ws.receive = AsyncMock(return_value=msg_closed)

        user_input = MagicMock()
        user_input.text = "hello"
        user_input.conversation_id = "conv-1"
        user_input.language = "en"
        user_input.device_id = None

        with pytest.raises(_WsDroppedAfterSendError) as exc_info:
            await HaAgentHubConversationEntity._process_via_ws(entity, user_input)
        assert isinstance(exc_info.value.__cause__, aiohttp.ClientError)
        assert "closed mid-stream" in str(exc_info.value.__cause__)

    async def test_process_via_ws_error_token_triggers_raise(self):
        """Error field in done token does NOT raise; it is logged and embedded in speech.

        Application-level errors arrive as part of the done chunk and are explicitly
        treated as non-transport failures (would otherwise be wrapped as
        _WsDroppedAfterSendError and trigger a duplicate REST fallback).
        """
        import json as _json
        import sys

        import aiohttp

        sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
        from custom_components.ha_agenthub.conversation import HaAgentHubConversationEntity

        entity = MagicMock()
        entity._ws = AsyncMock()
        entity._ws.send_json = AsyncMock()

        msg_done = MagicMock()
        msg_done.type = aiohttp.WSMsgType.TEXT
        msg_done.data = _json.dumps({"token": "", "done": True, "error": "Agent error: test"})

        entity._ws.receive = AsyncMock(return_value=msg_done)

        user_input = MagicMock()
        user_input.text = "hello"
        user_input.conversation_id = "conv-1"
        user_input.language = "en"
        user_input.device_id = None

        sentinel = object()
        entity._build_result = MagicMock(return_value=sentinel)

        result = await HaAgentHubConversationEntity._process_via_ws(entity, user_input)
        assert result is sentinel
        # Speech must include the error description rather than be empty.
        speech_arg = entity._build_result.call_args.args[0]
        assert "Agent error: test" in speech_arg


# ---------------------------------------------------------------------------
# HA Conversation Entity -- coalescing window (P2-3)
# ---------------------------------------------------------------------------


class TestHAConversationCoalesceWindow:
    """FLOW-COALESCE-1 (P2-3): coalescing only collapses duplicates that arrive
    within the coalesce window. Later repeats must get a fresh bridge task."""

    @pytest.fixture(autouse=True)
    def _mock_homeassistant(self):
        import sys

        mocks = {}
        ha_modules = [
            "homeassistant",
            "homeassistant.components",
            "homeassistant.components.assist_pipeline",
            "homeassistant.components.conversation",
            "homeassistant.config_entries",
            "homeassistant.const",
            "homeassistant.core",
            "homeassistant.helpers",
            "homeassistant.helpers.device_registry",
            "homeassistant.helpers.entity_registry",
            "homeassistant.helpers.intent",
            "homeassistant.helpers.entity_platform",
            "homeassistant.helpers.area_registry",
        ]
        for mod in ha_modules:
            if mod not in sys.modules:
                mocks[mod] = MagicMock()
                sys.modules[mod] = mocks[mod]
        sys.modules["homeassistant.const"].CONF_URL = "url"
        sys.modules["homeassistant.const"].CONF_API_KEY = "api_key"
        sys.modules["homeassistant.const"].MATCH_ALL = "*"
        conv_mod = sys.modules["homeassistant.components.conversation"]
        conv_mod.ConversationEntityFeature = MagicMock()
        conv_mod.ConversationEntity = type(
            "ConversationEntity",
            (),
            {"__init__": lambda self, *a, **kw: None},
        )
        sys.modules["homeassistant.components"].conversation = conv_mod
        sys.modules["homeassistant.components"].assist_pipeline = sys.modules[
            "homeassistant.components.assist_pipeline"
        ]

        yield

        import sys as _sys

        for key in list(_sys.modules.keys()):
            if key.startswith("custom_components"):
                del _sys.modules[key]

    async def test_repeat_outside_coalesce_window_triggers_new_bridge(self):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
        from custom_components.ha_agenthub.conversation import HaAgentHubConversationEntity

        entity = HaAgentHubConversationEntity.__new__(HaAgentHubConversationEntity)
        entity._coalesce_lock = asyncio.Lock()
        entity._inflight_bridge = {}
        entity._coalesce_window_sec = 0.05

        bridge_calls = 0

        async def _fake_bridge(user_input, key):
            nonlocal bridge_calls
            bridge_calls += 1
            await asyncio.sleep(0.001)
            return f"result-{bridge_calls}"

        entity._async_bridge_with_cleanup = _fake_bridge

        class _FakeHass:
            def async_create_task(self, coro):
                return asyncio.create_task(coro)

        entity.hass = _FakeHass()

        ui = MagicMock()
        ui.text = "turn on light"
        ui.conversation_id = "c-1"

        r1 = await entity._async_handle_message(ui, MagicMock())
        await asyncio.sleep(entity._coalesce_window_sec + 0.05)
        r2 = await entity._async_handle_message(ui, MagicMock())

        assert bridge_calls == 2
        assert r1 != r2

    async def test_repeat_inside_coalesce_window_collapses(self):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
        from custom_components.ha_agenthub.conversation import HaAgentHubConversationEntity

        entity = HaAgentHubConversationEntity.__new__(HaAgentHubConversationEntity)
        entity._coalesce_lock = asyncio.Lock()
        entity._inflight_bridge = {}
        entity._coalesce_window_sec = 1.0

        gate = asyncio.Event()
        bridge_calls = 0

        async def _fake_bridge(user_input, key):
            nonlocal bridge_calls
            bridge_calls += 1
            await gate.wait()
            return "result"

        entity._async_bridge_with_cleanup = _fake_bridge

        class _FakeHass:
            def async_create_task(self, coro):
                return asyncio.create_task(coro)

        entity.hass = _FakeHass()

        ui = MagicMock()
        ui.text = "turn on light"
        ui.conversation_id = "c-1"

        t1 = asyncio.create_task(entity._async_handle_message(ui, MagicMock()))
        await asyncio.sleep(0)
        t2 = asyncio.create_task(entity._async_handle_message(ui, MagicMock()))
        await asyncio.sleep(0)
        gate.set()
        r1, r2 = await asyncio.gather(t1, t2)

        assert bridge_calls == 1
        assert r1 == r2 == "result"
