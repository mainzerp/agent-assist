"""Bridge tests for filler-first return and post-idle announce push."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest

ROOT = Path(__file__).resolve().parents[1].parent


class _FakeConversationResult:
    def __init__(self, response=None, conversation_id=None):
        self.response = response
        self.conversation_id = conversation_id


class _FakeIntentResponse:
    def __init__(self, language="en"):
        self.language = language
        self.speech = None

    def async_set_speech(self, speech):
        self.speech = speech


class _FakeConversationEntity:
    async def async_added_to_hass(self):
        return None

    async def async_will_remove_from_hass(self):
        return None


class _FakeHass:
    def __init__(self, service_side_effect=None):
        self.services = SimpleNamespace(async_call=AsyncMock(side_effect=service_side_effect))
        self._states: dict[str, SimpleNamespace] = {}
        self.states = SimpleNamespace(get=self._get_state)

    def async_create_task(self, coro):
        return asyncio.create_task(coro)

    def _get_state(self, entity_id):
        return self._states.get(entity_id)

    def set_state(self, entity_id, state):
        self._states[entity_id] = SimpleNamespace(state=state)


class _FakeEntry:
    def __init__(self, options=None):
        self.options = options or {}
        self.data = {}
        self.unload_callbacks = []
        self.created_tasks = []
        self.async_create_background_task = MagicMock(side_effect=self._create_background_task)

    def _create_background_task(self, hass, coro, name=None):
        task = asyncio.create_task(coro)
        self.created_tasks.append(task)
        return task

    def async_on_unload(self, callback):
        self.unload_callbacks.append(callback)
        return callback


class _StateChangeTracker:
    def __init__(self):
        self.records = []
        self.active_records = []

    def track(self, hass, entity_ids, callback):
        record = {
            "entity_ids": list(entity_ids),
            "callback": callback,
            "unsubscribe": Mock(),
        }
        self.records.append(record)
        self.active_records.append(record)

        def _unsubscribe():
            record["unsubscribe"]()
            if record in self.active_records:
                self.active_records.remove(record)

        return _unsubscribe

    def fire_state_change(self, entity_id, old_state_str, new_state_str):
        matched = False
        event = SimpleNamespace(
            data={
                "old_state": None if old_state_str is None else SimpleNamespace(state=old_state_str),
                "new_state": None if new_state_str is None else SimpleNamespace(state=new_state_str),
            }
        )
        for record in list(self.active_records):
            if entity_id in record["entity_ids"]:
                matched = True
                record["callback"](event)
        if not matched:
            raise AssertionError(f"state callback was not registered for {entity_id}")

    def unsubscribe_call_count(self, entity_id):
        return sum(
            record["unsubscribe"].call_count
            for record in self.records
            if entity_id in record["entity_ids"]
        )


class _FakeWebSocket:
    def __init__(self):
        self._queue = asyncio.Queue()
        self.send_json = AsyncMock()
        self.receive = AsyncMock(side_effect=self._receive)
        self.close = AsyncMock(side_effect=self._close)
        self.closed = False

    def push(self, message):
        self._queue.put_nowait(message)

    async def _receive(self):
        return await self._queue.get()

    async def _close(self):
        self.closed = True


def _import_conversation_module():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from custom_components.ha_agenthub import conversation as conversation_module

    return conversation_module


def _ws_text(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(payload))


def _ws_closed() -> SimpleNamespace:
    return SimpleNamespace(type=aiohttp.WSMsgType.CLOSED)


def _user_input(
    *,
    text="hello",
    conversation_id="conv-1",
    language="en",
    device_id="device-a",
    area_id=None,
):
    return SimpleNamespace(
        text=text,
        conversation_id=conversation_id,
        language=language,
        device_id=device_id,
        area_id=area_id,
    )


def _only_push_task(entity):
    assert len(entity._inflight_pushes) == 1
    return next(iter(entity._inflight_pushes.values()))


def _speech_text(result):
    return result.response.speech


async def _wait_for_listener(tracker: _StateChangeTracker, entity_id: str) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 1.0
    while loop.time() < deadline:
        if any(entity_id in record["entity_ids"] for record in tracker.active_records):
            return
        await asyncio.sleep(0)
    raise AssertionError(f"state callback was not registered for {entity_id}")


def _make_entity(conversation_module, *, service_side_effect=None, build_result=None, options=None):
    entity_cls = conversation_module.HaAgentHubConversationEntity
    entity = entity_cls.__new__(entity_cls)
    entity.hass = _FakeHass(service_side_effect)
    entity._entry = _FakeEntry(options=options)
    entity._ws = None
    entity._session = None
    entity._ws_lock = asyncio.Lock()
    entity._coalesce_lock = asyncio.Lock()
    entity._inflight_bridge = {}
    entity._coalesce_window_sec = 0.25
    entity._inflight_pushes = {}
    entity._push_in_progress_satellites = set()
    entity._resolve_origin_context = MagicMock(return_value={})
    entity._is_native_plain_timers_enabled = MagicMock(return_value=False)
    entity._ws_last_active = 0.0
    entity._build_result = MagicMock(return_value=build_result if build_result is not None else object())
    entity._reconnect_loop = AsyncMock(return_value=None)
    entity._disconnect_ws = AsyncMock()
    entity.entity_id = "conversation.test"
    return entity


class TestHABridgeFillerWait:
    @pytest.fixture(autouse=True)
    def _mock_homeassistant(self):
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
            "homeassistant.helpers.area_registry",
            "homeassistant.helpers.device_registry",
            "homeassistant.helpers.entity_registry",
            "homeassistant.helpers.intent",
            "homeassistant.helpers.event",
            "homeassistant.helpers.entity_platform",
        ]
        for mod in ha_modules:
            if mod not in sys.modules:
                mocks[mod] = MagicMock()
                sys.modules[mod] = mocks[mod]

        sys.modules["homeassistant.const"].CONF_URL = "url"
        sys.modules["homeassistant.const"].CONF_API_KEY = "api_key"
        sys.modules["homeassistant.const"].MATCH_ALL = "*"
        sys.modules["homeassistant.config_entries"].ConfigEntry = type("ConfigEntry", (), {})
        sys.modules["homeassistant.core"].HomeAssistant = type("HomeAssistant", (), {})
        sys.modules["homeassistant.helpers.entity_platform"].AddConfigEntryEntitiesCallback = MagicMock()
        sys.modules["homeassistant.helpers.event"].async_track_state_change_event = MagicMock()
        sys.modules["homeassistant.helpers.intent"].IntentResponse = _FakeIntentResponse
        helpers_mod = sys.modules["homeassistant.helpers"]
        helpers_mod.area_registry = sys.modules["homeassistant.helpers.area_registry"]
        helpers_mod.device_registry = sys.modules["homeassistant.helpers.device_registry"]
        helpers_mod.entity_registry = sys.modules["homeassistant.helpers.entity_registry"]
        helpers_mod.intent = sys.modules["homeassistant.helpers.intent"]
        helpers_mod.event = sys.modules["homeassistant.helpers.event"]
        helpers_mod.entity_platform = sys.modules["homeassistant.helpers.entity_platform"]
        conv_mod = sys.modules["homeassistant.components.conversation"]
        conv_mod.ConversationEntityFeature = SimpleNamespace(CONTROL=1)
        conv_mod.ConversationEntity = _FakeConversationEntity
        conv_mod.ConversationResult = _FakeConversationResult
        sys.modules["homeassistant.components"].conversation = conv_mod
        sys.modules["homeassistant.components"].assist_pipeline = sys.modules[
            "homeassistant.components.assist_pipeline"
        ]
        sys.modules["homeassistant.components.assist_pipeline"].async_migrate_engine = MagicMock()

        yield

        for mod in mocks:
            sys.modules.pop(mod, None)
        for key in list(sys.modules):
            if key.startswith("custom_components"):
                del sys.modules[key]

    async def test_filler_first_return_spawns_push_and_returns_filler_speech(self, caplog):
        conversation_module = _import_conversation_module()
        caplog.set_level(logging.INFO)
        tracker = _StateChangeTracker()
        entity = _make_entity(conversation_module)
        entity._resolve_satellite_entity = MagicMock(return_value="assist_satellite.flur")
        ws = _FakeWebSocket()
        ws.push(_ws_text({"token": "**Moment**, ich _pruefe_ das", "is_filler": True}))
        entity._ws = ws
        user_input = _user_input()

        with patch.object(conversation_module, "async_track_state_change_event", tracker.track):
            result = await conversation_module.HaAgentHubConversationEntity._process_via_ws(entity, user_input)

            assert _speech_text(result) == "Moment, ich pruefe das"
            assert entity._ws is None
            task = entity._inflight_pushes["assist_satellite.flur"]

            await _wait_for_listener(tracker, "assist_satellite.flur")
            ws.push(_ws_text({"token": "Das Licht ist an.", "done": True, "sanitized": True}))
            await asyncio.sleep(0)
            tracker.fire_state_change("assist_satellite.flur", "responding", "idle")
            await task

        call = entity.hass.services.async_call.await_args
        assert call.args == (
            "assist_satellite",
            "announce",
            {
                "entity_id": "assist_satellite.flur",
                "message": "Das Licht ist an.",
                "preannounce": False,
            },
        )
        assert call.kwargs == {"blocking": False}
        assert entity._inflight_pushes == {}
        assert "ha-agenthub: filler-first return" in caplog.text
        assert "ha-agenthub: post-filler push received final" in caplog.text
        assert "ha-agenthub: post-filler push dispatching announce" in caplog.text

    async def test_no_filler_returns_final_directly_no_push(self):
        conversation_module = _import_conversation_module()
        result_sentinel = object()
        entity = _make_entity(conversation_module, build_result=result_sentinel)
        ws = _FakeWebSocket()
        ws.push(_ws_text({"token": "Done.", "done": True, "sanitized": True}))
        entity._ws = ws
        user_input = _user_input(device_id=None)

        result = await conversation_module.HaAgentHubConversationEntity._process_via_ws(entity, user_input)

        assert result is result_sentinel
        entity._build_result.assert_called_once_with("Done.", "conv-1", "en", sanitized=True)
        entity._entry.async_create_background_task.assert_not_called()
        assert entity._inflight_pushes == {}

    async def test_push_aborts_when_new_turn_detected(self, caplog):
        conversation_module = _import_conversation_module()
        tracker = _StateChangeTracker()
        caplog.set_level(logging.INFO)
        entity = _make_entity(conversation_module)
        entity._resolve_satellite_entity = MagicMock(return_value="assist_satellite.flur")
        ws = _FakeWebSocket()
        ws.push(_ws_text({"token": "Moment, ich pruefe das", "is_filler": True}))
        entity._ws = ws
        user_input = _user_input()

        with patch.object(conversation_module, "async_track_state_change_event", tracker.track):
            await conversation_module.HaAgentHubConversationEntity._process_via_ws(entity, user_input)
            task = _only_push_task(entity)

            await _wait_for_listener(tracker, "assist_satellite.flur")
            tracker.fire_state_change("assist_satellite.flur", "responding", "idle")
            tracker.fire_state_change("assist_satellite.flur", "idle", "listening")
            ws.push(_ws_text({"token": "Das Licht ist an.", "done": True, "sanitized": True}))
            await task

        entity.hass.services.async_call.assert_not_awaited()
        assert entity._inflight_pushes == {}
        assert "ha-agenthub: abandoning post-filler push (new turn detected)" in caplog.text

    async def test_push_times_out_when_satellite_never_idle(self, caplog):
        conversation_module = _import_conversation_module()
        tracker = _StateChangeTracker()
        caplog.set_level(logging.WARNING)
        entity = _make_entity(conversation_module)
        entity._resolve_satellite_entity = MagicMock(return_value="assist_satellite.flur")
        ws = _FakeWebSocket()
        ws.push(_ws_text({"token": "Moment, ich pruefe das", "is_filler": True}))
        entity._ws = ws

        with (
            patch.object(conversation_module, "MAX_POST_FILLER_WAIT_SECONDS", 0.05),
            patch.object(conversation_module, "async_track_state_change_event", tracker.track),
        ):
            await conversation_module.HaAgentHubConversationEntity._process_via_ws(entity, _user_input())
            task = _only_push_task(entity)
            ws.push(_ws_text({"token": "Das Licht ist an.", "done": True, "sanitized": True}))
            await task

        entity.hass.services.async_call.assert_not_awaited()
        assert "ha-agenthub: post-filler push satellite never reached idle within" in caplog.text

    async def test_two_satellites_each_get_their_own_push(self):
        conversation_module = _import_conversation_module()
        tracker = _StateChangeTracker()
        entity = _make_entity(conversation_module)
        satellite_map = {
            "dev-a": "assist_satellite.a",
            "dev-b": "assist_satellite.b",
        }
        entity._resolve_satellite_entity = MagicMock(
            side_effect=lambda user_input: satellite_map[user_input.device_id]
        )

        ws_a = _FakeWebSocket()
        ws_a.push(_ws_text({"token": "Moment", "is_filler": True}))
        entity._ws = ws_a
        user_a = _user_input(device_id="dev-a", conversation_id="conv-a")

        ws_b = _FakeWebSocket()
        ws_b.push(_ws_text({"token": "Bitte warten", "is_filler": True}))
        user_b = _user_input(device_id="dev-b", conversation_id="conv-b")

        with patch.object(conversation_module, "async_track_state_change_event", tracker.track):
            await conversation_module.HaAgentHubConversationEntity._process_via_ws(entity, user_a)

            entity._ws = ws_b
            await conversation_module.HaAgentHubConversationEntity._process_via_ws(entity, user_b)

            assert set(entity._inflight_pushes) == {"assist_satellite.a", "assist_satellite.b"}
            task_a = entity._inflight_pushes["assist_satellite.a"]
            task_b = entity._inflight_pushes["assist_satellite.b"]
            assert task_a is not task_b

            await _wait_for_listener(tracker, "assist_satellite.a")
            await _wait_for_listener(tracker, "assist_satellite.b")
            ws_a.push(_ws_text({"token": "Antwort A", "done": True, "sanitized": True}))
            ws_b.push(_ws_text({"token": "Antwort B", "done": True, "sanitized": True}))
            await asyncio.sleep(0)
            tracker.fire_state_change("assist_satellite.a", "responding", "idle")
            tracker.fire_state_change("assist_satellite.b", "responding", "idle")
            await asyncio.gather(task_a, task_b)

        calls = entity.hass.services.async_call.await_args_list
        assert len(calls) == 2
        assert calls[0].args[2]["entity_id"] == "assist_satellite.a"
        assert calls[0].args[2]["message"] == "Antwort A"
        assert calls[1].args[2]["entity_id"] == "assist_satellite.b"
        assert calls[1].args[2]["message"] == "Antwort B"

    async def test_filler_only_no_final_default_no_announce(self, caplog):
        conversation_module = _import_conversation_module()
        tracker = _StateChangeTracker()
        caplog.set_level(logging.WARNING)
        entity = _make_entity(conversation_module)
        entity._resolve_satellite_entity = MagicMock(return_value="assist_satellite.flur")
        ws = _FakeWebSocket()
        ws.push(_ws_text({"token": "Moment", "is_filler": True}))
        entity._ws = ws

        with patch.object(conversation_module, "async_track_state_change_event", tracker.track):
            await conversation_module.HaAgentHubConversationEntity._process_via_ws(entity, _user_input())
            task = _only_push_task(entity)
            ws.push(_ws_closed())
            await task

        entity.hass.services.async_call.assert_not_awaited()
        ws.close.assert_awaited_once()
        assert "ha-agenthub: post-filler push WS closed before final" in caplog.text

    async def test_filler_only_no_final_opt_in_pushes_error_utterance(self):
        conversation_module = _import_conversation_module()
        tracker = _StateChangeTracker()
        entity = _make_entity(conversation_module)
        entity._resolve_satellite_entity = MagicMock(return_value="assist_satellite.flur")
        ws = _FakeWebSocket()
        ws.push(_ws_text({"token": "Moment", "is_filler": True}))
        entity._ws = ws

        with (
            patch.object(conversation_module, "PUSH_FINAL_WAIT_SECONDS", 0.05),
            patch.object(conversation_module, "SPEAK_FILLER_ONLY_ON_TIMEOUT", True),
            patch.object(conversation_module, "async_track_state_change_event", tracker.track),
        ):
            await conversation_module.HaAgentHubConversationEntity._process_via_ws(entity, _user_input())
            task = _only_push_task(entity)
            await _wait_for_listener(tracker, "assist_satellite.flur")
            tracker.fire_state_change("assist_satellite.flur", "responding", "idle")
            await task

        call = entity.hass.services.async_call.await_args
        assert call.args[2]["message"] == conversation_module.FILLER_ONLY_TIMEOUT_TEXT
        assert call.args[2]["preannounce"] is False

    async def test_secondary_filler_in_push_is_ignored(self, caplog):
        conversation_module = _import_conversation_module()
        tracker = _StateChangeTracker()
        caplog.set_level(logging.INFO)
        entity = _make_entity(conversation_module)
        entity._resolve_satellite_entity = MagicMock(return_value="assist_satellite.flur")
        ws = _FakeWebSocket()
        ws.push(_ws_text({"token": "Moment", "is_filler": True}))
        entity._ws = ws

        with patch.object(conversation_module, "async_track_state_change_event", tracker.track):
            await conversation_module.HaAgentHubConversationEntity._process_via_ws(entity, _user_input())
            task = _only_push_task(entity)
            await _wait_for_listener(tracker, "assist_satellite.flur")
            tracker.fire_state_change("assist_satellite.flur", "responding", "idle")
            ws.push(_ws_text({"token": "Noch ein filler", "is_filler": True}))
            ws.push(_ws_text({"token": "Und noch einer", "is_filler": True}))
            ws.push(_ws_text({"token": "Finale Antwort", "done": True, "sanitized": True}))
            await task

        call = entity.hass.services.async_call.await_args
        assert call.args[2]["message"] == "Finale Antwort"
        assert caplog.text.count("ha-agenthub: ignoring secondary filler in push") >= 2

    async def test_directive_short_circuit_skips_push(self, caplog):
        conversation_module = _import_conversation_module()
        tracker = _StateChangeTracker()
        caplog.set_level(logging.INFO)
        entity = _make_entity(conversation_module)
        entity._resolve_satellite_entity = MagicMock(return_value="assist_satellite.flur")
        ws = _FakeWebSocket()
        ws.push(_ws_text({"token": "Moment", "is_filler": True}))
        entity._ws = ws

        with patch.object(conversation_module, "async_track_state_change_event", tracker.track):
            await conversation_module.HaAgentHubConversationEntity._process_via_ws(entity, _user_input())
            task = _only_push_task(entity)
            ws.push(
                _ws_text(
                    {
                        "token": "",
                        "done": True,
                        "directive": "delegate_native_plain_timer",
                        "reason": "native_timer",
                    }
                )
            )
            await task

        entity.hass.services.async_call.assert_not_awaited()
        assert "ha-agenthub: post-filler push received directive, skipping announce" in caplog.text

    async def test_dropped_stream_no_push_spawned(self):
        conversation_module = _import_conversation_module()
        entity = _make_entity(conversation_module)
        ws = _FakeWebSocket()
        ws.push(_ws_closed())
        entity._ws = ws

        with pytest.raises(conversation_module._WsDroppedAfterSendError):
            await conversation_module.HaAgentHubConversationEntity._process_via_ws(entity, _user_input())

        assert entity._inflight_pushes == {}
        entity._entry.async_create_background_task.assert_not_called()

    async def test_entry_unload_cancels_inflight_push(self, caplog):
        conversation_module = _import_conversation_module()
        tracker = _StateChangeTracker()
        caplog.set_level(logging.INFO)
        entity = _make_entity(conversation_module)
        entity._resolve_satellite_entity = MagicMock(return_value="assist_satellite.flur")
        await conversation_module.HaAgentHubConversationEntity.async_added_to_hass(entity)

        ws = _FakeWebSocket()
        ws.push(_ws_text({"token": "Moment", "is_filler": True}))
        entity._ws = ws

        with patch.object(conversation_module, "async_track_state_change_event", tracker.track):
            await conversation_module.HaAgentHubConversationEntity._process_via_ws(entity, _user_input())
            task = _only_push_task(entity)
            await _wait_for_listener(tracker, "assist_satellite.flur")
            for callback in entity._entry.unload_callbacks:
                callback()
            await asyncio.gather(task, return_exceptions=True)

        assert task.cancelled()
        assert entity._inflight_pushes == {}
        assert "ha-agenthub: post-filler push cancelled" in caplog.text

    async def test_consecutive_turns_cancel_previous_push(self, caplog):
        conversation_module = _import_conversation_module()
        tracker = _StateChangeTracker()
        caplog.set_level(logging.INFO)
        entity = _make_entity(conversation_module)
        entity._resolve_satellite_entity = MagicMock(return_value="assist_satellite.flur")

        ws_one = _FakeWebSocket()
        ws_one.push(_ws_text({"token": "Moment", "is_filler": True}))
        entity._ws = ws_one

        ws_two = _FakeWebSocket()
        ws_two.push(_ws_text({"token": "Noch einen Moment", "is_filler": True}))

        with patch.object(conversation_module, "async_track_state_change_event", tracker.track):
            await conversation_module.HaAgentHubConversationEntity._process_via_ws(
                entity,
                _user_input(conversation_id="conv-1"),
            )
            first_task = _only_push_task(entity)

            entity._ws = ws_two
            await conversation_module.HaAgentHubConversationEntity._process_via_ws(
                entity,
                _user_input(conversation_id="conv-2"),
            )
            second_task = _only_push_task(entity)
            await asyncio.gather(first_task, return_exceptions=True)

            tracker.fire_state_change("assist_satellite.flur", "responding", "idle")
            ws_two.push(_ws_text({"token": "Final", "done": True, "sanitized": True}))
            await second_task

        assert first_task.cancelled()
        assert second_task is entity._entry.created_tasks[-1]
        assert "ha-agenthub: cancelling previous post-filler push" in caplog.text

    async def test_ws_ownership_transferred_foreground_clears_self_ws(self):
        conversation_module = _import_conversation_module()
        tracker = _StateChangeTracker()
        entity = _make_entity(conversation_module)
        entity._resolve_satellite_entity = MagicMock(return_value="assist_satellite.flur")
        ws = _FakeWebSocket()
        ws.push(_ws_text({"token": "Moment", "is_filler": True}))
        entity._ws = ws

        with patch.object(conversation_module, "async_track_state_change_event", tracker.track):
            await conversation_module.HaAgentHubConversationEntity._process_via_ws(entity, _user_input())
            assert entity._ws is None

            task = _only_push_task(entity)
            await _wait_for_listener(tracker, "assist_satellite.flur")
            tracker.fire_state_change("assist_satellite.flur", "responding", "idle")
            ws.push(_ws_text({"token": "Final", "done": True, "sanitized": True}))
            await task

        assert ws.receive.await_count >= 2

    async def test_satellite_already_idle_seeds_observed_idle(self):
        conversation_module = _import_conversation_module()
        tracker = _StateChangeTracker()
        entity = _make_entity(conversation_module)
        entity._resolve_satellite_entity = MagicMock(return_value="assist_satellite.flur")
        entity.hass.set_state("assist_satellite.flur", "idle")
        ws = _FakeWebSocket()
        ws.push(_ws_text({"token": "Moment", "is_filler": True}))
        entity._ws = ws

        with patch.object(conversation_module, "async_track_state_change_event", tracker.track):
            await conversation_module.HaAgentHubConversationEntity._process_via_ws(entity, _user_input())
            task = _only_push_task(entity)
            ws.push(_ws_text({"token": "Final", "done": True, "sanitized": True}))
            await task

        entity.hass.services.async_call.assert_awaited_once()

    async def test_filler_text_is_markdown_stripped_in_foreground_result(self):
        conversation_module = _import_conversation_module()
        tracker = _StateChangeTracker()
        entity = _make_entity(conversation_module)
        entity._resolve_satellite_entity = MagicMock(return_value="assist_satellite.flur")
        ws = _FakeWebSocket()
        ws.push(_ws_text({"token": "**Moment**, ich _pruefe_ das", "is_filler": True}))
        entity._ws = ws

        with patch.object(conversation_module, "async_track_state_change_event", tracker.track):
            result = await conversation_module.HaAgentHubConversationEntity._process_via_ws(entity, _user_input())
            task = _only_push_task(entity)
            await _wait_for_listener(tracker, "assist_satellite.flur")
            tracker.fire_state_change("assist_satellite.flur", "responding", "idle")
            ws.push(_ws_text({"token": "**Das** Licht ist _an_.", "done": True, "sanitized": False}))
            await task

        assert _speech_text(result) == "Moment, ich pruefe das"
        call = entity.hass.services.async_call.await_args
        assert call.args[2]["message"] == "Das Licht ist an."

    async def test_unresolvable_satellite_skips_announce_logs_warning(self, caplog):
        conversation_module = _import_conversation_module()
        caplog.set_level(logging.WARNING)
        entity = _make_entity(conversation_module)
        entity._resolve_satellite_entity = MagicMock(return_value=None)
        ws = _FakeWebSocket()
        ws.push(_ws_text({"token": "Moment", "is_filler": True}))
        entity._ws = ws

        result = await conversation_module.HaAgentHubConversationEntity._process_via_ws(entity, _user_input())
        task = _only_push_task(entity)
        ws.push(_ws_text({"token": "Final", "done": True, "sanitized": True}))
        await task

        assert _speech_text(result) == "Moment"
        entity.hass.services.async_call.assert_not_awaited()
        assert "ha-agenthub: post-filler push has final but no satellite to announce on" in caplog.text
