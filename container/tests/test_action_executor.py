"""Tests for app.agents.action_executor -- parse_action and execute_action."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock litellm before importing app modules
_litellm_mock = MagicMock()


class _AuthenticationError(Exception):
    pass


_litellm_mock.exceptions.AuthenticationError = _AuthenticationError
sys.modules.setdefault("litellm", _litellm_mock)

from app.agents.action_executor import parse_action, execute_action  # noqa: E402


# ---------------------------------------------------------------------------
# parse_action tests
# ---------------------------------------------------------------------------

class TestParseAction:
    """Tests for parse_action()."""

    def test_fenced_json(self):
        response = (
            "Sure, I'll turn on the kitchen light.\n"
            '```json\n{"action": "turn_on", "entity": "kitchen light", "parameters": {}}\n```\n'
            "Turning on the kitchen light."
        )
        result = parse_action(response)
        assert result is not None
        assert result["action"] == "turn_on"
        assert result["entity"] == "kitchen light"
        assert result["parameters"] == {}

    def test_raw_json(self):
        response = (
            'Here you go: {"action": "turn_off", "entity": "bedroom lamp", "parameters": {}} '
            "Done."
        )
        result = parse_action(response)
        assert result is not None
        assert result["action"] == "turn_off"
        assert result["entity"] == "bedroom lamp"

    def test_no_json(self):
        response = "The kitchen light is currently on at 80% brightness."
        result = parse_action(response)
        assert result is None

    def test_malformed_json(self):
        response = '```json\n{"action": "turn_on", "entity": }\n```'
        result = parse_action(response)
        assert result is None

    def test_json_without_action_key(self):
        response = '```json\n{"command": "turn_on", "target": "lamp"}\n```'
        result = parse_action(response)
        assert result is None

    def test_brightness_action(self):
        response = '```json\n{"action": "set_brightness", "entity": "living room", "parameters": {"brightness": 128}}\n```'
        result = parse_action(response)
        assert result is not None
        assert result["action"] == "set_brightness"
        assert result["parameters"]["brightness"] == 128

    def test_color_action(self):
        response = '```json\n{"action": "set_color", "entity": "desk lamp", "parameters": {"color_name": "red"}}\n```'
        result = parse_action(response)
        assert result is not None
        assert result["action"] == "set_color"
        assert result["parameters"]["color_name"] == "red"

    def test_toggle_action(self):
        response = '{"action": "toggle", "entity": "hallway light", "parameters": {}}'
        result = parse_action(response)
        assert result is not None
        assert result["action"] == "toggle"


# ---------------------------------------------------------------------------
# execute_action tests
# ---------------------------------------------------------------------------

class TestExecuteAction:
    """Tests for execute_action() with mocked dependencies."""

    @pytest.fixture()
    def ha_client(self):
        client = AsyncMock()
        client.call_service = AsyncMock(return_value={})
        client.get_state = AsyncMock(return_value={"state": "on", "attributes": {}})
        return client

    @pytest.fixture()
    def entity_matcher(self):
        matcher = AsyncMock()
        match_result = MagicMock()
        match_result.entity_id = "light.kitchen_ceiling"
        match_result.friendly_name = "Kitchen Ceiling"
        matcher.match = AsyncMock(return_value=[match_result])
        return matcher

    @pytest.fixture()
    def entity_index(self):
        index = MagicMock()
        entry = MagicMock()
        entry.entity_id = "light.kitchen_ceiling"
        entry.friendly_name = "Kitchen Ceiling"
        index.search = MagicMock(return_value=[(entry, 0.1)])
        return index

    @pytest.mark.asyncio
    async def test_turn_on_success(self, ha_client, entity_matcher, entity_index):
        action = {"action": "turn_on", "entity": "kitchen light", "parameters": {}}
        result = await execute_action(action, ha_client, entity_index, entity_matcher)

        assert result["success"] is True
        assert result["entity_id"] == "light.kitchen_ceiling"
        assert result["new_state"] == "on"
        assert "Kitchen Ceiling" in result["speech"]
        ha_client.call_service.assert_awaited_once_with("light", "turn_on", "light.kitchen_ceiling", None)

    @pytest.mark.asyncio
    async def test_turn_off_success(self, ha_client, entity_matcher, entity_index):
        ha_client.get_state = AsyncMock(return_value={"state": "off", "attributes": {}})
        action = {"action": "turn_off", "entity": "kitchen light", "parameters": {}}
        result = await execute_action(action, ha_client, entity_index, entity_matcher)

        assert result["success"] is True
        assert result["new_state"] == "off"
        ha_client.call_service.assert_awaited_once_with("light", "turn_off", "light.kitchen_ceiling", None)

    @pytest.mark.asyncio
    async def test_set_brightness(self, ha_client, entity_matcher, entity_index):
        action = {"action": "set_brightness", "entity": "kitchen light", "parameters": {"brightness": 128}}
        result = await execute_action(action, ha_client, entity_index, entity_matcher)

        assert result["success"] is True
        ha_client.call_service.assert_awaited_once_with(
            "light", "turn_on", "light.kitchen_ceiling", {"brightness": 128}
        )

    @pytest.mark.asyncio
    async def test_unknown_action(self, ha_client, entity_matcher, entity_index):
        action = {"action": "explode", "entity": "kitchen light", "parameters": {}}
        result = await execute_action(action, ha_client, entity_index, entity_matcher)

        assert result["success"] is False
        assert "Unknown action" in result["speech"]
        ha_client.call_service.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_entity_not_found(self, ha_client, entity_index):
        matcher = AsyncMock()
        matcher.match = AsyncMock(return_value=[])
        entity_index.search = MagicMock(return_value=[])

        action = {"action": "turn_on", "entity": "nonexistent lamp", "parameters": {}}
        result = await execute_action(action, ha_client, entity_index, matcher)

        assert result["success"] is False
        assert "Could not find" in result["speech"]
        ha_client.call_service.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_service_call_failure(self, ha_client, entity_matcher, entity_index):
        ha_client.call_service = AsyncMock(side_effect=Exception("Connection refused"))
        action = {"action": "turn_on", "entity": "kitchen light", "parameters": {}}
        result = await execute_action(action, ha_client, entity_index, entity_matcher)

        assert result["success"] is False
        assert "Failed to execute" in result["speech"]

    @pytest.mark.asyncio
    async def test_fallback_to_entity_index(self, ha_client, entity_index):
        """When entity_matcher returns no results, fall back to entity_index.search."""
        matcher = AsyncMock()
        matcher.match = AsyncMock(return_value=[])

        action = {"action": "turn_on", "entity": "kitchen light", "parameters": {}}
        result = await execute_action(action, ha_client, entity_index, matcher)

        assert result["success"] is True
        assert result["entity_id"] == "light.kitchen_ceiling"

    @pytest.mark.asyncio
    async def test_no_matcher_uses_index(self, ha_client, entity_index):
        """When entity_matcher is None, fall back to entity_index."""
        action = {"action": "turn_on", "entity": "kitchen light", "parameters": {}}
        result = await execute_action(action, ha_client, entity_index, None)

        assert result["success"] is True
        assert result["entity_id"] == "light.kitchen_ceiling"

    @pytest.mark.asyncio
    async def test_state_verification_failure(self, ha_client, entity_matcher, entity_index):
        """State verification failure should not affect success status."""
        ha_client.get_state = AsyncMock(side_effect=Exception("timeout"))
        action = {"action": "turn_on", "entity": "kitchen light", "parameters": {}}
        result = await execute_action(action, ha_client, entity_index, entity_matcher)

        assert result["success"] is True
        assert result["new_state"] is None

    @pytest.mark.asyncio
    async def test_execute_action_passes_agent_id(self, ha_client, entity_matcher, entity_index):
        """Verify that entity_matcher.match is called with agent_id kwarg."""
        action = {"action": "turn_on", "entity": "kitchen light", "parameters": {}}
        result = await execute_action(action, ha_client, entity_index, entity_matcher, agent_id="light-agent")

        assert result["success"] is True
        entity_matcher.match.assert_awaited_once_with("kitchen light", agent_id="light-agent")


# ---------------------------------------------------------------------------
# execute_music_action tests
# ---------------------------------------------------------------------------

from app.agents.music_executor import execute_music_action  # noqa: E402


class TestMusicExecutor:
    """Tests for execute_music_action() with mocked dependencies."""

    @pytest.fixture()
    def ha_client(self):
        client = AsyncMock()
        client.call_service = AsyncMock(return_value={})
        client.get_state = AsyncMock(return_value={"state": "playing", "attributes": {}})
        return client

    @pytest.fixture()
    def entity_matcher(self):
        matcher = AsyncMock()
        match_result = MagicMock()
        match_result.entity_id = "media_player.ma_kitchen"
        match_result.friendly_name = "Kitchen Speaker"
        matcher.match = AsyncMock(return_value=[match_result])
        return matcher

    @pytest.fixture()
    def entity_index(self):
        index = MagicMock()
        entry = MagicMock()
        entry.entity_id = "media_player.ma_kitchen"
        entry.friendly_name = "Kitchen Speaker"
        index.search = MagicMock(return_value=[(entry, 0.1)])
        return index

    @pytest.mark.asyncio
    async def test_execute_play_media(self, ha_client, entity_matcher, entity_index):
        action = {"action": "play_media", "entity": "kitchen speaker", "parameters": {"media_id": "jazz", "media_type": "track", "enqueue": "play"}}
        result = await execute_music_action(action, ha_client, entity_index, entity_matcher)

        assert result["success"] is True
        assert result["entity_id"] == "media_player.ma_kitchen"
        ha_client.call_service.assert_awaited_once_with(
            "mass", "play_media", "media_player.ma_kitchen",
            {"media_id": "jazz", "media_type": "track", "enqueue": "play"},
        )

    @pytest.mark.asyncio
    async def test_execute_volume_set(self, ha_client, entity_matcher, entity_index):
        action = {"action": "volume_set", "entity": "kitchen speaker", "parameters": {"volume_level": 0.5}}
        result = await execute_music_action(action, ha_client, entity_index, entity_matcher)

        assert result["success"] is True
        ha_client.call_service.assert_awaited_once_with(
            "media_player", "volume_set", "media_player.ma_kitchen",
            {"volume_level": 0.5},
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action_name", ["media_play", "media_pause", "media_next_track", "media_previous_track"])
    async def test_execute_transport_controls(self, action_name, ha_client, entity_matcher, entity_index):
        action = {"action": action_name, "entity": "kitchen speaker", "parameters": {}}
        result = await execute_music_action(action, ha_client, entity_index, entity_matcher)

        assert result["success"] is True
        ha_client.call_service.assert_awaited_once_with(
            "media_player", action_name, "media_player.ma_kitchen", None,
        )

    @pytest.mark.asyncio
    async def test_execute_shuffle_set(self, ha_client, entity_matcher, entity_index):
        action = {"action": "shuffle_set", "entity": "kitchen speaker", "parameters": {"shuffle": True}}
        result = await execute_music_action(action, ha_client, entity_index, entity_matcher)

        assert result["success"] is True
        ha_client.call_service.assert_awaited_once_with(
            "media_player", "shuffle_set", "media_player.ma_kitchen",
            {"shuffle": True},
        )

    @pytest.mark.asyncio
    async def test_execute_repeat_set(self, ha_client, entity_matcher, entity_index):
        action = {"action": "repeat_set", "entity": "kitchen speaker", "parameters": {"repeat": "all"}}
        result = await execute_music_action(action, ha_client, entity_index, entity_matcher)

        assert result["success"] is True
        ha_client.call_service.assert_awaited_once_with(
            "media_player", "repeat_set", "media_player.ma_kitchen",
            {"repeat": "all"},
        )

    @pytest.mark.asyncio
    async def test_execute_search_returns_speech(self, ha_client, entity_matcher, entity_index):
        ha_client.call_service = AsyncMock(return_value=[
            {"name": "Jazz Suite", "artist": "Dave Brubeck"},
            {"name": "Blue Train", "artist": "John Coltrane"},
        ])
        action = {"action": "search", "entity": "kitchen speaker", "parameters": {"name": "jazz", "media_type": "track"}}
        result = await execute_music_action(action, ha_client, entity_index, entity_matcher)

        assert result["success"] is True
        assert result["new_state"] is None
        assert "Jazz Suite" in result["speech"]
        assert "Dave Brubeck" in result["speech"]

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, ha_client, entity_matcher, entity_index):
        action = {"action": "nonexistent", "entity": "kitchen speaker", "parameters": {}}
        result = await execute_music_action(action, ha_client, entity_index, entity_matcher)

        assert result["success"] is False
        assert "Unknown action" in result["speech"]
        ha_client.call_service.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_execute_entity_not_found(self, ha_client):
        matcher = AsyncMock()
        matcher.match = AsyncMock(return_value=[])
        index = MagicMock()
        index.search = MagicMock(return_value=[])

        action = {"action": "play_media", "entity": "nonexistent speaker", "parameters": {"media_id": "jazz"}}
        result = await execute_music_action(action, ha_client, index, matcher)

        assert result["success"] is False
        assert "Could not find" in result["speech"]
        ha_client.call_service.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_execute_service_call_failure(self, ha_client, entity_matcher, entity_index):
        ha_client.call_service = AsyncMock(side_effect=Exception("Connection refused"))
        action = {"action": "media_play", "entity": "kitchen speaker", "parameters": {}}
        result = await execute_music_action(action, ha_client, entity_index, entity_matcher)

        assert result["success"] is False
        assert "Failed" in result["speech"]

    @pytest.mark.asyncio
    async def test_entity_resolution_prefers_matcher(self, ha_client, entity_matcher, entity_index):
        action = {"action": "media_play", "entity": "kitchen speaker", "parameters": {}}
        result = await execute_music_action(action, ha_client, entity_index, entity_matcher)

        assert result["success"] is True
        entity_matcher.match.assert_awaited_once_with("kitchen speaker", agent_id=None)
        entity_index.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_music_action_passes_agent_id(self, ha_client, entity_matcher, entity_index):
        """Verify that entity_matcher.match is called with agent_id kwarg."""
        action = {"action": "media_play", "entity": "kitchen speaker", "parameters": {}}
        result = await execute_music_action(action, ha_client, entity_index, entity_matcher, agent_id="music-agent")

        assert result["success"] is True
        entity_matcher.match.assert_awaited_once_with("kitchen speaker", agent_id="music-agent")
