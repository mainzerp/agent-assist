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
from app.entity.index import EntityIndex  # noqa: E402
from app.entity.matcher import EntityMatcher  # noqa: E402
from tests.helpers import make_entity_index_entry  # noqa: E402


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

    def test_brace_in_string_literal_does_not_break_parsing(self):
        """COR-10: braces inside string literals must not confuse the
        balanced-brace scanner. The first ``{`` in the description must
        not be treated as a JSON object start."""
        response = (
            'Sure thing. {"action": "turn_on", "entity": "kitchen light", '
            '"parameters": {"note": "use {placeholder} value"}}'
        )
        result = parse_action(response)
        assert result is not None
        assert result["action"] == "turn_on"
        assert result["entity"] == "kitchen light"
        assert result["parameters"]["note"] == "use {placeholder} value"


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
    async def test_no_fallback_to_entity_index(self, ha_client, entity_index):
        """When entity_matcher returns no results, should NOT fall back to unfiltered entity_index."""
        matcher = AsyncMock()
        matcher.match = AsyncMock(return_value=[])

        action = {"action": "turn_on", "entity": "kitchen light", "parameters": {}}
        result = await execute_action(action, ha_client, entity_index, matcher)

        assert result["success"] is False
        assert "Could not find" in result["speech"]
        entity_index.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_matcher_no_fallback(self, ha_client, entity_index):
        """When entity_matcher is None, should NOT fall back to unfiltered entity_index."""
        action = {"action": "turn_on", "entity": "kitchen light", "parameters": {}}
        result = await execute_action(action, ha_client, entity_index, None)

        assert result["success"] is False
        assert "Could not find" in result["speech"]

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

    @pytest.mark.asyncio
    async def test_domain_validation_rejects_wrong_domain(self, ha_client):
        """Resolved entity in wrong domain should be treated as not found."""
        matcher = AsyncMock()
        match_result = MagicMock()
        match_result.entity_id = "media_player.living_room_tv"
        match_result.friendly_name = "Living Room TV"
        matcher.match = AsyncMock(return_value=[match_result])
        index = MagicMock()

        action = {"action": "turn_on", "entity": "living room", "parameters": {}}
        result = await execute_action(action, ha_client, index, matcher, agent_id="light-agent")

        assert result["success"] is False
        assert "Could not find" in result["speech"]

    @pytest.mark.asyncio
    async def test_domain_validation_accepts_light_domain(self, ha_client, entity_matcher, entity_index):
        """Entity in light domain should pass domain validation."""
        action = {"action": "turn_on", "entity": "kitchen light", "parameters": {}}
        result = await execute_action(action, ha_client, entity_index, entity_matcher)

        assert result["success"] is True
        assert result["entity_id"] == "light.kitchen_ceiling"

    @pytest.mark.asyncio
    async def test_domain_validation_accepts_switch_domain(self, ha_client):
        """Entity in switch domain should pass domain validation."""
        matcher = AsyncMock()
        match_result = MagicMock()
        match_result.entity_id = "switch.kitchen_outlet"
        match_result.friendly_name = "Kitchen Outlet"
        matcher.match = AsyncMock(return_value=[match_result])
        index = MagicMock()

        ha_client.get_state = AsyncMock(return_value={"state": "on", "attributes": {}})
        action = {"action": "turn_on", "entity": "kitchen outlet", "parameters": {}}
        result = await execute_action(action, ha_client, index, matcher)

        assert result["success"] is True
        assert result["entity_id"] == "switch.kitchen_outlet"

    @pytest.mark.asyncio
    async def test_exact_friendly_name_resolves_without_hybrid_match(self, ha_client):
        matcher = MagicMock(spec=EntityMatcher)
        matcher.match = AsyncMock(return_value=[])
        matcher.filter_visible_results = AsyncMock(side_effect=lambda agent_id, results: results)
        index = MagicMock(spec=EntityIndex)
        index.get_by_id.return_value = None
        index.list_entries_async = AsyncMock(
            return_value=[make_entity_index_entry("light.keller", "Keller", area="Basement")]
        )

        action = {"action": "turn_on", "entity": "Keller", "parameters": {}}
        result = await execute_action(action, ha_client, index, matcher, agent_id="light-agent")

        assert result["success"] is True
        assert result["entity_id"] == "light.keller"
        matcher.match.assert_not_awaited()
        ha_client.call_service.assert_awaited_once_with("light", "turn_on", "light.keller", None)

    @pytest.mark.asyncio
    async def test_trailing_device_noun_resolves_exact_name(self, ha_client):
        matcher = MagicMock(spec=EntityMatcher)
        matcher.match = AsyncMock(return_value=[])
        matcher.filter_visible_results = AsyncMock(side_effect=lambda agent_id, results: results)
        index = MagicMock(spec=EntityIndex)
        index.get_by_id.return_value = None
        index.list_entries_async = AsyncMock(
            return_value=[make_entity_index_entry("light.keller", "Keller", area="Basement")]
        )

        action = {"action": "turn_on", "entity": "Keller light", "parameters": {}}
        result = await execute_action(action, ha_client, index, matcher, agent_id="light-agent")

        assert result["success"] is True
        assert result["entity_id"] == "light.keller"
        matcher.match.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exact_entity_id_query_resolves_directly(self, ha_client):
        matcher = MagicMock(spec=EntityMatcher)
        matcher.match = AsyncMock(return_value=[])
        matcher.filter_visible_results = AsyncMock(side_effect=lambda agent_id, results: results)
        index = MagicMock(spec=EntityIndex)
        index.get_by_id.return_value = make_entity_index_entry("light.keller", "Keller", area="Basement")
        index.list_entries_async = AsyncMock(return_value=[])

        action = {"action": "turn_on", "entity": "light.keller", "parameters": {}}
        result = await execute_action(action, ha_client, index, matcher, agent_id="light-agent")

        assert result["success"] is True
        assert result["entity_id"] == "light.keller"
        index.get_by_id.assert_called_once_with("light.keller")
        matcher.match.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_area_fallback_returns_ambiguity_for_multiple_lights(self, ha_client):
        matcher = MagicMock(spec=EntityMatcher)
        matcher.match = AsyncMock(return_value=[])
        matcher.filter_visible_results = AsyncMock(side_effect=lambda agent_id, results: results)
        index = MagicMock(spec=EntityIndex)
        index.get_by_id.return_value = None
        index.list_entries_async = AsyncMock(
            return_value=[
                make_entity_index_entry("light.keller_main", "Deckenlicht", area="Keller"),
                make_entity_index_entry("light.keller_side", "Wandlicht", area="Keller"),
            ]
        )

        action = {"action": "turn_on", "entity": "Keller", "parameters": {}}
        result = await execute_action(action, ha_client, index, matcher, agent_id="light-agent")

        assert result["success"] is False
        assert "Multiple entities match 'Keller'" in result["speech"]
        ha_client.call_service.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_area_fallback_uses_refreshed_index(self, ha_client):
        matcher = MagicMock(spec=EntityMatcher)
        matcher.match = AsyncMock(return_value=[])
        matcher.filter_visible_results = AsyncMock(side_effect=lambda agent_id, results: results)
        index = MagicMock(spec=EntityIndex)
        index.get_by_id.return_value = None
        index.list_entries_async = AsyncMock(
            side_effect=[
                [],
                [make_entity_index_entry("light.keller", "Deckenlicht", area="Keller")],
            ]
        )

        action = {"action": "turn_on", "entity": "Keller", "parameters": {}}
        first_result = await execute_action(action, ha_client, index, matcher, agent_id="light-agent")
        second_result = await execute_action(action, ha_client, index, matcher, agent_id="light-agent")

        assert first_result["success"] is False
        assert second_result["success"] is True
        assert second_result["entity_id"] == "light.keller"

    @pytest.mark.asyncio
    async def test_query_light_state_uses_deterministic_resolution(self, ha_client):
        matcher = MagicMock(spec=EntityMatcher)
        matcher.match = AsyncMock(return_value=[])
        matcher.filter_visible_results = AsyncMock(side_effect=lambda agent_id, results: results)
        index = MagicMock(spec=EntityIndex)
        index.get_by_id.return_value = None
        index.list_entries_async = AsyncMock(
            return_value=[make_entity_index_entry("light.keller", "Keller", area="Basement")]
        )

        action = {"action": "query_light_state", "entity": "Keller"}
        result = await execute_action(action, ha_client, index, matcher, agent_id="light-agent")

        assert result["success"] is True
        assert result["entity_id"] == "light.keller"
        matcher.match.assert_not_awaited()


# ---------------------------------------------------------------------------
# Climate executor domain validation tests
# ---------------------------------------------------------------------------

from app.agents.climate_executor import execute_climate_action  # noqa: E402


class TestClimateExecutorDomainValidation:
    """Tests for climate executor domain validation."""

    @pytest.fixture()
    def ha_client(self):
        client = AsyncMock()
        client.call_service = AsyncMock(return_value={})
        client.get_state = AsyncMock(return_value={
            "state": "heat",
            "attributes": {"friendly_name": "Living Room Climate", "current_temperature": 21.5}
        })
        return client

    @pytest.mark.asyncio
    async def test_rejects_media_player_entity(self, ha_client):
        """Climate executor should reject media_player entities."""
        matcher = AsyncMock()
        match_result = MagicMock()
        match_result.entity_id = "media_player.wohnzimmer_tv"
        match_result.friendly_name = "TV Wohnzimmer-TV"
        matcher.match = AsyncMock(return_value=[match_result])
        index = MagicMock()

        action = {"action": "query_climate_state", "entity": "Wohnzimmer", "parameters": {}}
        result = await execute_climate_action(action, ha_client, index, matcher, agent_id="climate-agent")

        assert result["success"] is False
        assert "Could not find" in result["speech"]

    @pytest.mark.asyncio
    async def test_accepts_climate_entity(self, ha_client):
        """Climate executor should accept climate domain entities."""
        matcher = AsyncMock()
        match_result = MagicMock()
        match_result.entity_id = "climate.living_room"
        match_result.friendly_name = "Living Room Climate"
        matcher.match = AsyncMock(return_value=[match_result])
        index = MagicMock()

        action = {"action": "query_climate_state", "entity": "living room", "parameters": {}}
        result = await execute_climate_action(action, ha_client, index, matcher, agent_id="climate-agent")

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_accepts_sensor_entity(self, ha_client):
        """Climate executor should accept sensor domain entities."""
        matcher = AsyncMock()
        match_result = MagicMock()
        match_result.entity_id = "sensor.living_room_temperature"
        match_result.friendly_name = "Living Room Temperature"
        matcher.match = AsyncMock(return_value=[match_result])
        index = MagicMock()

        ha_client.get_state = AsyncMock(return_value={
            "state": "21.5",
            "attributes": {"friendly_name": "Living Room Temperature", "unit_of_measurement": "C"}
        })
        action = {"action": "query_climate_state", "entity": "living room temp", "parameters": {}}
        result = await execute_climate_action(action, ha_client, index, matcher, agent_id="climate-agent")

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_no_fallback_to_unfiltered_index(self, ha_client):
        """When matcher returns empty, should NOT fall back to entity_index.search()."""
        matcher = AsyncMock()
        matcher.match = AsyncMock(return_value=[])
        index = MagicMock()

        action = {"action": "query_climate_state", "entity": "Wohnzimmer", "parameters": {}}
        result = await execute_climate_action(action, ha_client, index, matcher, agent_id="climate-agent")

        assert result["success"] is False
        assert "Could not find" in result["speech"]
        index.search.assert_not_called()


# ---------------------------------------------------------------------------
# Media executor domain validation tests
# ---------------------------------------------------------------------------

from app.agents.media_executor import execute_media_action  # noqa: E402


class TestMediaExecutorDomainValidation:
    """Tests for media executor domain validation."""

    @pytest.fixture()
    def ha_client(self):
        client = AsyncMock()
        client.call_service = AsyncMock(return_value={})
        client.get_state = AsyncMock(return_value={"state": "playing", "attributes": {}})
        return client

    @pytest.mark.asyncio
    async def test_rejects_light_entity(self, ha_client):
        """Media executor should reject light entities."""
        matcher = AsyncMock()
        match_result = MagicMock()
        match_result.entity_id = "light.living_room"
        match_result.friendly_name = "Living Room Light"
        matcher.match = AsyncMock(return_value=[match_result])
        index = MagicMock()

        action = {"action": "play", "entity": "living room", "parameters": {}}
        result = await execute_media_action(action, ha_client, index, matcher, agent_id="media-agent")

        assert result["success"] is False
        assert "Could not find" in result["speech"]


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


# ---------------------------------------------------------------------------
# Entity match span verification tests
# ---------------------------------------------------------------------------

from app.analytics.tracer import SpanCollector  # noqa: E402


class TestEntityMatchSpan:
    """Tests that entity_match spans are recorded with correct metadata."""

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
        match_result.score = 0.95
        match_result.signal_scores = {"levenshtein": 0.9, "embedding": 0.8}
        matcher.match = AsyncMock(return_value=[match_result])
        return matcher

    @pytest.fixture()
    def entity_index(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_entity_match_span_recorded(self, ha_client, entity_matcher, entity_index):
        """Verify entity_match span is recorded with correct metadata when SpanCollector is passed."""
        span_collector = SpanCollector(trace_id="test-trace-123")
        action = {"action": "turn_on", "entity": "kitchen light", "parameters": {}}

        result = await execute_action(
            action, ha_client, entity_index, entity_matcher,
            agent_id="light-agent", span_collector=span_collector,
        )

        assert result["success"] is True

        # Find the entity_match span
        em_spans = [s for s in span_collector._spans if s["span_name"] == "entity_match"]
        assert len(em_spans) == 1
        span = em_spans[0]

        assert span["agent_id"] == "light-agent"
        assert span["status"] == "ok"
        assert span["metadata"]["query"] == "kitchen light"
        assert span["metadata"]["match_count"] == 1
        assert span["metadata"]["top_entity_id"] == "light.kitchen_ceiling"
        assert span["metadata"]["top_friendly_name"] == "Kitchen Ceiling"
        assert span["metadata"]["top_score"] == 0.95
        assert span["metadata"]["signal_scores"] == {"levenshtein": 0.9, "embedding": 0.8}

    @pytest.mark.asyncio
    async def test_entity_match_span_no_match(self, ha_client, entity_index):
        """Verify entity_match span records zero matches correctly."""
        matcher = AsyncMock()
        matcher.match = AsyncMock(return_value=[])
        span_collector = SpanCollector(trace_id="test-trace-456")
        action = {"action": "turn_on", "entity": "nonexistent", "parameters": {}}

        result = await execute_action(
            action, ha_client, entity_index, matcher,
            agent_id="light-agent", span_collector=span_collector,
        )

        assert result["success"] is False
        em_spans = [s for s in span_collector._spans if s["span_name"] == "entity_match"]
        assert len(em_spans) == 1
        assert em_spans[0]["metadata"]["match_count"] == 0

    @pytest.mark.asyncio
    async def test_no_span_without_collector(self, ha_client, entity_matcher, entity_index):
        """Verify execute_action works without span_collector (backward compatible)."""
        action = {"action": "turn_on", "entity": "kitchen light", "parameters": {}}
        result = await execute_action(action, ha_client, entity_index, entity_matcher)

        assert result["success"] is True
        assert result["entity_id"] == "light.kitchen_ceiling"

    @pytest.mark.asyncio
    async def test_entity_match_span_records_exact_resolution_path(self, ha_client):
        matcher = MagicMock(spec=EntityMatcher)
        matcher.match = AsyncMock(return_value=[])
        matcher.filter_visible_results = AsyncMock(side_effect=lambda agent_id, results: results)
        index = MagicMock(spec=EntityIndex)
        index.get_by_id.return_value = None
        index.list_entries_async = AsyncMock(
            return_value=[make_entity_index_entry("light.keller", "Keller", area="Basement")]
        )
        span_collector = SpanCollector(trace_id="test-trace-deterministic")

        result = await execute_action(
            {"action": "turn_on", "entity": "Keller", "parameters": {}},
            ha_client,
            index,
            matcher,
            agent_id="light-agent",
            span_collector=span_collector,
        )

        assert result["success"] is True
        em_spans = [s for s in span_collector._spans if s["span_name"] == "entity_match"]
        assert len(em_spans) == 1
        assert em_spans[0]["metadata"]["resolution_path"] == "exact_friendly_name"
        assert em_spans[0]["metadata"]["top_entity_id"] == "light.keller"


# ---------------------------------------------------------------------------
# Read-only actions return cacheable=False
# ---------------------------------------------------------------------------

class TestReadActionCacheable:
    """Tests that read-only executor actions return cacheable=False."""

    @pytest.fixture()
    def ha_client(self):
        client = AsyncMock()
        client.get_state = AsyncMock(return_value={
            "state": "on", "attributes": {"friendly_name": "Kitchen Light", "brightness": 200},
        })
        client.get_states = AsyncMock(return_value=[
            {"entity_id": "light.kitchen", "state": "on",
             "attributes": {"friendly_name": "Kitchen Light"}},
        ])
        return client

    @pytest.fixture()
    def entity_matcher(self):
        matcher = AsyncMock()
        match_result = MagicMock()
        match_result.entity_id = "light.kitchen"
        match_result.friendly_name = "Kitchen Light"
        match_result.score = 0.95
        match_result.signal_scores = {}
        matcher.match = AsyncMock(return_value=[match_result])
        return matcher

    @pytest.fixture()
    def entity_index(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_query_light_state_cacheable_false(self, ha_client, entity_matcher, entity_index):
        action = {"action": "query_light_state", "entity": "kitchen light"}
        result = await execute_action(action, ha_client, entity_index, entity_matcher)
        assert result["cacheable"] is False

    @pytest.mark.asyncio
    async def test_list_lights_cacheable_false(self, ha_client, entity_matcher, entity_index):
        action = {"action": "list_lights", "entity": ""}
        result = await execute_action(action, ha_client, entity_index, entity_matcher)
        assert result["cacheable"] is False


# ---------------------------------------------------------------------------
# Climate executor weather action tests
# ---------------------------------------------------------------------------


class TestClimateExecutorWeatherActions:
    """Tests for climate executor weather query actions."""

    @pytest.fixture()
    def ha_client(self):
        client = AsyncMock()
        client.call_service = AsyncMock(return_value={})
        client.get_state = AsyncMock(return_value={
            "state": "sunny",
            "attributes": {
                "friendly_name": "Home",
                "temperature": 22.5,
                "temperature_unit": "C",
                "humidity": 55,
                "wind_speed": 12.3,
                "wind_speed_unit": "km/h",
                "pressure": 1013,
                "pressure_unit": "hPa",
            },
        })
        client.get_states = AsyncMock(return_value=[
            {"entity_id": "weather.home", "state": "sunny",
             "attributes": {"friendly_name": "Home"}},
        ])
        return client

    @pytest.mark.asyncio
    async def test_weather_domain_validation_accepts(self):
        from app.agents.climate_executor import _validate_domain
        assert _validate_domain("weather.home") is True

    @pytest.mark.asyncio
    async def test_query_weather_with_entity_match(self, ha_client):
        matcher = AsyncMock()
        match_result = MagicMock()
        match_result.entity_id = "weather.home"
        match_result.friendly_name = "Home"
        match_result.score = 0.9
        match_result.signal_scores = {}
        matcher.match = AsyncMock(return_value=[match_result])

        result = await execute_climate_action(
            {"action": "query_weather", "entity": "home weather"},
            ha_client, None, matcher, agent_id="climate-agent",
        )
        assert result["success"] is True
        assert "sunny" in result["speech"]
        assert "22.5" in result["speech"]
        assert "55%" in result["speech"]

    @pytest.mark.asyncio
    async def test_query_weather_auto_discover(self, ha_client):
        matcher = AsyncMock()
        matcher.match = AsyncMock(return_value=[])

        result = await execute_climate_action(
            {"action": "query_weather", "entity": ""},
            ha_client, None, matcher, agent_id="climate-agent",
        )
        assert result["success"] is True
        assert "sunny" in result["speech"]

    @pytest.mark.asyncio
    async def test_query_weather_no_entity_found(self, ha_client):
        ha_client.get_states = AsyncMock(return_value=[])
        matcher = AsyncMock()
        matcher.match = AsyncMock(return_value=[])

        result = await execute_climate_action(
            {"action": "query_weather", "entity": ""},
            ha_client, None, matcher, agent_id="climate-agent",
        )
        assert result["success"] is False
        assert "No weather entities" in result["speech"]

    @pytest.mark.asyncio
    async def test_query_weather_forecast_service_call(self, ha_client):
        ha_client.call_service = AsyncMock(return_value={
            "weather.home": {
                "forecast": [
                    {"datetime": "2025-01-16T00:00:00", "condition": "cloudy",
                     "temperature": 18, "templow": 8, "precipitation": 2.5, "wind_speed": 15},
                    {"datetime": "2025-01-17T00:00:00", "condition": "rainy",
                     "temperature": 15, "templow": 6, "precipitation": 10, "wind_speed": 20},
                ],
            },
        })
        matcher = AsyncMock()
        match_result = MagicMock()
        match_result.entity_id = "weather.home"
        match_result.friendly_name = "Home"
        match_result.score = 0.9
        match_result.signal_scores = {}
        matcher.match = AsyncMock(return_value=[match_result])

        result = await execute_climate_action(
            {"action": "query_weather_forecast", "entity": "home"},
            ha_client, None, matcher, agent_id="climate-agent",
        )
        assert result["success"] is True
        assert "cloudy" in result["speech"]
        assert "rainy" in result["speech"]

    @pytest.mark.asyncio
    async def test_query_weather_forecast_fallback_to_state(self, ha_client):
        ha_client.call_service = AsyncMock(side_effect=Exception("Service not found"))
        ha_client.get_state = AsyncMock(return_value={
            "state": "sunny",
            "attributes": {
                "friendly_name": "Home",
                "forecast": [
                    {"datetime": "2025-01-16T00:00:00", "condition": "partly_cloudy",
                     "temperature": 20, "templow": 10},
                ],
            },
        })
        matcher = AsyncMock()
        match_result = MagicMock()
        match_result.entity_id = "weather.home"
        match_result.friendly_name = "Home"
        match_result.score = 0.9
        match_result.signal_scores = {}
        matcher.match = AsyncMock(return_value=[match_result])

        result = await execute_climate_action(
            {"action": "query_weather_forecast", "entity": "home"},
            ha_client, None, matcher, agent_id="climate-agent",
        )
        assert result["success"] is True
        assert "partly_cloudy" in result["speech"]

    @pytest.mark.asyncio
    async def test_query_weather_forecast_no_data(self, ha_client):
        ha_client.call_service = AsyncMock(side_effect=Exception("Service not found"))
        ha_client.get_state = AsyncMock(return_value={
            "state": "sunny",
            "attributes": {"friendly_name": "Home"},
        })
        matcher = AsyncMock()
        match_result = MagicMock()
        match_result.entity_id = "weather.home"
        match_result.friendly_name = "Home"
        match_result.score = 0.9
        match_result.signal_scores = {}
        matcher.match = AsyncMock(return_value=[match_result])

        result = await execute_climate_action(
            {"action": "query_weather_forecast", "entity": "home"},
            ha_client, None, matcher, agent_id="climate-agent",
        )
        assert result["success"] is False
        assert "not available" in result["speech"]
