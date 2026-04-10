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
