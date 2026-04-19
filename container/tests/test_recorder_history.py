"""Tests for HA Recorder history (REST), shared history_query, and domain executors."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import httpx
import pytest
import respx

from app.agents.climate_executor import execute_climate_action
from app.ha_client.history_query import execute_recorder_history_query
from app.ha_client.history_util import parse_history_window, summarize_history_for_speech
from app.ha_client.rest import HARestClient
from app.models.agent import TaskContext


class TestParseHistoryWindow:
    def test_last_24_hours_uses_timezone(self):
        ctx = TaskContext(timezone="Europe/Berlin")
        start, end = parse_history_window({}, ctx)
        assert start is not None and end is not None
        assert end > start
        assert (end - start).total_seconds() <= 25 * 3600

    def test_yesterday_midnight_bounds(self):
        ctx = TaskContext(timezone="UTC")
        start, end = parse_history_window({"period": "yesterday"}, ctx)
        assert start is not None and end is not None
        assert (end - start).total_seconds() == 86400

    def test_explicit_iso_span(self):
        ctx = TaskContext(timezone="UTC")
        start, end = parse_history_window(
            {"start": "2024-06-01T10:00:00+00:00", "end": "2024-06-01T18:00:00+00:00"},
            ctx,
        )
        assert start == datetime(2024, 6, 1, 10, 0, tzinfo=UTC)
        assert end == datetime(2024, 6, 1, 18, 0, tzinfo=UTC)

    def test_rejects_over_7_days(self):
        ctx = TaskContext(timezone="UTC")
        start, end = parse_history_window(
            {"start": "2024-06-01T10:00:00+00:00", "end": "2024-06-15T18:00:00+00:00"},
            ctx,
        )
        assert start is None and end is None


class TestSummarizeHistory:
    def test_numeric_min_max(self):
        rows = [
            {
                "state": "20.1",
                "last_changed": "2024-06-01T08:00:00+00:00",
                "attributes": {"unit_of_measurement": "°C"},
            },
            {
                "state": "21.5",
                "last_changed": "2024-06-01T12:00:00+00:00",
                "attributes": {"unit_of_measurement": "°C"},
            },
        ]
        speech = summarize_history_for_speech(
            "sensor.lr_temp",
            "Living temp",
            [rows],
            display_tz=ZoneInfo("UTC"),
        )
        assert "20.10" in speech or "20.1" in speech
        assert "21.50" in speech or "21.5" in speech
        assert "Living temp" in speech


class TestHARestClientHistory:
    pytestmark = pytest.mark.asyncio

    @respx.mock
    async def test_get_history_period_encodes_path(self):
        start = datetime(2024, 6, 1, 10, 0, 0, tzinfo=UTC)
        end = datetime(2024, 6, 2, 10, 0, 0, tzinfo=UTC)
        payload = [[{"entity_id": "sensor.x", "state": "1"}]]
        route = respx.get(url__regex=r"http://ha\.test/api/history/period/.+").mock(
            return_value=httpx.Response(200, json=payload)
        )

        client = HARestClient()
        client._base_url = "http://ha.test"
        client._client = httpx.AsyncClient(base_url="http://ha.test", headers={})

        result = await client.get_history_period(
            start,
            entity_id="sensor.x",
            end_time_utc=end,
        )
        assert result == payload
        assert route.called
        await client.close()


class TestQueryEntityHistoryExecutor:
    pytestmark = pytest.mark.asyncio

    async def test_history_action_calls_ha(self):
        matcher = AsyncMock()
        match_obj = MagicMock(entity_id="sensor.living_temperature", friendly_name="Living temperature")
        matcher.match = AsyncMock(return_value=[match_obj])

        ha = AsyncMock()
        ha.expect_state = None
        ha.get_history_period = AsyncMock(
            return_value=[
                [
                    {
                        "state": "19",
                        "last_changed": "2024-06-01T08:00:00+00:00",
                        "attributes": {"unit_of_measurement": "°C"},
                    },
                    {
                        "state": "22",
                        "last_changed": "2024-06-01T20:00:00+00:00",
                        "attributes": {"unit_of_measurement": "°C"},
                    },
                ]
            ]
        )

        ctx = TaskContext(timezone="UTC")
        result = await execute_climate_action(
            {
                "action": "query_entity_history",
                "entity": "living temperature",
                "parameters": {"period": "yesterday"},
            },
            ha,
            MagicMock(),
            matcher,
            agent_id="climate-agent",
            task_context=ctx,
        )
        assert result["success"] is True
        assert result["entity_id"] == "sensor.living_temperature"
        ha.get_history_period.assert_awaited_once()
        assert "19" in result["speech"] and "22" in result["speech"]


class TestExecuteRecorderHistoryQuery:
    pytestmark = pytest.mark.asyncio

    async def test_happy_path(self):
        ha = AsyncMock()
        ha.get_history_period = AsyncMock(
            return_value=[
                [
                    {
                        "state": "on",
                        "last_changed": "2024-06-01T08:00:00+00:00",
                        "attributes": {},
                    },
                ]
            ]
        )
        ctx = TaskContext(timezone="UTC")
        result = await execute_recorder_history_query(
            "light.kitchen",
            "Kitchen",
            {"period": "yesterday"},
            ha,
            allowed_domains=frozenset({"light", "switch", "sensor"}),
            task_context=ctx,
        )
        assert result["success"] is True
        assert result["entity_id"] == "light.kitchen"
        ha.get_history_period.assert_awaited_once()

    async def test_rejects_disallowed_domain(self):
        ha = AsyncMock()
        result = await execute_recorder_history_query(
            "climate.x",
            "HVAC",
            {"period": "yesterday"},
            ha,
            allowed_domains=frozenset({"light"}),
            task_context=TaskContext(timezone="UTC"),
        )
        assert result["success"] is False
        ha.get_history_period.assert_not_called()
