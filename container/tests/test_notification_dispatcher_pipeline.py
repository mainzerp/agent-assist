"""Tests for FLOW-HIGH-5 (real device_id for assist_pipeline.run) and
FLOW-HIGH-6 (assist_satellite resolved via EntityIndex)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents import notification_dispatcher as nd


class _FakeResp:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHaClient:
    """Minimal HA client stub exposing render_template + call_service."""

    def __init__(self, device_id_text: str = "abc123") -> None:
        self._device_id_text = device_id_text
        self.calls: list[tuple] = []
        self.template_calls: list[str] = []

    async def render_template(self, template: str) -> str | None:
        self.template_calls.append(template)
        if self._device_id_text == "":
            return None
        return self._device_id_text

    async def call_service(self, domain, service, entity_id, data):
        self.calls.append((domain, service, entity_id, dict(data or {})))
        return {}

    async def get_states(self):
        return []


@pytest.mark.asyncio
async def test_pipeline_run_uses_resolved_device_id() -> None:
    client = _FakeHaClient(device_id_text="abc123")
    profile = {"voice_followup_enabled": True, "tts_to_listen_delay": 0}

    with patch.object(nd, "asyncio") as fake_asyncio:
        fake_asyncio.sleep = AsyncMock()
        await nd._trigger_conversation_continuation(
            client,
            "media_player.kitchen",
            area=None,
            profile=profile,
            entity_index=None,
        )

    assert len(client.calls) == 1
    domain, service, entity_id, data = client.calls[0]
    assert (domain, service, entity_id) == ("assist_pipeline", "run", None)
    assert data["device_id"] == "abc123"
    assert data["start_stage"] == "stt"
    assert data["end_stage"] == "tts"


@pytest.mark.asyncio
async def test_pipeline_run_omits_device_id_when_resolution_fails() -> None:
    client = _FakeHaClient(device_id_text="")
    profile = {"voice_followup_enabled": True, "tts_to_listen_delay": 0}

    with patch.object(nd, "asyncio") as fake_asyncio:
        fake_asyncio.sleep = AsyncMock()
        await nd._trigger_conversation_continuation(
            client,
            "media_player.kitchen",
            area=None,
            profile=profile,
            entity_index=None,
        )

    assert len(client.calls) == 1
    _, _, _, data = client.calls[0]
    assert "device_id" not in data


@pytest.mark.asyncio
async def test_pipeline_run_omits_device_id_when_template_returns_none_string() -> None:
    client = _FakeHaClient(device_id_text="None")
    profile = {"voice_followup_enabled": True, "tts_to_listen_delay": 0}

    with patch.object(nd, "asyncio") as fake_asyncio:
        fake_asyncio.sleep = AsyncMock()
        await nd._trigger_conversation_continuation(
            client,
            "media_player.kitchen",
            area=None,
            profile=profile,
            entity_index=None,
        )

    _, _, _, data = client.calls[0]
    assert "device_id" not in data


@pytest.mark.asyncio
async def test_resolve_satellite_uses_entity_index_area() -> None:
    entry_kitchen = SimpleNamespace(
        entity_id="assist_satellite.kitchen_pi",
        area="kitchen",
        domain="assist_satellite",
    )
    entry_bedroom = SimpleNamespace(
        entity_id="assist_satellite.bedroom_pi",
        area="bedroom",
        domain="assist_satellite",
    )

    index = MagicMock()
    index.list_entries_async = AsyncMock(return_value=[entry_bedroom, entry_kitchen])

    client = _FakeHaClient()
    got = await nd._resolve_satellite_device(client, "kitchen", entity_index=index)
    assert got == "assist_satellite.kitchen_pi"
    index.list_entries_async.assert_awaited_once()
    kwargs = index.list_entries_async.await_args.kwargs
    assert kwargs.get("domains") == {"assist_satellite"}


@pytest.mark.asyncio
async def test_resolve_satellite_returns_none_when_no_match() -> None:
    entry_bedroom = SimpleNamespace(
        entity_id="assist_satellite.bedroom_pi",
        area="bedroom",
        domain="assist_satellite",
    )
    index = MagicMock()
    index.list_entries_async = AsyncMock(return_value=[entry_bedroom])

    client = _FakeHaClient()
    got = await nd._resolve_satellite_device(client, "kitchen", entity_index=index)
    assert got is None


@pytest.mark.asyncio
async def test_resolve_satellite_none_when_area_missing() -> None:
    client = _FakeHaClient()
    got = await nd._resolve_satellite_device(client, None, entity_index=None)
    assert got is None


@pytest.mark.asyncio
async def test_pipeline_prefers_satellite_from_entity_index() -> None:
    entry_kitchen = SimpleNamespace(
        entity_id="assist_satellite.kitchen_pi",
        area="kitchen",
        domain="assist_satellite",
    )
    index = MagicMock()
    index.list_entries_async = AsyncMock(return_value=[entry_kitchen])

    client = _FakeHaClient(device_id_text="dev-kitchen-001")
    profile = {"voice_followup_enabled": True, "tts_to_listen_delay": 0}

    with patch.object(nd, "asyncio") as fake_asyncio:
        fake_asyncio.sleep = AsyncMock()
        await nd._trigger_conversation_continuation(
            client,
            "media_player.kitchen",
            area="kitchen",
            profile=profile,
            entity_index=index,
        )

    assert client.template_calls == [
        "{{ device_id('assist_satellite.kitchen_pi') }}",
    ]
    _, _, _, data = client.calls[0]
    assert data["device_id"] == "dev-kitchen-001"
