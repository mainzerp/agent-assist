"""Tests for HAWebSocketClient state-change waiters (FLOW-VERIFY-1)."""

from __future__ import annotations

import asyncio

import pytest

from app.ha_client.websocket import HAWebSocketClient


def _state_event(entity_id: str, state: str) -> dict:
    """Build a minimal HA ``state_changed`` event payload."""
    return {
        "event_type": "state_changed",
        "data": {
            "entity_id": entity_id,
            "new_state": {"entity_id": entity_id, "state": state, "attributes": {}},
        },
    }


class TestStateWaiters:
    """Unit tests for register_state_waiter / _dispatch_state_waiters."""

    @pytest.mark.asyncio
    async def test_resolves_on_matching_event(self):
        client = HAWebSocketClient()
        fut = client.register_state_waiter("light.keller", expected="off")

        client._dispatch_state_waiters(_state_event("light.keller", "off"))

        state = await asyncio.wait_for(fut, timeout=0.1)
        assert state == "off"
        assert "light.keller" not in client._state_waiters

    @pytest.mark.asyncio
    async def test_ignores_mismatched_expected(self):
        client = HAWebSocketClient()
        fut = client.register_state_waiter("light.keller", expected="off")

        client._dispatch_state_waiters(_state_event("light.keller", "on"))

        assert not fut.done()
        # a later matching event still resolves it
        client._dispatch_state_waiters(_state_event("light.keller", "off"))
        state = await asyncio.wait_for(fut, timeout=0.1)
        assert state == "off"

    @pytest.mark.asyncio
    async def test_any_state_when_expected_is_none(self):
        client = HAWebSocketClient()
        fut = client.register_state_waiter("light.keller", expected=None)

        client._dispatch_state_waiters(_state_event("light.keller", "on"))

        state = await asyncio.wait_for(fut, timeout=0.1)
        assert state == "on"

    @pytest.mark.asyncio
    async def test_ignores_other_entities(self):
        client = HAWebSocketClient()
        fut = client.register_state_waiter("light.keller", expected="off")

        client._dispatch_state_waiters(_state_event("switch.keller", "off"))

        assert not fut.done()
        assert "light.keller" in client._state_waiters

    @pytest.mark.asyncio
    async def test_cancel_state_waiter_removes_pending_future(self):
        client = HAWebSocketClient()
        fut = client.register_state_waiter("light.keller", expected="off")

        client.cancel_state_waiter("light.keller", fut)

        assert "light.keller" not in client._state_waiters
        assert fut.cancelled()

    @pytest.mark.asyncio
    async def test_multiple_waiters_resolve_fifo(self):
        client = HAWebSocketClient()
        first = client.register_state_waiter("light.keller", expected="off")
        second = client.register_state_waiter("light.keller", expected="off")

        client._dispatch_state_waiters(_state_event("light.keller", "off"))

        assert first.done() and second.done()
        assert (await first) == "off"
        assert (await second) == "off"

    @pytest.mark.asyncio
    async def test_malformed_event_does_not_crash(self):
        client = HAWebSocketClient()
        client.register_state_waiter("light.keller", expected="off")

        # Missing new_state.
        client._dispatch_state_waiters({
            "event_type": "state_changed",
            "data": {"entity_id": "light.keller"},
        })
        # Entirely empty.
        client._dispatch_state_waiters({})
        # The waiter is still pending, not crashed.
        assert "light.keller" in client._state_waiters
