"""HA WebSocket client for real-time state updates."""

import asyncio
import json
import logging
import random
import time
from typing import Any, Callable, Coroutine, Optional

import aiohttp

from app.ha_client.auth import get_ha_token
from app.db.repository import SettingsRepository

logger = logging.getLogger(__name__)

BASE_DELAY = 1.0
MAX_DELAY = 60.0
MAX_JITTER = 1.0
HEARTBEAT_INTERVAL = 15.0
IDLE_TIMEOUT = 2 * HEARTBEAT_INTERVAL


class HAWebSocketClient:

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._running: bool = False
        self._use_rest_fallback: bool = False
        self._message_id: int = 0
        self._listeners: dict[str, list[Callable]] = {}
        self._logger = logging.getLogger("ha_client.websocket")
        self._ws_lock: asyncio.Lock = asyncio.Lock()
        self._ws_last_active: float = time.monotonic()
        # FLOW-VERIFY-1: per-entity state waiters, resolved from the single
        # WebSocket receive task. Key = entity_id, value = list of
        # (future, expected_state_or_None) tuples. Ordered FIFO so multiple
        # concurrent actions on the same entity resolve in the order they
        # registered.
        self._state_waiters: dict[
            str, list[tuple[asyncio.Future[str], str | None]]
        ] = {}

    def is_connected(self) -> bool:
        """Return True if the WebSocket connection is active and running."""
        return self._running and self._ws is not None and not self._ws.closed

    def _next_id(self) -> int:
        self._message_id += 1
        return self._message_id

    async def connect(self) -> bool:
        ha_url = await SettingsRepository.get_value("ha_url")
        token = await get_ha_token()
        if not ha_url or not token:
            self._logger.warning("HA URL or token not configured, skipping connection")
            return False

        ha_url = ha_url.rstrip("/")
        if ha_url.startswith("https://"):
            ws_url = ha_url.replace("https://", "wss://") + "/api/websocket"
        else:
            ws_url = ha_url.replace("http://", "ws://") + "/api/websocket"

        try:
            async with self._ws_lock:
                self._session = aiohttp.ClientSession()
                self._ws = await self._session.ws_connect(
                    ws_url, heartbeat=HEARTBEAT_INTERVAL
                )
            self._ws_last_active = time.monotonic()

            msg = await self._ws.receive_json()
            if msg.get("type") != "auth_required":
                self._logger.error("Unexpected initial message from HA WebSocket")
                await self._close_session()
                return False

            await self._ws.send_json({"type": "auth", "access_token": token})
            auth_response = await self._ws.receive_json()

            if auth_response.get("type") != "auth_ok":
                self._logger.error("HA WebSocket auth failed")
                await self._close_session()
                return False

            self._running = True
            self._logger.info("Connected to HA WebSocket")

            # Auto-subscribe to all registered event types
            for event_type in self._listeners:
                await self.subscribe_events(event_type)

            return True
        except Exception:
            self._logger.error("Failed to connect to HA WebSocket", exc_info=True)
            await self._close_session()
            return False

    async def _close_session(self) -> None:
        async with self._ws_lock:
            if self._ws and not self._ws.closed:
                await self._ws.close()
            self._ws = None
            if self._session and not self._session.closed:
                await self._session.close()
            self._session = None

    async def disconnect(self) -> None:
        self._running = False
        await self._close_session()
        self._logger.info("Disconnected from HA WebSocket")

    async def drop_connection(self) -> None:
        """Close the live socket without stopping ``run()``.

        Used after admin updates ``ha_url`` / token so the receive loop
        exits and ``run()`` reconnects with fresh settings from the DB.
        """
        if not self._running:
            return
        await self._close_session()
        self._logger.info("HA WebSocket connection dropped for reconnect")

    async def run(self) -> None:
        self._running = True
        while self._running:
            connected = await self.connect()
            if not connected:
                if not self._running:
                    return
                await self._reconnect_loop()
                continue
            try:
                await self._receive_loop()
            except Exception:
                self._logger.error("WebSocket receive loop error", exc_info=True)
            if self._running:
                await self._close_session()
                await self._reconnect_loop()

    async def _reconnect_loop(self) -> None:
        attempt = 0
        max_delay = MAX_DELAY
        try:
            val = await SettingsRepository.get_value("communication.ws_reconnect_interval")
            if val is not None:
                max_delay = float(val)
        except (ValueError, TypeError, Exception):
            pass
        while self._running:
            delay = min(BASE_DELAY * (2 ** attempt), max_delay) + random.uniform(0, MAX_JITTER)
            self._logger.info("Reconnecting in %.1fs (attempt %d)", delay, attempt + 1)
            await asyncio.sleep(delay)
            try:
                if await self.connect():
                    self._use_rest_fallback = False
                    return
            except Exception:
                self._logger.error("Reconnect attempt failed", exc_info=True)
            attempt += 1
            if attempt >= 5 and not self._use_rest_fallback:
                self._use_rest_fallback = True
                self._logger.warning(
                    "WebSocket reconnect failed after %d attempts, enabling REST fallback",
                    attempt,
                )

    async def _receive_loop(self) -> None:
        while self._running and self._ws and not self._ws.closed:
            try:
                msg = await asyncio.wait_for(
                    self._ws.receive(), timeout=IDLE_TIMEOUT
                )
            except asyncio.TimeoutError:
                self._logger.warning(
                    "HA WebSocket idle for >%.0fs, forcing reconnect",
                    IDLE_TIMEOUT,
                )
                break
            self._ws_last_active = time.monotonic()
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    event_type = data.get("type", "")
                    if event_type == "event":
                        event = data.get("event", {})
                        et = event.get("event_type", "")
                        if et == "state_changed":
                            self._dispatch_state_waiters(event)
                        for callback in self._listeners.get(et, []):
                            try:
                                result = callback(event)
                                if asyncio.iscoroutine(result):
                                    await result
                            except Exception:
                                self._logger.error("Event callback error", exc_info=True)
                except json.JSONDecodeError:
                    self._logger.warning("Received non-JSON WebSocket message")
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING,
                              aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break

    async def subscribe_events(self, event_type: str | None = None) -> int:
        msg_id = self._next_id()
        payload: dict[str, Any] = {"id": msg_id, "type": "subscribe_events"}
        if event_type:
            payload["event_type"] = event_type
        if self._ws and not self._ws.closed:
            await self._ws.send_json(payload)
        return msg_id

    def on_event(self, event_type: str, callback: Callable) -> None:
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)

    # -----------------------------------------------------------------
    # FLOW-VERIFY-1: state-change waiters for post-action verification
    # -----------------------------------------------------------------
    def register_state_waiter(
        self,
        entity_id: str,
        *,
        expected: str | None = None,
    ) -> asyncio.Future[str]:
        """Register a waiter that resolves on the next matching state_changed.

        Call this BEFORE issuing the action that is expected to cause the
        state change, to avoid a race where the event arrives before the
        waiter is registered.

        Args:
            entity_id: Entity to watch (e.g. ``"light.keller"``).
            expected: If given, only events whose ``new_state.state`` equals
                this string resolve the waiter. ``None`` resolves on the
                first ``state_changed`` event for the entity.

        Returns:
            An ``asyncio.Future`` that resolves with the observed
            ``new_state.state`` string. The caller is responsible for
            awaiting it with a timeout and for cancelling on its own error
            paths if desired.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._state_waiters.setdefault(entity_id, []).append((future, expected))
        return future

    def _dispatch_state_waiters(self, event: dict) -> None:
        """Resolve pending waiters for a ``state_changed`` event."""
        data = event.get("data") or {}
        entity_id = data.get("entity_id") or ""
        new_state = data.get("new_state") or {}
        if not entity_id or not isinstance(new_state, dict):
            return
        state_value = new_state.get("state")
        if not isinstance(state_value, str):
            return
        waiters = self._state_waiters.get(entity_id)
        if not waiters:
            return
        kept: list[tuple[asyncio.Future[str], str | None]] = []
        for future, expected in waiters:
            if future.done():
                continue
            if expected is not None and state_value != expected:
                kept.append((future, expected))
                continue
            try:
                future.set_result(state_value)
            except asyncio.InvalidStateError:
                pass
        if kept:
            self._state_waiters[entity_id] = kept
        else:
            self._state_waiters.pop(entity_id, None)

    def cancel_state_waiter(self, entity_id: str, future: asyncio.Future[str]) -> None:
        """Remove a pending waiter (e.g. on timeout) without resolving it."""
        waiters = self._state_waiters.get(entity_id)
        if not waiters:
            return
        remaining = [(f, exp) for f, exp in waiters if f is not future]
        if remaining:
            self._state_waiters[entity_id] = remaining
        else:
            self._state_waiters.pop(entity_id, None)
        if not future.done():
            future.cancel()
