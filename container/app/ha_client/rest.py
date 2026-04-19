"""Async HA REST API client."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx

from app.db.repository import SettingsRepository
from app.ha_client.auth import get_auth_headers

if TYPE_CHECKING:
    from app.ha_client.websocket import HAWebSocketClient

logger = logging.getLogger(__name__)


class HARestClient:
    """Async REST client for the Home Assistant API."""

    def __init__(self) -> None:
        self._base_url: str | None = None
        self._client: httpx.AsyncClient | None = None
        # FLOW-VERIFY-1: optional WebSocket observer for post-action state
        # verification. Wired from main.py once both clients exist.
        self._state_observer: HAWebSocketClient | None = None

    async def initialize(self) -> None:
        """Load HA URL from settings and create httpx client."""
        self._base_url = await SettingsRepository.get_value("ha_url")
        if self._base_url:
            self._base_url = self._base_url.rstrip("/")
        headers = await get_auth_headers()
        self._client = httpx.AsyncClient(
            base_url=self._base_url or "",
            headers=headers or {},
            timeout=httpx.Timeout(30.0),
        )
        logger.info("HARestClient initialized with base_url=%s", self._base_url)

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def reload(self) -> None:
        """Re-read ha_url and auth from settings; rebuild the httpx
        client if the base_url changed.

        FLOW-HIGH-9: ``httpx.AsyncClient.base_url`` is immutable after
        construction, so when the setup wizard (or admin settings UI)
        updates ``ha_url``, simply refreshing headers on the existing
        client leaves it pointing at the old host until container
        restart. Rebuild the client when the URL actually changes;
        otherwise fall back to cheap header-only refresh.
        """
        new_url = await SettingsRepository.get_value("ha_url")
        if new_url:
            new_url = new_url.rstrip("/")
        headers = await get_auth_headers()

        if self._client is None:
            self._base_url = new_url
            self._client = httpx.AsyncClient(
                base_url=new_url or "",
                headers=headers or {},
                timeout=httpx.Timeout(30.0),
            )
            logger.info("HARestClient initialized via reload() with base_url=%s", self._base_url)
            return

        if new_url and new_url != self._base_url:
            logger.info(
                "HARestClient base_url changed (%s -> %s); rebuilding client",
                self._base_url,
                new_url,
            )
            old = self._client
            self._base_url = new_url
            self._client = httpx.AsyncClient(
                base_url=new_url,
                headers=headers or {},
                timeout=httpx.Timeout(30.0),
            )
            try:
                await old.aclose()
            except Exception:
                logger.debug("Failed to close old HA REST client", exc_info=True)
            return

        if headers:
            self._client.headers.update(headers)

    async def _refresh_headers(self) -> None:
        """Refresh auth headers and (if the configured ``ha_url``
        changed) rebuild the underlying httpx client so the new
        base_url takes effect.

        Post-:meth:`close` this is a safe no-op: if ``_client`` is
        ``None`` we do not resurrect it here -- callers that want to
        recreate the client after a clean shutdown must call
        :meth:`initialize` or :meth:`reload` explicitly. This matches
        the contract asserted by COR-5 in ``test_ha_client.py``.
        """
        if self._client is None:
            return
        await self.reload()

    async def test_connection(self) -> bool:
        """Test connectivity to HA by hitting GET /api/.

        Returns True if HA responds with 200, False otherwise.
        """
        try:
            resp = await self._client.get("/api/")
            return resp.status_code == 200
        except httpx.HTTPError:
            logger.warning("HA connection test failed", exc_info=True)
            return False

    async def get_states(self) -> list[dict[str, Any]]:
        """GET /api/states -- returns all entity states."""
        resp = await self._client.get("/api/states")
        resp.raise_for_status()
        return resp.json()

    async def get_services(self) -> dict[str, Any]:
        """GET /api/services -- returns a dict of domain -> service list."""
        resp = await self._client.get("/api/services")
        resp.raise_for_status()
        data = resp.json()
        result: dict[str, Any] = {}
        for entry in data or []:
            result[entry.get("domain", "")] = entry.get("services", {})
        return result

    async def get_state(self, entity_id: str) -> dict[str, Any] | None:
        """GET /api/states/<entity_id> -- returns a single entity state."""
        resp = await self._client.get(f"/api/states/{entity_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    async def call_service(
        self,
        domain: str,
        service: str,
        entity_id: str | None = None,
        service_data: dict[str, Any] | None = None,
        *,
        return_response: bool = False,
    ) -> dict[str, Any]:
        """POST /api/services/<domain>/<service>."""
        payload: dict[str, Any] = {}
        if entity_id:
            payload["entity_id"] = entity_id
        if service_data:
            payload.update(service_data)
        url = f"/api/services/{domain}/{service}"
        if return_response:
            url += "?return_response"
        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    async def get_config(self) -> dict[str, Any]:
        """GET /api/config -- returns HA core configuration."""
        try:
            resp = await self._client.get("/api/config")
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.warning("Failed to fetch HA config", exc_info=True)
            return {}

    async def render_template(self, template: str) -> str | None:
        """POST /api/template -- render a Jinja2 template server-side.

        Returns the rendered text on success or ``None`` on any error.
        Used by callers that need to resolve registry-only data such as
        ``device_id('<entity_id>')`` without requiring full HA
        WebSocket / registry access.
        """
        if self._client is None:
            return None
        try:
            resp = await self._client.post(
                "/api/template",
                json={"template": template},
            )
            resp.raise_for_status()
            rendered = (resp.text or "").strip()
            return rendered or None
        except Exception:
            logger.debug("Template render failed for %r", template, exc_info=True)
            return None

    async def fire_event(
        self,
        event_type: str,
        event_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /api/events/<event_type>."""
        resp = await self._client.post(f"/api/events/{event_type}", json=event_data or {})
        resp.raise_for_status()
        return resp.json()

    async def get_history_period(
        self,
        start_time_utc: datetime,
        *,
        entity_id: str | None = None,
        end_time_utc: datetime | None = None,
        significant_changes_only: bool = True,
        minimal_response: bool = True,
    ) -> list[list[dict[str, Any]]]:
        """GET /api/history/period/<start> -- Recorder state changes.

        Returns HA's nested list format (one inner list per entity when
        ``filter_entity_id`` is set). Requires the Recorder integration.
        """
        if self._client is None:
            return []
        if start_time_utc.tzinfo is None:
            start_time_utc = start_time_utc.replace(tzinfo=UTC)
        start_iso = start_time_utc.astimezone(UTC).isoformat()
        path = f"/api/history/period/{quote(start_iso, safe='')}"
        params: dict[str, str] = {}
        if entity_id:
            params["filter_entity_id"] = entity_id
        if end_time_utc is not None:
            if end_time_utc.tzinfo is None:
                end_time_utc = end_time_utc.replace(tzinfo=UTC)
            params["end_time"] = end_time_utc.astimezone(UTC).isoformat()
        if significant_changes_only:
            params["significant_changes_only"] = "1"
        if minimal_response:
            params["minimal_response"] = "1"
        resp = await self._client.get(path, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    # ------------------------------------------------------------------
    # FLOW-VERIFY-1: post-action state verification helpers.
    # ------------------------------------------------------------------
    def set_state_observer(self, ws_client: HAWebSocketClient | None) -> None:
        """Attach the running WebSocket client so ``expect_state`` can use it."""
        self._state_observer = ws_client

    @asynccontextmanager
    async def expect_state(
        self,
        entity_id: str,
        *,
        expected: str | None,
        timeout: float = 1.5,
        poll_interval: float = 0.25,
        poll_max: float = 1.0,
    ):
        """Register a state-change waiter BEFORE the trigger, resolve it after.

        Priority order:
          1. If a connected WS observer is available, register a waiter for the
             next ``state_changed`` event on ``entity_id`` (matching
             ``expected`` if given). After the ``with`` body exits, wait up to
             ``timeout`` seconds for it to resolve.
          2. If no WS observer is connected or the waiter times out, fall back
             to polling :meth:`get_state` every ``poll_interval`` seconds for
             up to ``poll_max`` seconds (or until the state matches
             ``expected``; the last observed state wins on timeout).

        Yields a mutable ``dict`` with a single ``"new_state"`` entry that is
        populated on exit. Callers read the observed state from that dict.
        """
        result: dict[str, Any] = {"new_state": None}
        observer = self._state_observer
        future: asyncio.Future[str] | None = None
        if observer is not None and observer.is_connected():
            try:
                future = observer.register_state_waiter(entity_id, expected=expected)
            except Exception:
                logger.debug(
                    "Failed to register WS state waiter for %s",
                    entity_id,
                    exc_info=True,
                )
                future = None
        try:
            yield result
        except Exception:
            if future is not None and observer is not None:
                observer.cancel_state_waiter(entity_id, future)
            raise

        if future is not None and observer is not None:
            try:
                state = await asyncio.wait_for(future, timeout=timeout)
                result["new_state"] = state
                return
            except TimeoutError:
                observer.cancel_state_waiter(entity_id, future)
            except asyncio.CancelledError:
                observer.cancel_state_waiter(entity_id, future)
                raise
            except Exception:
                logger.debug(
                    "WS state waiter for %s raised, falling back to polling",
                    entity_id,
                    exc_info=True,
                )
                observer.cancel_state_waiter(entity_id, future)

        result["new_state"] = await self._poll_state_until(
            entity_id,
            expected,
            interval=poll_interval,
            max_seconds=poll_max,
        )

    async def _poll_state_until(
        self,
        entity_id: str,
        expected: str | None,
        *,
        interval: float,
        max_seconds: float,
    ) -> str | None:
        """Poll get_state until state matches expected or budget elapsed.

        Returns the last observed state (even if it never matched), so the
        caller can still include something meaningful in speech/logs.
        """
        deadline = time.monotonic() + max(0.0, max_seconds)
        last_state: str | None = None
        while True:
            try:
                state_resp = await self.get_state(entity_id)
            except Exception:
                logger.debug(
                    "get_state polling failed for %s",
                    entity_id,
                    exc_info=True,
                )
                state_resp = None
            if state_resp:
                last_state = state_resp.get("state")
                if expected is None or last_state == expected:
                    return last_state
            if time.monotonic() >= deadline:
                return last_state
            await asyncio.sleep(interval)


async def test_ha_connection(url: str, token: str) -> bool:
    """Test an HA connection with ad-hoc URL and token (for setup wizard)."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            resp = await client.get(f"{url.rstrip('/')}/api/", headers=headers)
            return resp.status_code == 200
    except httpx.HTTPError:
        return False
