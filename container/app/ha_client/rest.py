"""Async HA REST API client."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.db.repository import SettingsRepository
from app.ha_client.auth import get_auth_headers

logger = logging.getLogger(__name__)


class HARestClient:
    """Async REST client for the Home Assistant API."""

    def __init__(self) -> None:
        self._base_url: str | None = None
        self._client: httpx.AsyncClient | None = None

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

    async def _refresh_headers(self) -> None:
        """Refresh auth headers (e.g. after token update during setup)."""
        headers = await get_auth_headers()
        if headers and self._client:
            self._client.headers.update(headers)

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
    ) -> dict[str, Any]:
        """POST /api/services/<domain>/<service>."""
        payload: dict[str, Any] = {}
        if entity_id:
            payload["entity_id"] = entity_id
        if service_data:
            payload.update(service_data)
        resp = await self._client.post(f"/api/services/{domain}/{service}", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def fire_event(
        self,
        event_type: str,
        event_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /api/events/<event_type>."""
        resp = await self._client.post(
            f"/api/events/{event_type}", json=event_data or {}
        )
        resp.raise_for_status()
        return resp.json()


async def test_ha_connection(url: str, token: str) -> bool:
    """Test an HA connection with ad-hoc URL and token (for setup wizard)."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            resp = await client.get(f"{url.rstrip('/')}/api/", headers=headers)
            return resp.status_code == 200
    except httpx.HTTPError:
        return False
