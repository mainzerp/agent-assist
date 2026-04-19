"""Tests for app.ha_client.home_context -- HomeContext provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from app.ha_client.home_context import HomeContext, HomeContextProvider


class TestHomeContext:
    """Tests for the HomeContext Pydantic model."""

    def test_defaults(self):
        ctx = HomeContext()
        assert ctx.timezone == "UTC"
        assert ctx.location_name == ""

    def test_custom_values(self):
        ctx = HomeContext(timezone="Europe/Berlin", location_name="Berlin")
        assert ctx.timezone == "Europe/Berlin"
        assert ctx.location_name == "Berlin"


class TestHomeContextProvider:
    """Tests for HomeContextProvider caching and fallback."""

    async def test_refresh_from_ha_config(self):
        ha_client = AsyncMock()
        ha_client.get_config.return_value = {
            "time_zone": "America/New_York",
            "location_name": "New York",
            "latitude": 40.7,
            "longitude": -74.0,
        }
        provider = HomeContextProvider()
        ctx = await provider.refresh(ha_client)
        assert ctx.timezone == "America/New_York"
        assert ctx.location_name == "New York"

    async def test_get_returns_cached_within_ttl(self):
        ha_client = AsyncMock()
        ha_client.get_config.return_value = {
            "time_zone": "Europe/Berlin",
            "location_name": "Berlin",
        }
        provider = HomeContextProvider()
        ctx1 = await provider.get(ha_client)
        assert ctx1.timezone == "Europe/Berlin"

        # Second call should use cache, not call HA again
        ha_client.get_config.reset_mock()
        ctx2 = await provider.get(ha_client)
        assert ctx2.timezone == "Europe/Berlin"
        ha_client.get_config.assert_not_called()

    async def test_get_refreshes_after_ttl(self):
        ha_client = AsyncMock()
        ha_client.get_config.return_value = {
            "time_zone": "Europe/Berlin",
            "location_name": "Berlin",
        }
        provider = HomeContextProvider()
        provider._ttl_seconds = 0  # Expire immediately
        ctx1 = await provider.get(ha_client)
        assert ctx1.timezone == "Europe/Berlin"

        ha_client.get_config.return_value = {
            "time_zone": "Asia/Tokyo",
            "location_name": "Tokyo",
        }
        ctx2 = await provider.get(ha_client)
        assert ctx2.timezone == "Asia/Tokyo"

    async def test_refresh_falls_back_to_db_overrides(self):
        ha_client = AsyncMock()
        ha_client.get_config.return_value = {}  # Empty = HA fetch failed

        provider = HomeContextProvider()
        with patch("app.ha_client.home_context.HomeContextProvider._load_overrides") as mock_overrides:
            mock_overrides.return_value = HomeContext(
                timezone="Europe/London",
                location_name="London",
            )
            ctx = await provider.refresh(ha_client)
            assert ctx.timezone == "Europe/London"
            assert ctx.location_name == "London"

    async def test_refresh_returns_defaults_when_all_fail(self):
        ha_client = AsyncMock()
        ha_client.get_config.side_effect = Exception("connection failed")

        provider = HomeContextProvider()
        with patch("app.ha_client.home_context.HomeContextProvider._load_overrides") as mock_overrides:
            mock_overrides.return_value = None
            ctx = await provider.refresh(ha_client)
            assert ctx.timezone == "UTC"
            assert ctx.location_name == ""

    async def test_refresh_handles_none_timezone_in_config(self):
        ha_client = AsyncMock()
        ha_client.get_config.return_value = {
            "time_zone": None,
            "location_name": None,
        }
        provider = HomeContextProvider()
        ctx = await provider.refresh(ha_client)
        assert ctx.timezone == "UTC"
        assert ctx.location_name == ""
