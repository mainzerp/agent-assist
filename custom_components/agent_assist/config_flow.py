"""Config flow for agent-assist integration."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult
from homeassistant.const import CONF_URL, CONF_API_KEY

from .const import DOMAIN, DEFAULT_CONTAINER_URL, HEALTH_PATH

logger = logging.getLogger(__name__)


class AgentAssistConfigFlow(ConfigFlow, domain=DOMAIN):
    """Config flow for agent-assist."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial user configuration step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            url = user_input[CONF_URL].rstrip("/")
            api_key = user_input[CONF_API_KEY]

            # Test connection to container
            error = await self._test_connection(url, api_key)
            if error:
                errors["base"] = error
            else:
                await self.async_set_unique_id(DOMAIN)
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title="Agent Assist",
                    data={CONF_URL: url, CONF_API_KEY: api_key},
                )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_URL, default=DEFAULT_CONTAINER_URL): str,
                    vol.Required(CONF_API_KEY): str,
                }
            ),
            errors=errors,
        )

    async def _test_connection(self, url: str, api_key: str) -> str | None:
        """Test connection to the container. Returns error key or None."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {api_key}"}
                async with session.get(
                    f"{url}{HEALTH_PATH}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 401:
                        return "invalid_auth"
                    if resp.status != 200:
                        return "cannot_connect"
                    data = await resp.json()
                    if data.get("status") != "ok":
                        return "cannot_connect"
        except (aiohttp.ClientError, TimeoutError):
            return "cannot_connect"
        return None
