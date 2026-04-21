"""Config flow for HA-AgentHub integration."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.const import CONF_URL, CONF_API_KEY
from homeassistant.helpers.selector import TextSelector, TextSelectorConfig, TextSelectorType

from .const import DOMAIN, DEFAULT_CONTAINER_URL, HEALTH_PATH, INTEGRATION_TITLE

logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    return (url or "").strip().rstrip("/")


def _password_selector() -> TextSelector:
    return TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))


def _build_user_schema() -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_URL, default=DEFAULT_CONTAINER_URL): str,
            vol.Required(CONF_API_KEY): _password_selector(),
        }
    )


def _build_options_schema(current: dict[str, Any]) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_URL, default=current.get(CONF_URL, DEFAULT_CONTAINER_URL)): str,
            vol.Optional(CONF_API_KEY, default=""): _password_selector(),
        }
    )


async def _validate_connection(url: str, api_key: str) -> str | None:
    """Test connection to the container. Returns error key or None."""
    normalized_url = _normalize_url(url)
    trimmed_key = (api_key or "").strip()
    if not normalized_url or not trimmed_key:
        return "invalid_auth"

    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {trimmed_key}"}
            async with session.get(
                f"{normalized_url}{HEALTH_PATH}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status in {401, 403}:
                    return "invalid_auth"
                if resp.status != 200:
                    return "cannot_connect"
                data = await resp.json()
                if data.get("status") != "ok":
                    return "cannot_connect"
    except (aiohttp.ClientError, TimeoutError, ValueError):
        return "cannot_connect"
    return None


class HaAgentHubConfigFlow(ConfigFlow, domain=DOMAIN):
    """Config flow for HA-AgentHub."""

    VERSION = 1

    @staticmethod
    def async_get_options_flow(config_entry: ConfigEntry) -> HaAgentHubOptionsFlow:
        return HaAgentHubOptionsFlow(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial user configuration step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            url = _normalize_url(user_input[CONF_URL])
            api_key = (user_input[CONF_API_KEY] or "").strip()

            error = await _validate_connection(url, api_key)
            if error:
                errors["base"] = error
            else:
                await self.async_set_unique_id(DOMAIN)
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title=INTEGRATION_TITLE,
                    data={CONF_URL: url, CONF_API_KEY: api_key},
                )

        return self.async_show_form(
            step_id="user",
            data_schema=_build_user_schema(),
            errors=errors,
        )


class HaAgentHubOptionsFlow(OptionsFlow):
    """Options flow for reconfiguring HA-AgentHub."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        self._entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle options flow."""
        errors: dict[str, str] = {}
        current = self._entry.data

        if user_input is not None:
            url = _normalize_url(user_input[CONF_URL])
            new_api_key = (user_input.get(CONF_API_KEY) or "").strip()
            api_key = new_api_key or current.get(CONF_API_KEY, "")

            error = await _validate_connection(url, api_key)
            if error:
                errors["base"] = error
            else:
                self.hass.config_entries.async_update_entry(
                    self._entry,
                    data={CONF_URL: url, CONF_API_KEY: api_key},
                )
                return self.async_create_entry(data={})

        return self.async_show_form(
            step_id="init",
            data_schema=_build_options_schema(current),
            errors=errors,
        )
