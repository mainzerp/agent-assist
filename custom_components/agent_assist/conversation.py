"""Conversation entity for agent-assist (I/O bridge)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import aiohttp

from homeassistant.components import conversation
from homeassistant.components.conversation import ConversationEntity, ConversationInput, ConversationResult
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_URL, CONF_API_KEY, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers import intent

from .const import (
    DOMAIN,
    WS_PATH,
    RECONNECT_BASE_DELAY,
    RECONNECT_MAX_DELAY,
)

logger = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the conversation entity from a config entry."""
    data = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([AgentAssistConversationEntity(entry, data["url"], data["api_key"])])


class AgentAssistConversationEntity(ConversationEntity):
    """Conversation entity that bridges HA voice to the agent-assist container."""

    _attr_has_entity_name = True
    _attr_name = "Agent Assist"
    _attr_supported_languages = MATCH_ALL

    def __init__(self, entry: ConfigEntry, url: str, api_key: str) -> None:
        self._entry = entry
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}"
        self._reconnect_delay = RECONNECT_BASE_DELAY

    @property
    def supported_features(self) -> conversation.ConversationEntityFeature:
        """Return supported features."""
        return conversation.ConversationEntityFeature.CONTROL

    async def async_added_to_hass(self) -> None:
        """Initialize WebSocket connection when entity is added."""
        await self._connect_ws()

    async def async_will_remove_from_hass(self) -> None:
        """Close WebSocket connection when entity is removed."""
        await self._disconnect_ws()

    async def _connect_ws(self) -> bool:
        """Establish persistent WebSocket connection to the container."""
        try:
            if self._session is None:
                self._session = aiohttp.ClientSession()

            ws_url = self._url.replace("http://", "ws://").replace("https://", "wss://")
            self._ws = await self._session.ws_connect(
                f"{ws_url}{WS_PATH}?token={self._api_key}",
                timeout=aiohttp.ClientTimeout(total=10),
            )
            self._reconnect_delay = RECONNECT_BASE_DELAY
            logger.info("Connected to agent-assist container at %s", self._url)
            return True
        except (aiohttp.ClientError, TimeoutError):
            logger.warning("Failed to connect to container at %s", self._url)
            return False

    async def _disconnect_ws(self) -> None:
        """Close the WebSocket and session."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = None
        if self._session:
            await self._session.close()
            self._session = None

    async def _ensure_connected(self) -> bool:
        """Ensure WebSocket is connected, reconnect if needed."""
        if self._ws is not None and not self._ws.closed:
            return True
        # Attempt reconnect
        connected = await self._connect_ws()
        if not connected:
            self._reconnect_delay = min(self._reconnect_delay * 2, RECONNECT_MAX_DELAY)
        return connected

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a conversation turn by forwarding to the container."""
        if not await self._ensure_connected():
            # Fallback: try REST endpoint
            return await self._process_via_rest(user_input)

        try:
            return await self._process_via_ws(user_input)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            logger.warning("WebSocket error, falling back to REST")
            await self._disconnect_ws()
            return await self._process_via_rest(user_input)

    async def _process_via_ws(self, user_input: ConversationInput) -> ConversationResult:
        """Send request via WebSocket and accumulate streaming tokens."""
        payload = {
            "text": user_input.text,
            "conversation_id": user_input.conversation_id,
            "language": user_input.language or "en",
        }
        await self._ws.send_json(payload)

        speech_parts: list[str] = []
        final_conversation_id = user_input.conversation_id

        while True:
            msg = await asyncio.wait_for(self._ws.receive(), timeout=30.0)
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                token_text = data.get("token", "")
                if token_text:
                    speech_parts.append(token_text)
                if data.get("done", False):
                    final_conversation_id = data.get("conversation_id", final_conversation_id)
                    break
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break

        speech = "".join(speech_parts)
        return self._build_result(speech, final_conversation_id, user_input.language)

    async def _process_via_rest(self, user_input: ConversationInput) -> ConversationResult:
        """Fallback: send request via REST and get full response."""
        try:
            if self._session is None:
                self._session = aiohttp.ClientSession()
            headers = {"Authorization": f"Bearer {self._api_key}"}
            payload = {
                "text": user_input.text,
                "conversation_id": user_input.conversation_id,
                "language": user_input.language or "en",
            }
            async with self._session.post(
                f"{self._url}/api/conversation",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    return self._build_result(
                        "Sorry, I could not reach the assistant container.",
                        user_input.conversation_id,
                        user_input.language,
                    )
                data = await resp.json()
                return self._build_result(
                    data.get("speech", ""),
                    data.get("conversation_id", user_input.conversation_id),
                    user_input.language,
                )
        except (aiohttp.ClientError, TimeoutError):
            return self._build_result(
                "Sorry, the assistant container is unavailable.",
                user_input.conversation_id,
                user_input.language,
            )

    def _build_result(self, speech: str, conversation_id: str | None, language: str | None) -> ConversationResult:
        """Assemble a ConversationResult from the response."""
        response = intent.IntentResponse(language=language or "en")
        response.async_set_speech(speech)
        return ConversationResult(response=response, conversation_id=conversation_id)
