"""Conversation entity for agent-assist (I/O bridge)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Literal

import aiohttp

from homeassistant.components import assist_pipeline, conversation
from homeassistant.components.conversation import ConversationEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_URL, CONF_API_KEY, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er, intent
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    DOMAIN,
    WS_PATH,
    RECONNECT_BASE_DELAY,
    RECONNECT_MAX_DELAY,
)

logger = logging.getLogger(__name__)


def _strip_markdown(text: str) -> str:
    """Remove Markdown formatting for TTS-friendly output."""
    if not text:
        return text
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"^[\s]*([-*_]){3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the conversation entity from a config entry."""
    # Migrate legacy unique_id formats to entry.entry_id
    entity_registry = er.async_get(hass)
    for old_uid in (DOMAIN, f"{DOMAIN}_conversation"):
        entity_id = entity_registry.async_get_entity_id("conversation", DOMAIN, old_uid)
        if entity_id:
            entity_registry.async_update_entity(entity_id, new_unique_id=entry.entry_id)
            logger.info("Migrated entity %s unique_id from '%s' to '%s'", entity_id, old_uid, entry.entry_id)

    data = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([AgentAssistConversationEntity(entry, data["url"], data["api_key"])])


class AgentAssistConversationEntity(
    conversation.ConversationEntity,
):
    """Conversation entity that bridges HA voice to the agent-assist container."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_should_poll = False
    _attr_supported_features = ConversationEntityFeature.CONTROL

    def __init__(self, entry: ConfigEntry, url: str, api_key: str) -> None:
        self._entry = entry
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._attr_unique_id = entry.entry_id
        self._reconnect_delay = RECONNECT_BASE_DELAY
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Agent Assist",
            model="Multi-Agent Assistant",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        try:
            assist_pipeline.async_migrate_engine(
                self.hass, "conversation", self._entry.entry_id, self.entity_id
            )
        except Exception:
            logger.debug("Pipeline engine migration skipped (not critical)")
        self._reconnect_task = self._entry.async_create_background_task(
            self.hass,
            self._reconnect_loop(),
            name="agent_assist_ws_reconnect",
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        if hasattr(self, "_reconnect_task") and self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None
        await self._disconnect_ws()
        await super().async_will_remove_from_hass()

    async def _connect_ws(self) -> bool:
        """Establish persistent WebSocket connection to the container."""
        try:
            if self._session is None:
                self._session = aiohttp.ClientSession()

            ws_url = self._url.replace("http://", "ws://").replace("https://", "wss://")
            self._ws = await self._session.ws_connect(
                f"{ws_url}{WS_PATH}",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=aiohttp.ClientTimeout(total=10),
            )
            self._reconnect_delay = RECONNECT_BASE_DELAY
            logger.info("Connected to agent-assist container at %s", self._url)
            return True
        except (aiohttp.ClientError, TimeoutError):
            logger.warning("Failed to connect to container at %s", self._url)
            # Clean up session to prevent resource leak
            if self._session:
                try:
                    await self._session.close()
                except Exception:
                    pass
                self._session = None
            self._ws = None
            return False

    async def _disconnect_ws(self) -> None:
        """Close the WebSocket and session."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = None
        if self._session:
            await self._session.close()
            self._session = None

    async def _reconnect_loop(self) -> None:
        """Background loop that maintains the WebSocket connection."""
        while True:
            if self._ws is None or self._ws.closed:
                connected = await self._connect_ws()
                if not connected:
                    delay = self._reconnect_delay
                    self._reconnect_delay = min(self._reconnect_delay * 2, RECONNECT_MAX_DELAY)
                    logger.debug("Reconnect in %.1fs", delay)
                    await asyncio.sleep(delay)
                    continue
            # Connection is alive -- sleep before checking again
            await asyncio.sleep(30)

    async def _ensure_connected(self) -> bool:
        """Ensure WebSocket is connected, reconnect if needed."""
        if self._ws is not None and not self._ws.closed:
            return True
        # Attempt reconnect
        connected = await self._connect_ws()
        if not connected:
            self._reconnect_delay = min(self._reconnect_delay * 2, RECONNECT_MAX_DELAY)
        return connected

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process a conversation turn by forwarding to the container."""
        if not await self._ensure_connected():
            return await self._process_via_rest(user_input)

        try:
            return await self._process_via_ws(user_input)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            logger.warning("WebSocket error, falling back to REST")
            await self._disconnect_ws()
            return await self._process_via_rest(user_input)

    async def _process_via_ws(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        """Send request via WebSocket and accumulate streaming tokens."""
        device_id = getattr(user_input, "device_id", None)
        area_id = None
        if device_id:
            device_reg = dr.async_get(self.hass)
            device = device_reg.async_get(device_id)
            if device:
                area_id = device.area_id
        payload = {
            "text": user_input.text,
            "conversation_id": user_input.conversation_id,
            "language": user_input.language or "en",
        }
        if device_id:
            payload["device_id"] = device_id
        if area_id:
            payload["area_id"] = area_id
        await self._ws.send_json(payload)

        speech_parts: list[str] = []
        final_conversation_id = user_input.conversation_id

        while True:
            msg = await asyncio.wait_for(self._ws.receive(), timeout=30.0)
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)

                # Handle filler tokens -- speak immediately via TTS, do not accumulate
                if data.get("is_filler", False):
                    filler_text = data.get("token", "")
                    if filler_text:
                        await self._speak_filler(filler_text, user_input)
                    continue

                token_text = data.get("token", "")
                if token_text:
                    speech_parts.append(token_text)
                if data.get("done", False):
                    final_conversation_id = data.get("conversation_id", final_conversation_id)
                    mediated = data.get("mediated_speech")
                    if mediated:
                        speech_parts = [mediated]
                    break
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break

        speech = "".join(speech_parts)
        return self._build_result(speech, final_conversation_id, user_input.language)

    async def _process_via_rest(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        """Fallback: send request via REST and get full response."""
        device_id = getattr(user_input, "device_id", None)
        area_id = None
        if device_id:
            device_reg = dr.async_get(self.hass)
            device = device_reg.async_get(device_id)
            if device:
                area_id = device.area_id
        try:
            if self._session is None:
                self._session = aiohttp.ClientSession()
            headers = {"Authorization": f"Bearer {self._api_key}"}
            payload = {
                "text": user_input.text,
                "conversation_id": user_input.conversation_id,
                "language": user_input.language or "en",
            }
            if device_id:
                payload["device_id"] = device_id
            if area_id:
                payload["area_id"] = area_id
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

    def _build_result(self, speech: str, conversation_id: str | None, language: str | None) -> conversation.ConversationResult:
        """Assemble a ConversationResult from the response."""
        response = intent.IntentResponse(language=language or "en")
        response.async_set_speech(_strip_markdown(speech))
        return conversation.ConversationResult(response=response, conversation_id=conversation_id)

    async def _speak_filler(self, text: str, user_input) -> None:
        """Speak filler text immediately via TTS, bypassing the conversation result."""
        try:
            device_id = getattr(user_input, "device_id", None)
            if not device_id:
                return
            tts_entity = self._resolve_tts_entity(device_id)
            if not tts_entity:
                return
            await self.hass.services.async_call(
                "tts",
                "speak",
                {
                    "entity_id": tts_entity,
                    "message": _strip_markdown(text),
                },
                blocking=False,
            )
        except Exception:
            logger.debug("Failed to speak filler text", exc_info=True)

    def _resolve_tts_entity(self, device_id: str) -> str | None:
        """Resolve a device_id to a TTS-capable media_player entity in the same area."""
        try:
            device_reg = dr.async_get(self.hass)
            device = device_reg.async_get(device_id)
            if not device or not device.area_id:
                return None
            entity_reg = er.async_get(self.hass)
            for entry in entity_reg.entities.values():
                if (
                    entry.domain == "media_player"
                    and entry.device_id
                ):
                    mp_device = device_reg.async_get(entry.device_id)
                    if mp_device and mp_device.area_id == device.area_id:
                        return entry.entity_id
            return None
        except Exception:
            logger.debug("Failed to resolve TTS entity for device %s", device_id)
            return None
