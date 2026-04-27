"""Conversation entity for HA-AgentHub (I/O bridge)."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp

from homeassistant.components import assist_pipeline, conversation
from homeassistant.components.conversation import ConversationEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_URL, CONF_API_KEY, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar, device_registry as dr, entity_registry as er, intent
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

try:
    from homeassistant.helpers.event import async_track_state_change_event
except ModuleNotFoundError:
    async_track_state_change_event = None

from .const import (
    DOMAIN,
    WS_PATH,
    RECONNECT_BASE_DELAY,
    RECONNECT_MAX_DELAY,
    WS_HEARTBEAT_INTERVAL,
    WS_IDLE_THRESHOLD,
    CONF_NATIVE_PLAIN_TIMERS,
    DEFAULT_NATIVE_PLAIN_TIMERS,
    CONF_ENABLE_POST_FILLER_PUSH,
    DEFAULT_ENABLE_POST_FILLER_PUSH,
    NATIVE_HA_AGENT_ID,
    NATIVE_PLAIN_TIMER_DIRECTIVE,
    NATIVE_PLAIN_TIMER_ELIGIBLE_FIELD,
    NATIVE_PLAIN_TIMER_ELIGIBLE_HEADER,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Native plain-timer delegation (0.25.1)
# ---------------------------------------------------------------------------
#
# The integration no longer classifies utterances locally. Instead, when the
# per-config-entry ``CONF_NATIVE_PLAIN_TIMERS`` opt-in is enabled, every
# bridge request is marked eligible (additive JSON field + REST header).
# The container timer-agent owns the semantic decision and may return a
# ``directive=delegate_native_plain_timer`` response through the normal
# orchestrator path. The integration honours the directive by calling the
# proven native seam (``conversation.async_converse(...,
# agent_id=NATIVE_HA_AGENT_ID)``).
#
# Recursion safety: ``_async_delegate_to_native`` falls back to the bridge
# on pre-handler errors. To prevent that fallback from triggering a second
# directive loop, eligibility is suppressed via a task-local ContextVar
# while a directive is being honoured.


@dataclass(slots=True)
class _BridgeDirective:
    """Internal carrier returned by bridge senders when the container
    instructs the integration to delegate to native HA Assist."""

    directive: str
    reason: str | None = None
    conversation_id: str | None = None


# Task-local suppression of the eligibility flag/header. When True, neither
# the WebSocket payload nor the REST request includes the eligibility
# signal so the bridge cannot
# emit a second native directive for the same turn.
_suppress_native_plain_timer_eligibility: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ha_agenthub_suppress_native_plain_timer_eligibility",
    default=False,
)

MAX_POST_FILLER_WAIT_SECONDS = 8.0
PUSH_FINAL_WAIT_SECONDS = 30.0
SPEAK_FILLER_ONLY_ON_TIMEOUT = False
FILLER_ONLY_TIMEOUT_TEXT = "Entschuldigung, da ist etwas schiefgelaufen."
_SAT_BUSY_STATES = frozenset({"listening", "processing", "responding"})
_SAT_IDLE_STATES = frozenset({"idle"})


class _WsDroppedAfterSendError(Exception):
    """Request was written to the WebSocket; REST fallback would duplicate server work."""


def _rest_fallback_error_message(status_code: int | None) -> str:
    """Return an actionable fallback message for REST error responses."""
    if status_code in {401, 403}:
        return (
            "Sorry, the HA-AgentHub integration API key was rejected. "
            "Update the API key in the HA-AgentHub integration settings."
        )
    if status_code is not None and status_code >= 500:
        return (
            "Sorry, the assistant container returned an error. "
            "Check the configured container URL and the container logs."
        )
    return (
        "Sorry, the assistant container returned an unexpected response. "
        "Check the configured container URL and the container logs."
    )


def _strip_markdown(text: str) -> str:
    """Remove Markdown formatting for TTS-friendly output.

    FLOW-MED-4 / P3-1: this function is now a *defensive fallback only*.
    The container backend strips Markdown via
    ``container/app/agents/sanitize.strip_markdown`` and advertises the
    fact through the ``sanitized`` field on its REST/WebSocket responses
    (see ``ConversationResponse`` / ``StreamToken``). When that flag is
    True, ``_build_result`` skips this pass and treats the backend as
    the single source of truth. The implementation is kept in lock-step
    with the backend so legacy containers (< 0.18.35) and filler tokens
    (which are emitted unsanitized) still produce TTS-friendly output.
    """
    if not text:
        return text
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", text)
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
    # Migrate legacy unique_id formats (incl. pre-0.5 domain ``agent_assist``)
    entity_registry = er.async_get(hass)
    _legacy_domain = "agent_assist"
    migration_pairs = [
        (_legacy_domain, "agent_assist"),
        (_legacy_domain, "agent_assist_conversation"),
        (_legacy_domain, _legacy_domain),
        (DOMAIN, DOMAIN),
        (DOMAIN, f"{DOMAIN}_conversation"),
    ]
    for int_domain, old_uid in migration_pairs:
        entity_id = entity_registry.async_get_entity_id("conversation", int_domain, old_uid)
        if entity_id:
            entity_registry.async_update_entity(entity_id, new_unique_id=entry.entry_id)
            logger.info(
                "Migrated entity %s unique_id from %s/%s to %s",
                entity_id, int_domain, old_uid, entry.entry_id,
            )

    data = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([HaAgentHubConversationEntity(entry, data["url"], data["api_key"])])


class HaAgentHubConversationEntity(
    conversation.ConversationEntity,
):
    """Conversation entity that bridges HA voice to the HA-AgentHub container."""

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
        self._ws_lock = asyncio.Lock()
        self._ws_last_active: float = 0.0
        # Coalesce parallel HA calls with the same conversation_id + text (duplicate
        # pipeline invocations or WS+REST overlap) into a single bridge request.
        self._coalesce_lock = asyncio.Lock()
        # FLOW-COALESCE-1 (P2-3): value is (started_monotonic, task). The
        # started-timestamp guards a legitimate repeat of the same utterance
        # that arrives after the original response was already rendered --
        # without it we would short-circuit the second request onto the
        # first completed task forever.
        self._inflight_bridge: dict[tuple[str, str], tuple[float, asyncio.Task]] = {}
        self._coalesce_window_sec: float = 0.25
        # V4: at most one in-flight post-filler push task per satellite.
        self._inflight_pushes: dict[str, asyncio.Task] = {}
        # V4 reentrancy guard for assist_satellite.announce echo loops.
        self._push_in_progress_satellites: set[str] = set()
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="HA-AgentHub",
            model="Conversation bridge",
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
            name="ha_agenthub_ws_reconnect",
        )

        def _cancel_pushes() -> None:
            for sat_id, task in list(self._inflight_pushes.items()):
                if not task.done():
                    task.cancel()
            self._inflight_pushes.clear()

        self._entry.async_on_unload(_cancel_pushes)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        if hasattr(self, "_reconnect_task") and self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None
        for sat_id, task in list(self._inflight_pushes.items()):
            task.cancel()
        self._inflight_pushes.clear()
        await self._disconnect_ws()
        await super().async_will_remove_from_hass()

    async def _connect_ws(self) -> bool:
        """Establish persistent WebSocket connection to the container."""
        async with self._ws_lock:
            return await self._connect_ws_locked()

    async def _connect_ws_locked(self) -> bool:
        """Locked body of :meth:`_connect_ws`. Caller MUST hold
        ``self._ws_lock``. See FLOW-HIGH-8."""
        try:
            if self._session is None:
                self._session = aiohttp.ClientSession()

            ws_url = self._url.replace("http://", "ws://").replace("https://", "wss://")
            self._ws = await self._session.ws_connect(
                f"{ws_url}{WS_PATH}",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=aiohttp.ClientTimeout(total=10),
                heartbeat=WS_HEARTBEAT_INTERVAL,
            )
            self._reconnect_delay = RECONNECT_BASE_DELAY
            self._ws_last_active = time.monotonic()
            logger.info("Connected to HA-AgentHub container at %s", self._url)
            return True
        except (aiohttp.ClientError, TimeoutError):
            logger.warning("Failed to connect to container at %s", self._url)
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
        async with self._ws_lock:
            await self._disconnect_ws_locked()

    async def _disconnect_ws_locked(self) -> None:
        """Locked body of :meth:`_disconnect_ws`. Caller MUST hold
        ``self._ws_lock``."""
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
        async with self._ws_lock:
            return await self._ensure_connected_locked()

    async def _ensure_connected_locked(self) -> bool:
        """Body of :meth:`_ensure_connected` that assumes the caller
        already holds ``self._ws_lock``.

        FLOW-HIGH-8 extracts this so ``_async_handle_message`` can
        hold the lock across both the connectivity check and the
        subsequent send -- closing the race where the WS flips to
        closed between the two calls.
        """
        if self._ws is not None and not self._ws.closed:
            if time.monotonic() - self._ws_last_active > WS_IDLE_THRESHOLD:
                try:
                    pong = await self._ws.ping()
                    await asyncio.wait_for(pong, timeout=2.0)
                    self._ws_last_active = time.monotonic()
                except (asyncio.TimeoutError, Exception):
                    logger.warning("WebSocket idle ping failed, reconnecting")
                    await self._disconnect_ws_locked()
                    return await self._connect_ws_locked()
            return True
        connected = await self._connect_ws_locked()
        if not connected:
            self._reconnect_delay = min(self._reconnect_delay * 2, RECONNECT_MAX_DELAY)
        return connected

    def _schedule_reconnect(self) -> None:
        """Schedule an immediate background WS reconnect."""
        self._reconnect_delay = RECONNECT_BASE_DELAY
        self._entry.async_create_background_task(
            self.hass,
            self._connect_ws(),
            name="ha_agenthub_ws_immediate_reconnect",
        )

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process a conversation turn by forwarding to the container.

        FLOW-HIGH-8: hold ``self._ws_lock`` across both the
        connectivity probe and the actual send so the socket cannot
        flip to closed between the two steps. All REST-fallback paths
        run *outside* the lock to avoid serializing fallback traffic
        behind a hung WS send.

        Duplicate invocations with the same ``conversation_id`` and user
        text are coalesced so only one WebSocket/REST round-trip runs;
        this matches traces where the container saw two identical turns
        back-to-back from production HA setups.

        0.25.1: the integration no longer classifies utterances locally.
        Native plain-timer delegation is decided by the container's timer-agent
        path and surfaces as a directive on the bridge response;
        ``_async_bridge_with_cleanup`` honours the directive inside the
        coalesced task so duplicate suppression still applies.
        """
        cid = user_input.conversation_id or ""
        text = (user_input.text or "").strip()
        device_id = getattr(user_input, "device_id", None)
        if isinstance(device_id, str) and device_id:
            sat = self._resolve_satellite_entity(user_input)
            if sat and sat in self._push_in_progress_satellites:
                logger.warning(
                    "ha-agenthub: suppressing reentrant turn from satellite mid-push sat=%s text_len=%d",
                    sat, len(text),
                )
                response = intent.IntentResponse(language=user_input.language or "en")
                response.async_set_speech("")
                return conversation.ConversationResult(
                    response=response,
                    conversation_id=user_input.conversation_id,
                )
        key = (cid, text)

        coalesced = False
        async with self._coalesce_lock:
            existing = self._inflight_bridge.get(key)
            now = time.monotonic()
            if existing is not None and (now - existing[0]) < self._coalesce_window_sec:
                bridge_task = existing[1]
                coalesced = True
            else:
                bridge_task = self.hass.async_create_task(
                    self._async_bridge_with_cleanup(user_input, key)
                )
                self._inflight_bridge[key] = (now, bridge_task)
        if coalesced:
            logger.info(
                "HA-AgentHub: coalescing duplicate request (same conversation + text) onto in-flight bridge"
            )
        return await bridge_task

    async def _async_bridge_with_cleanup(
        self,
        user_input: conversation.ConversationInput,
        key: tuple[str, str],
    ) -> conversation.ConversationResult:
        task = asyncio.current_task()
        try:
            outcome = await self._async_bridge_to_container(user_input)
            if isinstance(outcome, _BridgeDirective):
                return await self._handle_bridge_directive(user_input, outcome)
            return outcome
        finally:
            async with self._coalesce_lock:
                existing = self._inflight_bridge.get(key)
                if task is not None and existing is not None and existing[1] is task:
                    self._inflight_bridge.pop(key, None)

    async def _handle_bridge_directive(
        self,
        user_input: conversation.ConversationInput,
        directive: _BridgeDirective,
    ) -> conversation.ConversationResult:
        """Honour a directive returned by the container bridge.

        Currently only ``delegate_native_plain_timer`` is supported. The
        eligibility flag is suppressed for the duration of the native
        attempt so that ``_async_delegate_to_native``'s own pre-handler
        bridge fallback cannot trigger a second directive loop.
        """
        if directive.directive == NATIVE_PLAIN_TIMER_DIRECTIVE:
            native_callable = self._resolve_native_delegate()
            reason = directive.reason or "native"
            if native_callable is None:
                logger.warning(
                    "HA-AgentHub: native delegate unavailable, retrying bridge "
                    "(path=agenthub, reason=native_unavailable)"
                )
                token = _suppress_native_plain_timer_eligibility.set(True)
                try:
                    fallback = await self._async_bridge_to_container(user_input)
                finally:
                    _suppress_native_plain_timer_eligibility.reset(token)
                if isinstance(fallback, _BridgeDirective):
                    # Bridge unexpectedly emitted a second directive even with
                    # suppression on. Surface a benign error instead of looping.
                    return self._build_result(
                        "Sorry, the assistant could not complete that request.",
                        user_input.conversation_id,
                        user_input.language,
                    )
                return fallback
            logger.debug(
                "HA-AgentHub: honouring native plain-timer directive "
                "(path=native, reason=%s)",
                reason,
            )
            token = _suppress_native_plain_timer_eligibility.set(True)
            try:
                return await self._async_delegate_to_native(
                    user_input, native_callable, reason
                )
            finally:
                _suppress_native_plain_timer_eligibility.reset(token)

        # Unknown directive: log and run one bridge fallback with eligibility
        # suppressed. Never recurse on unknown values.
        logger.warning(
            "HA-AgentHub: ignoring unknown bridge directive %r (path=agenthub)",
            directive.directive,
        )
        token = _suppress_native_plain_timer_eligibility.set(True)
        try:
            fallback = await self._async_bridge_to_container(user_input)
        finally:
            _suppress_native_plain_timer_eligibility.reset(token)
        if isinstance(fallback, _BridgeDirective):
            return self._build_result(
                "Sorry, the assistant could not complete that request.",
                user_input.conversation_id,
                user_input.language,
            )
        return fallback

    # ------------------------------------------------------------------
    # Native plain-timer delegation helpers (0.25.0)
    # ------------------------------------------------------------------

    def _is_native_plain_timers_enabled(self) -> bool:
        """Return True if the integration is opted into native plain-timer
        delegation. Default False keeps existing behavior unchanged when the
        flag is absent or the entry data is missing."""
        try:
            data = getattr(self._entry, "data", None) or {}
            return bool(data.get(CONF_NATIVE_PLAIN_TIMERS, DEFAULT_NATIVE_PLAIN_TIMERS))
        except Exception:
            return DEFAULT_NATIVE_PLAIN_TIMERS

    def _is_post_filler_push_enabled(self) -> bool:
        """Return True if post-filler announce push is enabled."""
        try:
            options = getattr(self._entry, "options", None) or {}
            return bool(
                options.get(
                    CONF_ENABLE_POST_FILLER_PUSH,
                    DEFAULT_ENABLE_POST_FILLER_PUSH,
                )
            )
        except Exception:
            return DEFAULT_ENABLE_POST_FILLER_PUSH

    def _resolve_native_delegate(self):
        """Resolve the HA conversation delegate seam.

        Phase 1 proven seam: ``conversation.async_converse(..., agent_id=
        NATIVE_HA_AGENT_ID)``. The ``agent_id`` ensures HA core dispatches
        directly to the built-in default agent, never re-entering this
        custom entity. Returns the callable or None if the API is missing
        on the running HA core (e.g., very old core or the stub used in
        tests).
        """
        try:
            return getattr(conversation, "async_converse", None)
        except Exception:
            return None

    async def _async_delegate_to_native(
        self,
        user_input: conversation.ConversationInput,
        native_callable,
        reason_code: str,
    ) -> conversation.ConversationResult:
        """Delegate the request to HA's built-in default conversation agent.

        On success the native ConversationResult is returned directly; per
        plan D9 we never retry the request through AgentHub once native
        has produced a definitive response. Only delegate-side construction
        errors (raised before the native handler runs) fall through to the
        AgentHub bridge as a safety net.
        """
        context = getattr(user_input, "context", None)
        try:
            result = await native_callable(
                self.hass,
                user_input.text,
                conversation_id=user_input.conversation_id,
                context=context,
                language=user_input.language,
                agent_id=NATIVE_HA_AGENT_ID,
            )
            logger.info(
                "HA-AgentHub: native Assist handled plain timer "
                "(path=native, reason=%s)",
                reason_code,
            )
            return result
        except Exception:
            # Definitive native handler errors (e.g., intent-not-matched)
            # are surfaced inside ConversationResult, not raised. A raised
            # exception here means we never reached the native handler --
            # safe to fall back to AgentHub once.
            logger.warning(
                "HA-AgentHub: native delegation failed before handler ran, "
                "falling back to AgentHub (path=agenthub, reason=native_error)",
                exc_info=True,
            )
            return await self._async_bridge_to_container(user_input)

    async def _async_bridge_to_container(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult | _BridgeDirective:
        """Single WS (preferred) or REST attempt to the HA-AgentHub container."""
        try:
            async with self._ws_lock:
                if await self._ensure_connected_locked():
                    try:
                        return await self._process_via_ws(user_input)
                    except _WsDroppedAfterSendError:
                        logger.warning(
                            "WebSocket failed after the request was sent; skipping REST "
                            "(avoids duplicate container traces)",
                            exc_info=True,
                        )
                        await self._disconnect_ws_locked()
                        return self._build_result(
                            "The connection dropped before the reply finished. "
                            "If the action may have run, check your devices.",
                            user_input.conversation_id,
                            user_input.language,
                        )
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        logger.warning("WebSocket error, falling back to REST")
                        await self._disconnect_ws_locked()
        except Exception:
            logger.warning("Unexpected WS dispatch failure, falling back to REST", exc_info=True)

        result = await self._process_via_rest(user_input)
        self._schedule_reconnect()
        return result

    def _resolve_origin_context(self, user_input: conversation.ConversationInput) -> dict[str, str]:
        """Resolve device_id/area_id and their human-readable names.

        FLOW-CTX-1 (0.18.6): IDs alone were not enough for traces or
        area-aware entity resolution. Adding the display names here
        means the container can annotate speech and the trace UI
        with "Kitchen Satellite / Kitchen" instead of opaque UUIDs.
        Lookup failures degrade silently -- the IDs stay authoritative.
        """
        extra: dict[str, str] = {}
        device_id = getattr(user_input, "device_id", None)
        if not device_id:
            return extra
        extra["device_id"] = device_id
        try:
            device_reg = dr.async_get(self.hass)
            device = device_reg.async_get(device_id)
        except Exception:
            device = None
        if not device:
            return extra

        device_name = device.name_by_user or device.name
        if device_name:
            extra["device_name"] = device_name

        area_id = device.area_id
        if not area_id:
            return extra
        extra["area_id"] = area_id
        try:
            area_reg = ar.async_get(self.hass)
            area = area_reg.async_get_area(area_id)
            if area and area.name:
                extra["area_name"] = area.name
        except Exception:
            logger.debug("area_registry lookup failed for %s", area_id, exc_info=True)
        return extra

    def _filler_gate_key(self, user_input) -> str:
        """Return the per-origin key used to gate filler completion."""
        device_id = getattr(user_input, "device_id", None)
        if isinstance(device_id, str) and device_id:
            return f"device:{device_id}"
        area_id = getattr(user_input, "area_id", None)
        if isinstance(area_id, str) and area_id:
            return f"area:{area_id}"
        return "__global__"

    def _spawn_post_filler_push(
        self,
        *,
        local_ws: aiohttp.ClientWebSocketResponse,
        satellite_entity_id: str | None,
        user_input: conversation.ConversationInput,
        gate_key: str,
    ) -> None:
        """Spawn the post-filler background push task."""
        key = satellite_entity_id or f"__no_sat__:{gate_key}"
        previous = self._inflight_pushes.get(key)
        if previous is not None and not previous.done():
            logger.info(
                "ha-agenthub: cancelling previous post-filler push key=%s sat=%s",
                gate_key, satellite_entity_id,
            )
            previous.cancel()
        task = self._entry.async_create_background_task(
            self.hass,
            self._post_filler_push(
                local_ws=local_ws,
                satellite_entity_id=satellite_entity_id,
                user_input=user_input,
                gate_key=gate_key,
                key=key,
            ),
            name=f"ha_agenthub_post_filler_push:{key}",
        )
        self._inflight_pushes[key] = task

    async def _post_filler_push(
        self,
        *,
        local_ws: aiohttp.ClientWebSocketResponse,
        satellite_entity_id: str | None,
        user_input: conversation.ConversationInput,
        gate_key: str,
        key: str,
    ) -> None:
        """Read the post-filler final response and push it after idle."""
        final_text: str | None = None
        final_parts: list[str] = []
        observed_idle = asyncio.Event()
        aborted_new_turn = False
        unsub = None

        def _on_state(event) -> None:
            nonlocal aborted_new_turn
            new_state = event.data.get("new_state") if event else None
            new_state_value = getattr(new_state, "state", None)
            if new_state_value in _SAT_IDLE_STATES:
                observed_idle.set()
            elif new_state_value in _SAT_BUSY_STATES and observed_idle.is_set():
                aborted_new_turn = True
                observed_idle.set()

        try:
            if satellite_entity_id and async_track_state_change_event is not None:
                unsub = async_track_state_change_event(
                    self.hass,
                    [satellite_entity_id],
                    _on_state,
                )
                try:
                    current = self.hass.states.get(satellite_entity_id)
                    if current is not None and current.state in _SAT_IDLE_STATES:
                        observed_idle.set()
                except Exception:
                    logger.debug("ha-agenthub: state seed lookup failed", exc_info=True)

            deadline_final = time.monotonic() + PUSH_FINAL_WAIT_SECONDS
            while True:
                remaining = deadline_final - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "ha-agenthub: post-filler push timed out waiting for final frame key=%s sat=%s",
                        gate_key, satellite_entity_id,
                    )
                    if SPEAK_FILLER_ONLY_ON_TIMEOUT and satellite_entity_id:
                        final_text = FILLER_ONLY_TIMEOUT_TEXT
                    break
                try:
                    msg = await asyncio.wait_for(local_ws.receive(), timeout=remaining)
                except asyncio.TimeoutError:
                    continue

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("is_filler", False):
                        logger.info(
                            "ha-agenthub: ignoring secondary filler in push key=%s",
                            gate_key,
                        )
                        continue
                    if data.get("directive"):
                        logger.info(
                            "ha-agenthub: post-filler push received directive, skipping announce key=%s sat=%s",
                            gate_key, satellite_entity_id,
                        )
                        break

                    token_text = data.get("token", "")
                    if token_text:
                        final_parts.append(token_text)
                    if data.get("done", False):
                        mediated = data.get("mediated_speech")
                        if mediated:
                            final_parts = [mediated]
                        stream_sanitized = bool(data.get("sanitized", False))
                        raw = "".join(final_parts)
                        final_text = raw if stream_sanitized else _strip_markdown(raw)
                        final_text = (final_text or "").strip()
                        logger.info(
                            "ha-agenthub: post-filler push received final key=%s sat=%s final_chars=%d",
                            gate_key, satellite_entity_id, len(final_text),
                        )
                        break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning(
                        "ha-agenthub: post-filler push WS closed before final key=%s sat=%s type=%s",
                        gate_key, satellite_entity_id, msg.type,
                    )
                    break

            if final_text is None or not final_text:
                return

            if not satellite_entity_id:
                logger.warning(
                    "ha-agenthub: post-filler push has final but no satellite to announce on key=%s",
                    gate_key,
                )
                return

            if not observed_idle.is_set():
                try:
                    await asyncio.wait_for(
                        observed_idle.wait(),
                        timeout=MAX_POST_FILLER_WAIT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "ha-agenthub: post-filler push satellite never reached idle within %.1fs key=%s sat=%s",
                        MAX_POST_FILLER_WAIT_SECONDS, gate_key, satellite_entity_id,
                    )
                    return

            if aborted_new_turn:
                logger.info(
                    "ha-agenthub: abandoning post-filler push (new turn detected) key=%s sat=%s",
                    gate_key, satellite_entity_id,
                )
                return

            self._push_in_progress_satellites.add(satellite_entity_id)
            try:
                logger.info(
                    "ha-agenthub: post-filler push dispatching announce key=%s sat=%s final_chars=%d",
                    gate_key, satellite_entity_id, len(final_text),
                )
                await self.hass.services.async_call(
                    "assist_satellite",
                    "announce",
                    {
                        "entity_id": satellite_entity_id,
                        "message": final_text,
                        "preannounce": False,
                    },
                    blocking=False,
                )
            except Exception:
                logger.warning(
                    "ha-agenthub: assist_satellite.announce failed in push key=%s sat=%s",
                    gate_key, satellite_entity_id, exc_info=True,
                )
            finally:
                self._push_in_progress_satellites.discard(satellite_entity_id)
        except asyncio.CancelledError:
            logger.info(
                "ha-agenthub: post-filler push cancelled key=%s sat=%s",
                gate_key, satellite_entity_id,
            )
            raise
        except Exception:
            logger.warning(
                "ha-agenthub: post-filler push raised unexpectedly key=%s sat=%s",
                gate_key, satellite_entity_id, exc_info=True,
            )
        finally:
            if unsub is not None:
                try:
                    unsub()
                except Exception:
                    logger.debug("ha-agenthub: state listener unsub raised", exc_info=True)
            try:
                if local_ws is not None and not local_ws.closed:
                    await local_ws.close()
            except Exception:
                logger.debug("ha-agenthub: local_ws close raised", exc_info=True)
            current = self._inflight_pushes.get(key)
            if current is asyncio.current_task():
                self._inflight_pushes.pop(key, None)

    async def _process_via_ws(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult | _BridgeDirective:
        """Send request via WebSocket and accumulate streaming tokens."""
        gate_key = HaAgentHubConversationEntity._filler_gate_key(self, user_input)
        push_enabled = self._is_post_filler_push_enabled()
        payload: dict[str, Any] = {
            "text": user_input.text,
            "conversation_id": user_input.conversation_id,
            "language": user_input.language or "en",
        }
        payload.update(self._resolve_origin_context(user_input))
        if (
            self._is_native_plain_timers_enabled()
            and not _suppress_native_plain_timer_eligibility.get()
        ):
            payload[NATIVE_PLAIN_TIMER_ELIGIBLE_FIELD] = True
        await self._ws.send_json(payload)

        try:
            speech_parts: list[str] = []
            buffered_filler_parts: list[str] = []
            final_conversation_id = user_input.conversation_id

            received_done = False
            # P3-1: track per-stream sanitization. The orchestrator emits
            # token / mediated_speech chunks already stripped by
            # ``app.agents.sanitize.strip_markdown``; the done frame
            # carries the flag explicitly. Default False so legacy
            # backends fall through the local strip pass.
            stream_sanitized = False

            while True:
                msg = await asyncio.wait_for(self._ws.receive(), timeout=30.0)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    if data.get("is_filler", False):
                        filler_text = (data.get("token") or "").strip()
                        if not filler_text:
                            continue
                        stripped_filler = _strip_markdown(filler_text)
                        if not stripped_filler:
                            continue
                        if not push_enabled:
                            buffered_filler_parts.append(stripped_filler)
                            continue

                        satellite = self._resolve_satellite_entity(user_input)
                        local_ws = self._ws
                        self._ws = None
                        logger.info(
                            "ha-agenthub: filler-first return key=%s sat=%s filler_chars=%d",
                            gate_key, satellite, len(stripped_filler),
                        )
                        self._spawn_post_filler_push(
                            local_ws=local_ws,
                            satellite_entity_id=satellite,
                            user_input=user_input,
                            gate_key=gate_key,
                        )
                        self._ws_last_active = time.monotonic()
                        response = intent.IntentResponse(language=user_input.language or "en")
                        response.async_set_speech(stripped_filler)
                        return conversation.ConversationResult(
                            response=response,
                            conversation_id=user_input.conversation_id,
                        )
                        continue

                    token_text = data.get("token", "")
                    if token_text:
                        speech_parts.append(token_text)
                    if data.get("done", False):
                        received_done = True
                        stream_err = data.get("error")
                        final_conversation_id = data.get("conversation_id", final_conversation_id)
                        # 0.25.1: directive on the final frame short-circuits the
                        # bridge response. The integration delegates to native
                        # Assist instead of returning the (empty) speech.
                        directive_value = data.get("directive")
                        if directive_value:
                            self._ws_last_active = time.monotonic()
                            return _BridgeDirective(
                                directive=str(directive_value),
                                reason=data.get("reason"),
                                conversation_id=final_conversation_id,
                            )
                        mediated = data.get("mediated_speech")
                        if mediated:
                            speech_parts = [mediated]
                        # P3-1: backend signals sanitization on the done
                        # frame. Honour it for both ``mediated_speech``
                        # and accumulated tokens (the orchestrator
                        # strips both before emitting).
                        stream_sanitized = bool(data.get("sanitized", False))
                        if stream_err:
                            # Application-level error from the container (done chunk), not a
                            # transport failure — do not raise (would become _WsDroppedAfterSend).
                            logger.warning(
                                "Container reported error in stream done chunk: %s", stream_err
                            )
                            if not "".join(speech_parts).strip():
                                speech_parts = [
                                    "The assistant could not complete that request. "
                                    f"({stream_err})"
                                ]
                        break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    self._ws = None
                    raise aiohttp.ClientError(
                        f"WebSocket {'closed' if msg.type == aiohttp.WSMsgType.CLOSED else 'error'} mid-stream"
                    )

            if not received_done:
                self._ws = None
                raise aiohttp.ClientError("WebSocket stream ended without done token")

            self._ws_last_active = time.monotonic()
            speech = "".join(speech_parts)
            if buffered_filler_parts:
                stripped_final = speech if stream_sanitized else _strip_markdown(speech)
                stripped_final = (stripped_final or "").strip()
                merged_filler = " ".join(part for part in buffered_filler_parts if part).strip()
                if merged_filler and stripped_final:
                    speech = f"{merged_filler}. {stripped_final}"
                elif merged_filler:
                    speech = merged_filler
                else:
                    speech = stripped_final
                return self._build_result(
                    speech,
                    final_conversation_id,
                    user_input.language,
                    sanitized=True,
                )
            return self._build_result(speech, final_conversation_id, user_input.language, sanitized=stream_sanitized)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as err:
            raise _WsDroppedAfterSendError() from err

    async def _process_via_rest(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult | _BridgeDirective:
        """Fallback: send request via REST and get full response."""
        try:
            if self._session is None:
                self._session = aiohttp.ClientSession()
            headers = {"Authorization": f"Bearer {self._api_key}"}
            payload: dict[str, Any] = {
                "text": user_input.text,
                "conversation_id": user_input.conversation_id,
                "language": user_input.language or "en",
            }
            payload.update(self._resolve_origin_context(user_input))
            if (
                self._is_native_plain_timers_enabled()
                and not _suppress_native_plain_timer_eligibility.get()
            ):
                payload[NATIVE_PLAIN_TIMER_ELIGIBLE_FIELD] = True
                headers[NATIVE_PLAIN_TIMER_ELIGIBLE_HEADER] = "1"
            async with self._session.post(
                f"{self._url}/api/conversation",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    return self._build_result(
                        _rest_fallback_error_message(resp.status),
                        user_input.conversation_id,
                        user_input.language,
                    )
                data = await resp.json()
                directive_value = data.get("directive")
                if directive_value:
                    return _BridgeDirective(
                        directive=str(directive_value),
                        reason=data.get("reason"),
                        conversation_id=data.get("conversation_id", user_input.conversation_id),
                    )
                return self._build_result(
                    data.get("speech", ""),
                    data.get("conversation_id", user_input.conversation_id),
                    user_input.language,
                    sanitized=bool(data.get("sanitized", False)),
                )
        except (aiohttp.ClientError, TimeoutError):
            return self._build_result(
                "Sorry, the assistant container is unavailable. Check that the container is running and reachable from Home Assistant.",
                user_input.conversation_id,
                user_input.language,
            )

    def _build_result(self, speech: str, conversation_id: str | None, language: str | None, *, sanitized: bool = False) -> conversation.ConversationResult:
        """Assemble a ConversationResult from the response.

        P3-1: ``sanitized`` indicates that the backend already stripped
        Markdown for TTS. When True we trust the backend (single source
        of truth) and skip the local ``_strip_markdown`` pass. Older
        backends that do not advertise the flag default to False so the
        defensive fallback still runs.
        """
        response = intent.IntentResponse(language=language or "en")
        response.async_set_speech(speech if sanitized else _strip_markdown(speech))
        return conversation.ConversationResult(response=response, conversation_id=conversation_id)

    def _resolve_satellite_entity(self, user_input) -> str | None:
        """Resolve the originating assist_satellite entity from device or area context."""
        try:
            entity_reg = er.async_get(self.hass)
            device_reg = dr.async_get(self.hass)

            device_id = getattr(user_input, "device_id", None)
            area_id = getattr(user_input, "area_id", None)
            if isinstance(device_id, str) and device_id:
                for entry in entity_reg.entities.values():
                    if entry.domain == "assist_satellite" and entry.device_id == device_id:
                        return entry.entity_id
                device = device_reg.async_get(device_id)
                if device and device.area_id:
                    area_id = device.area_id

            if not isinstance(area_id, str) or not area_id:
                return None

            for entry in entity_reg.entities.values():
                if entry.domain != "assist_satellite" or not entry.device_id:
                    continue
                sat_device = device_reg.async_get(entry.device_id)
                if sat_device and sat_device.area_id == area_id:
                    return entry.entity_id
        except Exception:
            logger.debug("Failed to resolve assist satellite entity", exc_info=True)
        return None
