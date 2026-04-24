"""Timer-specific action execution via HA timer and input_datetime services."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from app.agents.action_executor import (
    build_verified_speech,
    call_service_with_verification,
    filter_matches_by_domain,
)
from app.agents.delayed_tasks import delayed_task_manager
from app.analytics.tracer import _optional_span

logger = logging.getLogger(__name__)

_TIMER_ACTION_MAP: dict[str, tuple[str, str]] = {
    "start_timer": ("timer", "start"),
    "cancel_timer": ("timer", "cancel"),
    "pause_timer": ("timer", "pause"),
    "resume_timer": ("timer", "start"),
    "finish_timer": ("timer", "finish"),
    "set_datetime": ("input_datetime", "set_datetime"),
}

# FLOW-VERIFY-SHARED (0.18.5): HA timer entities expose
# "active"/"paused"/"idle". input_datetime has no fixed target state
# (the "state" is the datetime itself), so we leave it open.
_EXPECTED_STATE_BY_ACTION: dict[str, str] = {
    "start_timer": "active",
    "cancel_timer": "idle",
    "pause_timer": "paused",
    "resume_timer": "active",
    "finish_timer": "idle",
}

_ACTION_PHRASES: dict[str, str] = {
    "start_timer": "started",
    "cancel_timer": "cancelled",
    "pause_timer": "paused",
    "resume_timer": "resumed",
    "finish_timer": "finished",
    "set_datetime": "updated",
}

_ALLOWED_DOMAINS: frozenset[str] = frozenset({"timer", "input_datetime"})

# FLOW-DOMAIN-1 (0.19.2): per-action HA-domain allow-set used to filter
# the hybrid matcher before picking matches[0]. Reminder paths target
# calendar.* which is intentionally outside _ALLOWED_DOMAINS (those
# paths bypass _validate_domain entirely).
_ACTION_DOMAINS: dict[str, frozenset[str]] = {
    "start_timer": frozenset({"timer"}),
    "cancel_timer": frozenset({"timer"}),
    "pause_timer": frozenset({"timer"}),
    "resume_timer": frozenset({"timer"}),
    "finish_timer": frozenset({"timer"}),
    "snooze_timer": frozenset({"timer"}),
    "start_timer_with_notification": frozenset({"timer"}),
    "sleep_timer": frozenset({"timer"}),
    "set_datetime": frozenset({"input_datetime"}),
}
_TIMER_QUERY_DOMAINS: frozenset[str] = frozenset({"timer", "input_datetime"})
_CALENDAR_DOMAINS: frozenset[str] = frozenset({"calendar"})


def _validate_domain(entity_id: str) -> bool:
    """Check that entity_id belongs to an allowed domain for this executor."""
    domain = entity_id.split(".")[0] if "." in entity_id else ""
    return domain in _ALLOWED_DOMAINS


# ---------------------------------------------------------------------------
# Timer Metadata
# ---------------------------------------------------------------------------


@dataclass
class TimerMetadata:
    name: str
    entity_id: str
    duration: str | None = None
    origin_device_id: str | None = None
    origin_area: str | None = None
    media_player_entity: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    notification_type: str = "standard"


# ---------------------------------------------------------------------------
# Recently Expired Tracking
# ---------------------------------------------------------------------------


@dataclass
class ExpiredTimer:
    name: str
    entity_id: str
    expired_at: datetime
    metadata: TimerMetadata | None = None


_recently_expired: deque[ExpiredTimer] = deque(maxlen=20)
_EXPIRED_TTL = timedelta(minutes=10)


def record_expired(name: str, entity_id: str, metadata: TimerMetadata | None = None) -> None:
    """Record a timer as recently expired."""
    _recently_expired.appendleft(
        ExpiredTimer(
            name=name,
            entity_id=entity_id,
            expired_at=datetime.now(),
            metadata=metadata,
        )
    )


def get_last_expired() -> ExpiredTimer | None:
    """Get the most recently expired timer (within TTL)."""
    now = datetime.now()
    for exp in _recently_expired:
        if now - exp.expired_at <= _EXPIRED_TTL:
            return exp
    return None


def get_recently_expired() -> list[ExpiredTimer]:
    """Get all recently expired timers within TTL."""
    now = datetime.now()
    return [e for e in _recently_expired if now - e.expired_at <= _EXPIRED_TTL]


# ---------------------------------------------------------------------------
# Timer Pool
# ---------------------------------------------------------------------------


class _TimerPool:
    """In-memory mapping of user-given names to timer entity IDs.

    When a user starts a timer with a descriptive name (e.g., "egg timer")
    that does not match an existing entity, the pool assigns an idle
    timer entity from the available pool. When the timer finishes or is
    cancelled, the mapping is released.
    """

    def __init__(self) -> None:
        self._name_to_entity: dict[str, str] = {}  # "egg timer" -> "timer.pool_1"
        self._entity_to_name: dict[str, str] = {}  # "timer.pool_1" -> "egg timer"
        self._entity_to_metadata: dict[str, TimerMetadata] = {}

    def get_entity(self, name: str) -> str | None:
        """Look up entity_id by user-given timer name."""
        return self._name_to_entity.get(name.lower().strip())

    def assign(self, name: str, entity_id: str, metadata: TimerMetadata | None = None) -> None:
        """Assign a name mapping to an entity."""
        key = name.lower().strip()
        self._name_to_entity[key] = entity_id
        self._entity_to_name[entity_id] = key
        if metadata:
            self._entity_to_metadata[entity_id] = metadata

    def release(self, entity_id: str) -> None:
        """Release a timer entity back to the pool."""
        name = self._entity_to_name.pop(entity_id, None)
        if name:
            self._name_to_entity.pop(name, None)
        self._entity_to_metadata.pop(entity_id, None)

    def get_name(self, entity_id: str) -> str | None:
        """Get the user-given name for an entity, if any."""
        return self._entity_to_name.get(entity_id)

    def get_metadata(self, entity_id: str) -> TimerMetadata | None:
        """Get the metadata for an entity, if any."""
        return self._entity_to_metadata.get(entity_id)

    def all_mappings(self) -> dict[str, str]:
        """Return a copy of all current name->entity mappings."""
        return dict(self._name_to_entity)

    def all_metadata(self) -> dict[str, TimerMetadata]:
        """Return a copy of all current entity->metadata mappings."""
        return dict(self._entity_to_metadata)


# Module-level singleton pool instance
_timer_pool = _TimerPool()


async def _find_idle_timer(ha_client: Any) -> str | None:
    """Find the first idle timer entity not currently assigned in the pool."""
    try:
        states = await ha_client.get_states()
    except Exception:
        logger.warning("Failed to fetch states for pool allocation", exc_info=True)
        return None

    for s in states:
        entity_id = s.get("entity_id", "")
        if not entity_id.startswith("timer."):
            continue
        if s.get("state") == "idle" and entity_id not in _timer_pool._entity_to_name:
            return entity_id
    return None


async def _resolve_media_player(ha_client: Any, device_id: str | None, area_id: str | None) -> str | None:
    """Find a media_player entity in the same area as the originating device."""
    if not area_id:
        return None
    try:
        states = await ha_client.get_states()
        for s in states:
            eid = s.get("entity_id", "")
            if eid.startswith("media_player.") and s.get("attributes", {}).get("area_id") == area_id:
                return eid
    except Exception:
        logger.warning("Failed to resolve media_player for area %s", area_id, exc_info=True)
    return None


async def on_timer_finished(
    entity_id: str,
    ha_client: Any,
    entity_index: Any = None,
) -> None:
    """Handle timer.finished WebSocket event -- dispatch notifications.

    ``entity_index`` is forwarded to the dispatcher so the follow-up
    voice pipeline can target the correct assist_satellite via area
    (FLOW-HIGH-6).
    """
    metadata = _timer_pool.get_metadata(entity_id)
    timer_name = _timer_pool.get_name(entity_id) or entity_id

    logger.info("Timer finished: %s (name=%s)", entity_id, timer_name)

    record_expired(timer_name, entity_id, metadata)

    try:
        from app.agents.notification_dispatcher import dispatch_timer_notification

        await dispatch_timer_notification(
            ha_client=ha_client,
            timer_name=timer_name,
            entity_id=entity_id,
            metadata=metadata,
            entity_index=entity_index,
        )
    except Exception:
        logger.error("Notification dispatch failed for %s", entity_id, exc_info=True)

    # Release pool mapping
    _timer_pool.release(entity_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_timer_service_data(action: dict) -> dict[str, Any]:
    """Build HA service_data from a timer action's parameters."""
    params = action.get("parameters") or {}
    data: dict[str, Any] = {}

    if "duration" in params:
        data["duration"] = str(params["duration"])
    if "datetime" in params:
        data["datetime"] = str(params["datetime"])
    if "time" in params:
        data["time"] = str(params["time"])
    if "date" in params:
        data["date"] = str(params["date"])

    return data


def _parse_duration_seconds(duration_str: str) -> int | None:
    """Parse HH:MM:SS or MM:SS or seconds string into total seconds."""
    if not duration_str:
        return None
    parts = str(duration_str).split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return int(float(parts[0]))
    except (ValueError, IndexError):
        return None


def _format_duration_human(total_seconds: int) -> str:
    """Format seconds into human-readable duration string."""
    if total_seconds <= 0:
        return "0 seconds"
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    return " and ".join(parts) if len(parts) <= 2 else ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _format_elapsed(dt: datetime) -> str:
    """Format elapsed time since a datetime as a short human-readable string."""
    delta = datetime.now() - dt
    total_secs = int(delta.total_seconds())
    if total_secs < 60:
        return f"{total_secs}s"
    minutes = total_secs // 60
    return f"{minutes} min"


def _format_timer_state(entity_id: str, state_resp: dict) -> str:
    """Format a timer or input_datetime entity state as human-readable text."""
    state = state_resp.get("state", "unknown")
    attrs = state_resp.get("attributes", {})
    friendly_name = attrs.get("friendly_name", entity_id)

    if entity_id.startswith("timer."):
        if state == "active":
            remaining = attrs.get("remaining")
            secs = _parse_duration_seconds(remaining)
            if secs is not None:
                return f"{friendly_name} is active with {_format_duration_human(secs)} remaining."
            return f"{friendly_name} is active."
        if state == "paused":
            remaining = attrs.get("remaining")
            secs = _parse_duration_seconds(remaining)
            if secs is not None:
                return f"{friendly_name} is paused with {_format_duration_human(secs)} remaining."
            return f"{friendly_name} is paused."
        if state == "idle":
            return f"{friendly_name} is idle (not running)."
        return f"{friendly_name} is {state}."

    if entity_id.startswith("input_datetime."):
        has_date = attrs.get("has_date", False)
        has_time = attrs.get("has_time", False)
        if has_date and has_time:
            return f"{friendly_name} is set to {state}."
        if has_time:
            return f"{friendly_name} is set to {state}."
        if has_date:
            return f"{friendly_name} is set to {state}."
        return f"{friendly_name}: {state}."

    return f"{friendly_name}: {state}."


def _build_unverified_timer_speech(
    *,
    friendly_name: str,
    expected_state: str,
    observed_state: str | None,
) -> str:
    """Build non-optimistic speech for unverified timer state changes."""
    if observed_state:
        return (
            f"I could not verify {friendly_name}. "
            f"Home Assistant reported {observed_state} instead of {expected_state}."
        )
    return f"I could not verify that {friendly_name} reached {expected_state}."


# ---------------------------------------------------------------------------
# Read-only action handlers
# ---------------------------------------------------------------------------


async def _handle_read_action(
    action_name: str,
    entity_query: str,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
    span_collector=None,
) -> dict:
    """Handle read-only actions that query state without calling a service."""

    if action_name == "query_timer":
        return await _query_timer(
            entity_query, ha_client, entity_index, entity_matcher, agent_id, span_collector=span_collector
        )
    if action_name == "list_timers":
        return await _list_timers(ha_client)
    if action_name == "list_alarms":
        return await _list_alarms(ha_client)

    return {"success": False, "entity_id": "", "new_state": None, "speech": f"Unknown read action: {action_name}"}


async def _query_timer(
    entity_query: str,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
    span_collector=None,
) -> dict:
    """Query the status of a specific timer or input_datetime entity."""
    entity_id = None

    # Check pool mapping first
    pool_entity = _timer_pool.get_entity(entity_query)
    if pool_entity:
        entity_id = pool_entity

    if not entity_id:
        try:
            if entity_matcher:
                async with _optional_span(span_collector, "entity_match", agent_id=agent_id) as em_span:
                    matches = await entity_matcher.match(entity_query, agent_id=agent_id)
                    em_span["metadata"] = {"query": entity_query, "match_count": len(matches)}
                    # FLOW-DOMAIN-1 (0.19.2): query spans timer + input_datetime.
                    filtered = filter_matches_by_domain(matches, _TIMER_QUERY_DOMAINS)
                    if filtered:
                        entity_id = filtered[0].entity_id
                        em_span["metadata"]["top_entity_id"] = entity_id
                        em_span["metadata"]["top_score"] = filtered[0].score
                        em_span["metadata"]["signal_scores"] = getattr(filtered[0], "signal_scores", {})
        except Exception:
            logger.warning("Entity resolution failed for '%s'", entity_query, exc_info=True)

    if entity_id and not _validate_domain(entity_id):
        logger.warning("Resolved entity %s not in allowed domains %s", entity_id, _ALLOWED_DOMAINS)
        entity_id = None

    if not entity_id:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": f"Could not find an entity matching '{entity_query}'.",
            "cacheable": False,
        }

    try:
        state_resp = await ha_client.get_state(entity_id)
        if not state_resp:
            return {
                "success": False,
                "entity_id": entity_id,
                "new_state": None,
                "speech": f"Could not retrieve state for {entity_id}.",
                "cacheable": False,
            }
        speech = _format_timer_state(entity_id, state_resp)
        return {
            "success": True,
            "entity_id": entity_id,
            "new_state": state_resp.get("state"),
            "speech": speech,
            "cacheable": False,
        }
    except Exception as exc:
        logger.error("State query failed for %s", entity_id, exc_info=True)
        return {
            "success": False,
            "entity_id": entity_id,
            "new_state": None,
            "speech": f"Failed to query timer status: {exc}",
            "cacheable": False,
        }


async def _list_timers(ha_client: Any) -> dict:
    """List all timer entities with their current state and remaining time."""
    try:
        states = await ha_client.get_states()
    except Exception as exc:
        logger.error("Failed to fetch states for list_timers", exc_info=True)
        return {"success": False, "entity_id": "", "new_state": None, "speech": f"Failed to list timers: {exc}"}

    timers = [s for s in states if s.get("entity_id", "").startswith("timer.")]

    if not timers:
        return {"success": True, "entity_id": "", "new_state": None, "speech": "No timer entities found."}

    active = []
    paused = []
    idle = []
    for t in timers:
        entity_id = t.get("entity_id", "")
        state = t.get("state", "unknown")
        attrs = t.get("attributes", {})
        friendly_name = attrs.get("friendly_name", entity_id)
        pool_name = _timer_pool.get_name(entity_id)
        if pool_name:
            friendly_name = f"{pool_name} ({friendly_name})"
        remaining = attrs.get("remaining")

        if state == "active":
            secs = _parse_duration_seconds(remaining)
            time_str = f" ({_format_duration_human(secs)} remaining)" if secs is not None else ""
            active.append(f"{friendly_name}{time_str}")
        elif state == "paused":
            secs = _parse_duration_seconds(remaining)
            time_str = f" (paused, {_format_duration_human(secs)} remaining)" if secs is not None else " (paused)"
            paused.append(f"{friendly_name}{time_str}")
        else:
            idle.append(friendly_name)

    parts = []
    if active:
        parts.append(f"Active: {', '.join(active)}")
    if paused:
        parts.append(f"Paused: {', '.join(paused)}")
    if idle:
        parts.append(f"Idle: {', '.join(idle)}")

    expired = get_recently_expired()
    if expired:
        expired_names = [f"{e.name} (finished {_format_elapsed(e.expired_at)} ago)" for e in expired]
        parts.append(f"Recently finished: {', '.join(expired_names)}")

    if not active and not paused:
        speech = "No timers are currently running."
        if idle:
            speech += f" {len(idle)} idle timer{'s' if len(idle) != 1 else ''} available."
        if expired:
            speech += f" {'. '.join(parts[-1:])}."
    else:
        speech = ". ".join(parts) + "."

    return {"success": True, "entity_id": "", "new_state": None, "speech": speech, "cacheable": False}


async def _list_alarms(ha_client: Any) -> dict:
    """List all input_datetime entities (alarms/reminders)."""
    try:
        states = await ha_client.get_states()
    except Exception as exc:
        logger.error("Failed to fetch states for list_alarms", exc_info=True)
        return {"success": False, "entity_id": "", "new_state": None, "speech": f"Failed to list alarms: {exc}"}

    alarms = [s for s in states if s.get("entity_id", "").startswith("input_datetime.")]

    if not alarms:
        return {
            "success": True,
            "entity_id": "",
            "new_state": None,
            "speech": "No alarm or input_datetime entities found.",
        }

    lines = []
    for a in alarms:
        state = a.get("state", "unknown")
        attrs = a.get("attributes", {})
        friendly_name = attrs.get("friendly_name", a.get("entity_id", ""))
        lines.append(f"{friendly_name}: {state}")

    speech = "Alarms: " + "; ".join(lines) + "."
    return {"success": True, "entity_id": "", "new_state": None, "speech": speech, "cacheable": False}


# ---------------------------------------------------------------------------
# Multi-step action handlers
# ---------------------------------------------------------------------------

_DEFAULT_SNOOZE_DURATION = "00:05:00"


async def _snooze_timer(
    action: dict,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
    span_collector=None,
) -> dict:
    """Snooze a timer by cancelling it and restarting with a snooze duration."""
    entity_query = action.get("entity", "")
    params = action.get("parameters") or {}
    snooze_duration = str(params.get("duration", _DEFAULT_SNOOZE_DURATION))

    # Resolve entity
    entity_id = None
    friendly_name = entity_query
    try:
        if entity_matcher:
            async with _optional_span(span_collector, "entity_match", agent_id=agent_id) as em_span:
                matches = await entity_matcher.match(entity_query, agent_id=agent_id)
                em_span["metadata"] = {"query": entity_query, "match_count": len(matches)}
                # FLOW-DOMAIN-1 (0.19.2): snooze targets timer.* only.
                filtered = filter_matches_by_domain(matches, _ACTION_DOMAINS["snooze_timer"])
                if filtered:
                    entity_id = filtered[0].entity_id
                    friendly_name = filtered[0].friendly_name or entity_id
                    em_span["metadata"]["top_entity_id"] = entity_id
                    em_span["metadata"]["top_friendly_name"] = friendly_name
                    em_span["metadata"]["top_score"] = filtered[0].score
                    em_span["metadata"]["signal_scores"] = getattr(filtered[0], "signal_scores", {})
    except Exception:
        logger.warning("Entity resolution failed for '%s'", entity_query, exc_info=True)

    if entity_id and not _validate_domain(entity_id):
        logger.warning("Resolved entity %s not in allowed domains %s", entity_id, _ALLOWED_DOMAINS)
        entity_id = None

    if not entity_id:
        # Try last expired timer for snooze
        last = get_last_expired()
        if last:
            entity_id = last.entity_id
            friendly_name = last.name

    if not entity_id:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": f"Could not find an entity matching '{entity_query}'.",
        }

    try:
        # Check current state -- if idle (recently expired), skip cancel
        state_resp = await ha_client.get_state(entity_id)
        current_state = state_resp.get("state") if state_resp else None
        if current_state and current_state != "idle":
            # FLOW-VERIFY-SHARED: verified cancel -> idle
            await call_service_with_verification(
                ha_client,
                "timer",
                "cancel",
                entity_id,
                expected_state="idle",
            )

        # Restart with snooze duration (verified -> active)
        verify = await call_service_with_verification(
            ha_client,
            "timer",
            "start",
            entity_id,
            service_data={"duration": snooze_duration},
            expected_state="active",
        )

        # Re-register in pool if it was expired
        if not _timer_pool.get_name(entity_id):
            _timer_pool.assign(friendly_name, entity_id)

        new_state = verify.get("observed_state") if verify.get("success") else None

        secs = _parse_duration_seconds(snooze_duration)
        human_dur = _format_duration_human(secs) if secs else snooze_duration
        return {
            "success": True,
            "entity_id": entity_id,
            "new_state": new_state,
            "speech": f"Snoozed {friendly_name} for {human_dur}.",
        }
    except Exception as exc:
        logger.error("Snooze failed for %s", entity_id, exc_info=True)
        return {
            "success": False,
            "entity_id": entity_id,
            "new_state": None,
            "speech": f"Failed to snooze {friendly_name}: {exc}",
        }


# ---------------------------------------------------------------------------
# Delayed / notification action handlers
# ---------------------------------------------------------------------------


async def _start_timer_with_notification(
    action: dict,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
    span_collector=None,
) -> dict:
    """Start a timer and schedule a notification for when it finishes."""
    entity_query = action.get("entity", "")
    params = action.get("parameters") or {}
    duration = str(params.get("duration", ""))
    notification_message = str(params.get("notification_message", "Timer finished!"))

    if not duration:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Duration is required for start_timer_with_notification.",
        }

    # Resolve entity (reuse pool logic through regular start)
    start_action = {"action": "start_timer", "entity": entity_query, "parameters": {"duration": duration}}
    result = await execute_timer_action(
        start_action, ha_client, entity_index, entity_matcher, agent_id=agent_id, span_collector=span_collector
    )

    if not result.get("success"):
        return result

    entity_id = result.get("entity_id")
    if not entity_id:
        return result

    # Schedule notification via delayed task manager (fallback if WS event missed)
    async def _send_notification():
        try:
            from app.agents.notification_dispatcher import dispatch_timer_notification

            timer_name = _timer_pool.get_name(entity_id) or entity_query
            metadata = _timer_pool.get_metadata(entity_id)
            await dispatch_timer_notification(
                ha_client=ha_client,
                timer_name=timer_name,
                entity_id=entity_id,
                metadata=metadata,
                entity_index=entity_index,
            )
        except Exception:
            logger.error("Failed to send timer notification", exc_info=True)

    delayed_task_manager.set_ha_client(ha_client)
    delayed_task_manager.schedule(entity_id, f"notify: {notification_message}", _send_notification)

    secs = _parse_duration_seconds(duration)
    human_dur = _format_duration_human(secs) if secs else duration
    return {
        "success": True,
        "entity_id": entity_id,
        "new_state": result.get("new_state"),
        "speech": f'Started timer for {human_dur} with notification: "{notification_message}".',
    }


async def _delayed_action(
    action: dict,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
) -> dict:
    """Start a timer and execute a service call when it finishes.

    Parameters in the action:
        timer_entity: which timer to use (optional, will use pool if not matched)
        delay_duration: HH:MM:SS delay
        target_entity: entity to act on after delay
        target_action: HA service to call (e.g. "light/turn_off", "switch/turn_off")
    """
    params = action.get("parameters") or {}
    delay_duration = str(params.get("delay_duration", ""))
    target_entity = str(params.get("target_entity", ""))
    target_action = str(params.get("target_action", ""))

    if not delay_duration:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "delay_duration is required for delayed_action.",
        }
    if not target_entity:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "target_entity is required for delayed_action.",
        }
    if not target_action or "/" not in target_action:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "target_action is required in 'domain/service' format (e.g. 'light/turn_off').",
        }

    target_domain, target_service = target_action.split("/", 1)

    # Start a timer for the delay
    timer_entity_query = action.get("entity", "delay timer")
    start_action = {"action": "start_timer", "entity": timer_entity_query, "parameters": {"duration": delay_duration}}
    result = await execute_timer_action(start_action, ha_client, entity_index, entity_matcher, agent_id=agent_id)

    if not result.get("success"):
        return result

    timer_entity_id = result.get("entity_id")
    if not timer_entity_id:
        return result

    # Schedule the delayed service call
    async def _execute_delayed():
        try:
            await ha_client.call_service(target_domain, target_service, target_entity)
        except Exception:
            logger.error(
                "Delayed action failed: %s/%s on %s", target_domain, target_service, target_entity, exc_info=True
            )

    delayed_task_manager.set_ha_client(ha_client)
    delayed_task_manager.schedule(
        timer_entity_id,
        f"{target_action} on {target_entity}",
        _execute_delayed,
    )

    secs = _parse_duration_seconds(delay_duration)
    human_dur = _format_duration_human(secs) if secs else delay_duration
    return {
        "success": True,
        "entity_id": timer_entity_id,
        "new_state": result.get("new_state"),
        "speech": f"Scheduled {target_action.replace('/', ' ')} on {target_entity} in {human_dur}.",
    }


async def _sleep_timer(
    action: dict,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
    span_collector=None,
) -> dict:
    """Set a sleep timer that stops a media player after a duration.

    Parameters:
        duration: HH:MM:SS for how long media should keep playing
        media_player: entity_id of the media player to stop (e.g. "media_player.bedroom")
    """
    params = action.get("parameters") or {}
    duration = str(params.get("duration", ""))
    media_player_entity = str(params.get("media_player", ""))

    if not duration:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Duration is required for sleep_timer.",
        }
    if not media_player_entity:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "media_player entity_id is required for sleep_timer.",
        }

    # Start a timer for the duration
    timer_entity_query = action.get("entity", "sleep timer")
    start_action = {"action": "start_timer", "entity": timer_entity_query, "parameters": {"duration": duration}}
    result = await execute_timer_action(
        start_action, ha_client, entity_index, entity_matcher, agent_id=agent_id, span_collector=span_collector
    )

    if not result.get("success"):
        return result

    timer_entity_id = result.get("entity_id")
    if not timer_entity_id:
        return result

    # Schedule media stop
    async def _stop_media():
        try:
            await ha_client.call_service("media_player", "media_stop", media_player_entity)
        except Exception:
            logger.error("Sleep timer: failed to stop media on %s", media_player_entity, exc_info=True)

    delayed_task_manager.set_ha_client(ha_client)
    delayed_task_manager.schedule(
        timer_entity_id,
        f"media_player/media_stop on {media_player_entity}",
        _stop_media,
    )

    secs = _parse_duration_seconds(duration)
    human_dur = _format_duration_human(secs) if secs else duration
    return {
        "success": True,
        "entity_id": timer_entity_id,
        "new_state": result.get("new_state"),
        "speech": f"Sleep timer set for {human_dur}. Media on {media_player_entity} will stop when the timer ends.",
    }


# ---------------------------------------------------------------------------
# Calendar action handlers
# ---------------------------------------------------------------------------


async def _create_reminder(
    action: dict,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
    span_collector=None,
) -> dict:
    """Create a calendar event as a reminder using calendar.create_event."""
    entity_query = action.get("entity", "")
    params = action.get("parameters") or {}
    summary = str(params.get("summary", ""))
    start_time = str(params.get("start_date_time", ""))
    end_time = str(params.get("end_date_time", ""))
    description = str(params.get("description", ""))

    if not summary:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Summary is required for create_reminder.",
        }
    if not start_time:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "start_date_time is required for create_reminder.",
        }

    # Resolve calendar entity
    entity_id = None
    friendly_name = entity_query
    try:
        if entity_matcher:
            async with _optional_span(span_collector, "entity_match", agent_id=agent_id) as em_span:
                matches = await entity_matcher.match(entity_query, agent_id=agent_id)
                em_span["metadata"] = {"query": entity_query, "match_count": len(matches)}
                # FLOW-DOMAIN-1 (0.19.2): reminders target calendar.* only.
                filtered = filter_matches_by_domain(matches, _CALENDAR_DOMAINS)
                if filtered:
                    entity_id = filtered[0].entity_id
                    friendly_name = filtered[0].friendly_name or entity_id
                    em_span["metadata"]["top_entity_id"] = entity_id
                    em_span["metadata"]["top_friendly_name"] = friendly_name
                    em_span["metadata"]["top_score"] = filtered[0].score
                    em_span["metadata"]["signal_scores"] = getattr(filtered[0], "signal_scores", {})
        if not entity_id and entity_index:
            results = await entity_index.search_async(entity_query, n_results=1)
            if results:
                entity_id = results[0][0].entity_id
                friendly_name = results[0][0].friendly_name or entity_id
    except Exception:
        logger.warning("Entity resolution failed for '%s'", entity_query, exc_info=True)

    if not entity_id:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": f"Could not find a calendar entity matching '{entity_query}'.",
        }

    service_data: dict[str, str] = {"summary": summary, "start_date_time": start_time}
    if end_time:
        service_data["end_date_time"] = end_time
    else:
        service_data["end_date_time"] = start_time
    if description:
        service_data["description"] = description

    try:
        await ha_client.call_service("calendar", "create_event", entity_id, service_data)
    except Exception as exc:
        logger.error("Failed to create calendar event on %s", entity_id, exc_info=True)
        return {
            "success": False,
            "entity_id": entity_id,
            "new_state": None,
            "speech": f"Failed to create reminder: {exc}",
        }

    return {
        "success": True,
        "entity_id": entity_id,
        "new_state": None,
        "speech": f'Created reminder "{summary}" at {start_time} on {friendly_name}.',
    }


async def _create_recurring_reminder(
    action: dict,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
    span_collector=None,
) -> dict:
    """Create a recurring calendar event using calendar.create_event with rrule."""
    entity_query = action.get("entity", "")
    params = action.get("parameters") or {}
    summary = str(params.get("summary", ""))
    start_time = str(params.get("start_date_time", ""))
    end_time = str(params.get("end_date_time", ""))
    rrule = str(params.get("rrule", ""))

    if not summary:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Summary is required for create_recurring_reminder.",
        }
    if not start_time:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "start_date_time is required for create_recurring_reminder.",
        }
    if not rrule:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "rrule is required for create_recurring_reminder (e.g. 'FREQ=DAILY', 'FREQ=WEEKLY;BYDAY=MO,WE,FR').",
        }

    # Resolve calendar entity
    entity_id = None
    friendly_name = entity_query
    try:
        if entity_matcher:
            async with _optional_span(span_collector, "entity_match", agent_id=agent_id) as em_span:
                matches = await entity_matcher.match(entity_query, agent_id=agent_id)
                em_span["metadata"] = {"query": entity_query, "match_count": len(matches)}
                # FLOW-DOMAIN-1 (0.19.2): recurring reminders target calendar.* only.
                filtered = filter_matches_by_domain(matches, _CALENDAR_DOMAINS)
                if filtered:
                    entity_id = filtered[0].entity_id
                    friendly_name = filtered[0].friendly_name or entity_id
                    em_span["metadata"]["top_entity_id"] = entity_id
                    em_span["metadata"]["top_friendly_name"] = friendly_name
                    em_span["metadata"]["top_score"] = filtered[0].score
                    em_span["metadata"]["signal_scores"] = getattr(filtered[0], "signal_scores", {})
        if not entity_id and entity_index:
            results = await entity_index.search_async(entity_query, n_results=1)
            if results:
                entity_id = results[0][0].entity_id
                friendly_name = results[0][0].friendly_name or entity_id
    except Exception:
        logger.warning("Entity resolution failed for '%s'", entity_query, exc_info=True)

    if not entity_id:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": f"Could not find a calendar entity matching '{entity_query}'.",
        }

    service_data: dict[str, str] = {
        "summary": summary,
        "start_date_time": start_time,
        "rrule": rrule,
    }
    if end_time:
        service_data["end_date_time"] = end_time
    else:
        service_data["end_date_time"] = start_time

    try:
        await ha_client.call_service("calendar", "create_event", entity_id, service_data)
    except Exception as exc:
        logger.error("Failed to create recurring event on %s", entity_id, exc_info=True)
        return {
            "success": False,
            "entity_id": entity_id,
            "new_state": None,
            "speech": f"Failed to create recurring reminder: {exc}",
        }

    # Human-friendly rrule description
    freq_map = {"DAILY": "daily", "WEEKLY": "weekly", "MONTHLY": "monthly", "YEARLY": "yearly"}
    freq = "recurring"
    for key, val in freq_map.items():
        if key in rrule.upper():
            freq = val
            break

    return {
        "success": True,
        "entity_id": entity_id,
        "new_state": None,
        "speech": f'Created {freq} reminder "{summary}" at {start_time} on {friendly_name}.',
    }


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------


async def execute_timer_action(
    action: dict,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None = None,
    device_id: str | None = None,
    area_id: str | None = None,
    span_collector=None,
) -> dict:
    """Resolve an entity, call a timer HA service, and verify the result.

    Args:
        action: Parsed action dict with "action", "entity", and optional "parameters".
        ha_client: HARestClient instance.
        entity_index: EntityIndex instance.
        entity_matcher: EntityMatcher instance.
        agent_id: Optional agent identifier for entity matching context.

    Returns:
        dict with "success", "entity_id", "new_state", and "speech".
    """
    action_name = action.get("action", "").lower()
    entity_query = action.get("entity", "")

    # Read-only actions (no service call)
    if action_name in ("query_timer", "list_timers", "list_alarms"):
        return await _handle_read_action(
            action_name, entity_query, ha_client, entity_index, entity_matcher, agent_id, span_collector=span_collector
        )

    # Multi-step actions (custom logic, not simple service calls)
    if action_name == "snooze_timer":
        return await _snooze_timer(
            action, ha_client, entity_index, entity_matcher, agent_id, span_collector=span_collector
        )
    if action_name == "start_timer_with_notification":
        return await _start_timer_with_notification(
            action, ha_client, entity_index, entity_matcher, agent_id, span_collector=span_collector
        )
    if action_name == "delayed_action":
        return await _delayed_action(action, ha_client, entity_index, entity_matcher, agent_id)
    if action_name == "sleep_timer":
        return await _sleep_timer(
            action, ha_client, entity_index, entity_matcher, agent_id, span_collector=span_collector
        )
    if action_name == "create_reminder":
        return await _create_reminder(
            action, ha_client, entity_index, entity_matcher, agent_id, span_collector=span_collector
        )
    if action_name == "create_recurring_reminder":
        return await _create_recurring_reminder(
            action, ha_client, entity_index, entity_matcher, agent_id, span_collector=span_collector
        )

    # Validate action name for simple service-call actions
    mapping = _TIMER_ACTION_MAP.get(action_name)
    if not mapping:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": f"Unknown action: {action_name}",
        }

    domain, service = mapping

    # Resolve entity via pool name first, then matcher, then entity_index
    entity_id = None
    friendly_name = entity_query

    # Check timer pool for named timer mapping
    pool_entity = _timer_pool.get_entity(entity_query)
    if pool_entity:
        entity_id = pool_entity
        friendly_name = entity_query

    if not entity_id:
        try:
            if entity_matcher:
                async with _optional_span(span_collector, "entity_match", agent_id=agent_id) as em_span:
                    matches = await entity_matcher.match(entity_query, agent_id=agent_id)
                    em_span["metadata"] = {"query": entity_query, "match_count": len(matches)}
                    # FLOW-DOMAIN-1 (0.19.2): per-action allow-set; falls back
                    # to the broad timer/input_datetime pool for unmapped actions.
                    required_domains = _ACTION_DOMAINS.get(action_name, _TIMER_QUERY_DOMAINS)
                    filtered = filter_matches_by_domain(matches, required_domains)
                    if len(filtered) != len(matches):
                        em_span["metadata"]["domain_filter_dropped"] = len(matches) - len(filtered)
                        em_span["metadata"]["domain_filter_allowed"] = sorted(required_domains)
                    if filtered:
                        entity_id = filtered[0].entity_id
                        friendly_name = filtered[0].friendly_name or entity_id
                        em_span["metadata"]["top_entity_id"] = entity_id
                        em_span["metadata"]["top_friendly_name"] = friendly_name
                        em_span["metadata"]["top_score"] = filtered[0].score
                        em_span["metadata"]["signal_scores"] = getattr(filtered[0], "signal_scores", {})
        except Exception:
            logger.warning("Entity resolution failed for '%s'", entity_query, exc_info=True)

    if entity_id and not _validate_domain(entity_id):
        logger.warning("Resolved entity %s not in allowed domains %s", entity_id, _ALLOWED_DOMAINS)
        entity_id = None

    # For start_timer: if no entity found, try to allocate from pool
    if not entity_id and action_name == "start_timer":
        idle_entity = await _find_idle_timer(ha_client)
        if idle_entity:
            entity_id = idle_entity
            media_player = await _resolve_media_player(ha_client, device_id, area_id)
            params = action.get("parameters") or {}
            meta = TimerMetadata(
                name=entity_query,
                entity_id=idle_entity,
                duration=str(params.get("duration", "")) or None,
                origin_device_id=device_id,
                origin_area=area_id,
                media_player_entity=media_player,
            )
            _timer_pool.assign(entity_query, idle_entity, meta)
            friendly_name = entity_query
            logger.info("Pool assigned '%s' -> %s", entity_query, idle_entity)

    if not entity_id:
        if action_name == "start_timer":
            return {
                "success": False,
                "entity_id": None,
                "new_state": None,
                "speech": f"Could not find an entity matching '{entity_query}', and no idle timer entities are available for pool allocation.",
            }
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": f"Could not find an entity matching '{entity_query}'.",
        }

    # Build service data
    service_data = _build_timer_service_data(action)

    expected_state = _EXPECTED_STATE_BY_ACTION.get(action_name)
    verify = await call_service_with_verification(
        ha_client,
        domain,
        service,
        entity_id,
        service_data=service_data,
        expected_state=expected_state,
    )
    if not verify["success"]:
        return {
            "success": False,
            "entity_id": entity_id,
            "new_state": None,
            "speech": f"Failed to execute {action_name} on {friendly_name}: {verify['error']}",
        }

    new_state = verify["observed_state"]

    if expected_state and not verify["verified"]:
        return {
            "success": False,
            "entity_id": entity_id,
            "new_state": new_state,
            "speech": _build_unverified_timer_speech(
                friendly_name=friendly_name,
                expected_state=expected_state,
                observed_state=new_state,
            ),
        }

    # Release pool mapping and cancel delayed tasks when timer is cancelled or finished
    if action_name in ("cancel_timer", "finish_timer") and entity_id:
        _timer_pool.release(entity_id)
        delayed_task_manager.cancel(entity_id)

    return {
        "success": True,
        "entity_id": entity_id,
        "new_state": new_state,
        "speech": build_verified_speech(
            friendly_name=friendly_name,
            action_name=action_name,
            expected_state=expected_state,
            observed_state=new_state,
            verified=verify["verified"],
            action_phrases=_ACTION_PHRASES,
        ),
    }
