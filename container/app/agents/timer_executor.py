"""Timer-specific action execution via HA timer and input_datetime services."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.agents.delayed_tasks import delayed_task_manager

logger = logging.getLogger(__name__)

_TIMER_ACTION_MAP: dict[str, tuple[str, str]] = {
    "start_timer":   ("timer", "start"),
    "cancel_timer":  ("timer", "cancel"),
    "pause_timer":   ("timer", "pause"),
    "resume_timer":  ("timer", "start"),
    "finish_timer":  ("timer", "finish"),
    "set_datetime":  ("input_datetime", "set_datetime"),
}


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

    def get_entity(self, name: str) -> str | None:
        """Look up entity_id by user-given timer name."""
        return self._name_to_entity.get(name.lower().strip())

    def assign(self, name: str, entity_id: str) -> None:
        """Assign a name mapping to an entity."""
        key = name.lower().strip()
        self._name_to_entity[key] = entity_id
        self._entity_to_name[entity_id] = key

    def release(self, entity_id: str) -> None:
        """Release a timer entity back to the pool."""
        name = self._entity_to_name.pop(entity_id, None)
        if name:
            self._name_to_entity.pop(name, None)

    def get_name(self, entity_id: str) -> str | None:
        """Get the user-given name for an entity, if any."""
        return self._entity_to_name.get(entity_id)

    def all_mappings(self) -> dict[str, str]:
        """Return a copy of all current name->entity mappings."""
        return dict(self._name_to_entity)


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
) -> dict:
    """Handle read-only actions that query state without calling a service."""

    if action_name == "query_timer":
        return await _query_timer(entity_query, ha_client, entity_index, entity_matcher, agent_id)
    if action_name == "list_timers":
        return await _list_timers(ha_client)
    if action_name == "list_alarms":
        return await _list_alarms(ha_client)

    return {"success": False, "entity_id": "", "new_state": None,
            "speech": f"Unknown read action: {action_name}"}


async def _query_timer(
    entity_query: str,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
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
                matches = await entity_matcher.match(entity_query, agent_id=agent_id)
                if matches:
                    entity_id = matches[0].entity_id
            if not entity_id and entity_index:
                results = entity_index.search(entity_query, n_results=1)
                if results:
                    entity_id = results[0][0].entity_id
        except Exception:
            logger.warning("Entity resolution failed for '%s'", entity_query, exc_info=True)

    if not entity_id:
        return {"success": False, "entity_id": None, "new_state": None,
                "speech": f"Could not find an entity matching '{entity_query}'."}

    try:
        state_resp = await ha_client.get_state(entity_id)
        if not state_resp:
            return {"success": False, "entity_id": entity_id, "new_state": None,
                    "speech": f"Could not retrieve state for {entity_id}."}
        speech = _format_timer_state(entity_id, state_resp)
        return {"success": True, "entity_id": entity_id,
                "new_state": state_resp.get("state"), "speech": speech}
    except Exception as exc:
        logger.error("State query failed for %s", entity_id, exc_info=True)
        return {"success": False, "entity_id": entity_id, "new_state": None,
                "speech": f"Failed to query timer status: {exc}"}


async def _list_timers(ha_client: Any) -> dict:
    """List all timer entities with their current state and remaining time."""
    try:
        states = await ha_client.get_states()
    except Exception as exc:
        logger.error("Failed to fetch states for list_timers", exc_info=True)
        return {"success": False, "entity_id": "", "new_state": None,
                "speech": f"Failed to list timers: {exc}"}

    timers = [s for s in states if s.get("entity_id", "").startswith("timer.")]

    if not timers:
        return {"success": True, "entity_id": "", "new_state": None,
                "speech": "No timer entities found."}

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

    if not active and not paused:
        speech = "No timers are currently running."
        if idle:
            speech += f" {len(idle)} idle timer{'s' if len(idle) != 1 else ''} available."
    else:
        speech = ". ".join(parts) + "."

    return {"success": True, "entity_id": "", "new_state": None, "speech": speech}


async def _list_alarms(ha_client: Any) -> dict:
    """List all input_datetime entities (alarms/reminders)."""
    try:
        states = await ha_client.get_states()
    except Exception as exc:
        logger.error("Failed to fetch states for list_alarms", exc_info=True)
        return {"success": False, "entity_id": "", "new_state": None,
                "speech": f"Failed to list alarms: {exc}"}

    alarms = [s for s in states if s.get("entity_id", "").startswith("input_datetime.")]

    if not alarms:
        return {"success": True, "entity_id": "", "new_state": None,
                "speech": "No alarm or input_datetime entities found."}

    lines = []
    for a in alarms:
        state = a.get("state", "unknown")
        attrs = a.get("attributes", {})
        friendly_name = attrs.get("friendly_name", a.get("entity_id", ""))
        lines.append(f"{friendly_name}: {state}")

    speech = "Alarms: " + "; ".join(lines) + "."
    return {"success": True, "entity_id": "", "new_state": None, "speech": speech}


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
            matches = await entity_matcher.match(entity_query, agent_id=agent_id)
            if matches:
                entity_id = matches[0].entity_id
                friendly_name = matches[0].friendly_name or entity_id
        if not entity_id and entity_index:
            results = entity_index.search(entity_query, n_results=1)
            if results:
                entity_id = results[0][0].entity_id
                friendly_name = results[0][0].friendly_name or entity_id
    except Exception:
        logger.warning("Entity resolution failed for '%s'", entity_query, exc_info=True)

    if not entity_id:
        return {"success": False, "entity_id": None, "new_state": None,
                "speech": f"Could not find an entity matching '{entity_query}'."}

    try:
        # Step 1: Cancel the current timer
        await ha_client.call_service("timer", "cancel", entity_id)
        await asyncio.sleep(0.2)

        # Step 2: Restart with snooze duration
        await ha_client.call_service("timer", "start", entity_id, {"duration": snooze_duration})
        await asyncio.sleep(0.3)

        state_resp = await ha_client.get_state(entity_id)
        new_state = state_resp.get("state") if state_resp else None

        secs = _parse_duration_seconds(snooze_duration)
        human_dur = _format_duration_human(secs) if secs else snooze_duration
        return {"success": True, "entity_id": entity_id, "new_state": new_state,
                "speech": f"Snoozed {friendly_name} for {human_dur}."}
    except Exception as exc:
        logger.error("Snooze failed for %s", entity_id, exc_info=True)
        return {"success": False, "entity_id": entity_id, "new_state": None,
                "speech": f"Failed to snooze {friendly_name}: {exc}"}


# ---------------------------------------------------------------------------
# Delayed / notification action handlers
# ---------------------------------------------------------------------------

async def _start_timer_with_notification(
    action: dict,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
) -> dict:
    """Start a timer and schedule a notification for when it finishes."""
    entity_query = action.get("entity", "")
    params = action.get("parameters") or {}
    duration = str(params.get("duration", ""))
    notification_message = str(params.get("notification_message", "Timer finished!"))

    if not duration:
        return {"success": False, "entity_id": None, "new_state": None,
                "speech": "Duration is required for start_timer_with_notification."}

    # Resolve entity (reuse pool logic through regular start)
    start_action = {"action": "start_timer", "entity": entity_query,
                    "parameters": {"duration": duration}}
    result = await execute_timer_action(start_action, ha_client, entity_index,
                                         entity_matcher, agent_id=agent_id)

    if not result.get("success"):
        return result

    entity_id = result.get("entity_id")
    if not entity_id:
        return result

    # Schedule notification via delayed task manager
    async def _send_notification():
        try:
            await ha_client.call_service(
                "persistent_notification", "create", None,
                {"message": notification_message, "title": "Timer Alert"},
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
        "speech": f"Started timer for {human_dur} with notification: \"{notification_message}\".",
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
        return {"success": False, "entity_id": None, "new_state": None,
                "speech": "delay_duration is required for delayed_action."}
    if not target_entity:
        return {"success": False, "entity_id": None, "new_state": None,
                "speech": "target_entity is required for delayed_action."}
    if not target_action or "/" not in target_action:
        return {"success": False, "entity_id": None, "new_state": None,
                "speech": "target_action is required in 'domain/service' format (e.g. 'light/turn_off')."}

    target_domain, target_service = target_action.split("/", 1)

    # Start a timer for the delay
    timer_entity_query = action.get("entity", "delay timer")
    start_action = {"action": "start_timer", "entity": timer_entity_query,
                    "parameters": {"duration": delay_duration}}
    result = await execute_timer_action(start_action, ha_client, entity_index,
                                         entity_matcher, agent_id=agent_id)

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
            logger.error("Delayed action failed: %s/%s on %s",
                         target_domain, target_service, target_entity, exc_info=True)

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
        return {"success": False, "entity_id": None, "new_state": None,
                "speech": "Duration is required for sleep_timer."}
    if not media_player_entity:
        return {"success": False, "entity_id": None, "new_state": None,
                "speech": "media_player entity_id is required for sleep_timer."}

    # Start a timer for the duration
    timer_entity_query = action.get("entity", "sleep timer")
    start_action = {"action": "start_timer", "entity": timer_entity_query,
                    "parameters": {"duration": duration}}
    result = await execute_timer_action(start_action, ha_client, entity_index,
                                         entity_matcher, agent_id=agent_id)

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
            logger.error("Sleep timer: failed to stop media on %s",
                         media_player_entity, exc_info=True)

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
) -> dict:
    """Create a calendar event as a reminder using calendar.create_event."""
    entity_query = action.get("entity", "")
    params = action.get("parameters") or {}
    summary = str(params.get("summary", ""))
    start_time = str(params.get("start_date_time", ""))
    end_time = str(params.get("end_date_time", ""))
    description = str(params.get("description", ""))

    if not summary:
        return {"success": False, "entity_id": None, "new_state": None,
                "speech": "Summary is required for create_reminder."}
    if not start_time:
        return {"success": False, "entity_id": None, "new_state": None,
                "speech": "start_date_time is required for create_reminder."}

    # Resolve calendar entity
    entity_id = None
    friendly_name = entity_query
    try:
        if entity_matcher:
            matches = await entity_matcher.match(entity_query, agent_id=agent_id)
            if matches:
                entity_id = matches[0].entity_id
                friendly_name = matches[0].friendly_name or entity_id
        if not entity_id and entity_index:
            results = entity_index.search(entity_query, n_results=1)
            if results:
                entity_id = results[0][0].entity_id
                friendly_name = results[0][0].friendly_name or entity_id
    except Exception:
        logger.warning("Entity resolution failed for '%s'", entity_query, exc_info=True)

    if not entity_id:
        return {"success": False, "entity_id": None, "new_state": None,
                "speech": f"Could not find a calendar entity matching '{entity_query}'."}

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
        return {"success": False, "entity_id": entity_id, "new_state": None,
                "speech": f"Failed to create reminder: {exc}"}

    return {
        "success": True,
        "entity_id": entity_id,
        "new_state": None,
        "speech": f"Created reminder \"{summary}\" at {start_time} on {friendly_name}.",
    }


async def _create_recurring_reminder(
    action: dict,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
) -> dict:
    """Create a recurring calendar event using calendar.create_event with rrule."""
    entity_query = action.get("entity", "")
    params = action.get("parameters") or {}
    summary = str(params.get("summary", ""))
    start_time = str(params.get("start_date_time", ""))
    end_time = str(params.get("end_date_time", ""))
    rrule = str(params.get("rrule", ""))

    if not summary:
        return {"success": False, "entity_id": None, "new_state": None,
                "speech": "Summary is required for create_recurring_reminder."}
    if not start_time:
        return {"success": False, "entity_id": None, "new_state": None,
                "speech": "start_date_time is required for create_recurring_reminder."}
    if not rrule:
        return {"success": False, "entity_id": None, "new_state": None,
                "speech": "rrule is required for create_recurring_reminder (e.g. 'FREQ=DAILY', 'FREQ=WEEKLY;BYDAY=MO,WE,FR')."}

    # Resolve calendar entity
    entity_id = None
    friendly_name = entity_query
    try:
        if entity_matcher:
            matches = await entity_matcher.match(entity_query, agent_id=agent_id)
            if matches:
                entity_id = matches[0].entity_id
                friendly_name = matches[0].friendly_name or entity_id
        if not entity_id and entity_index:
            results = entity_index.search(entity_query, n_results=1)
            if results:
                entity_id = results[0][0].entity_id
                friendly_name = results[0][0].friendly_name or entity_id
    except Exception:
        logger.warning("Entity resolution failed for '%s'", entity_query, exc_info=True)

    if not entity_id:
        return {"success": False, "entity_id": None, "new_state": None,
                "speech": f"Could not find a calendar entity matching '{entity_query}'."}

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
        return {"success": False, "entity_id": entity_id, "new_state": None,
                "speech": f"Failed to create recurring reminder: {exc}"}

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
        "speech": f"Created {freq} reminder \"{summary}\" at {start_time} on {friendly_name}.",
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
        return await _handle_read_action(action_name, entity_query, ha_client,
                                          entity_index, entity_matcher, agent_id)

    # Multi-step actions (custom logic, not simple service calls)
    if action_name == "snooze_timer":
        return await _snooze_timer(action, ha_client, entity_index, entity_matcher, agent_id)
    if action_name == "start_timer_with_notification":
        return await _start_timer_with_notification(action, ha_client, entity_index,
                                                      entity_matcher, agent_id)
    if action_name == "delayed_action":
        return await _delayed_action(action, ha_client, entity_index, entity_matcher, agent_id)
    if action_name == "sleep_timer":
        return await _sleep_timer(action, ha_client, entity_index, entity_matcher, agent_id)
    if action_name == "create_reminder":
        return await _create_reminder(action, ha_client, entity_index, entity_matcher, agent_id)
    if action_name == "create_recurring_reminder":
        return await _create_recurring_reminder(action, ha_client, entity_index, entity_matcher, agent_id)

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
    pool_name = None

    # Check timer pool for named timer mapping
    pool_entity = _timer_pool.get_entity(entity_query)
    if pool_entity:
        entity_id = pool_entity
        friendly_name = entity_query
        pool_name = entity_query

    if not entity_id:
        try:
            if entity_matcher:
                matches = await entity_matcher.match(entity_query, agent_id=agent_id)
                if matches:
                    entity_id = matches[0].entity_id
                    friendly_name = matches[0].friendly_name or entity_id
            if not entity_id and entity_index:
                results = entity_index.search(entity_query, n_results=1)
                if results:
                    entity_id = results[0][0].entity_id
                    friendly_name = results[0][0].friendly_name or entity_id
        except Exception:
            logger.warning("Entity resolution failed for '%s'", entity_query, exc_info=True)

    # For start_timer: if no entity found, try to allocate from pool
    if not entity_id and action_name == "start_timer":
        idle_entity = await _find_idle_timer(ha_client)
        if idle_entity:
            entity_id = idle_entity
            _timer_pool.assign(entity_query, idle_entity)
            pool_name = entity_query
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

    # Execute the service call
    try:
        await ha_client.call_service(domain, service, entity_id, service_data or None)
    except Exception as exc:
        logger.error("Service call failed: %s/%s on %s", domain, service, entity_id, exc_info=True)
        return {
            "success": False,
            "entity_id": entity_id,
            "new_state": None,
            "speech": f"Failed to execute {action_name} on {friendly_name}: {exc}",
        }

    # Brief wait for state propagation, then verify
    await asyncio.sleep(0.3)
    new_state = None
    try:
        state_resp = await ha_client.get_state(entity_id)
        if state_resp:
            new_state = state_resp.get("state")
    except Exception:
        logger.warning("State verification failed for %s", entity_id, exc_info=True)

    # Release pool mapping and cancel delayed tasks when timer is cancelled or finished
    if action_name in ("cancel_timer", "finish_timer") and entity_id:
        _timer_pool.release(entity_id)
        delayed_task_manager.cancel(entity_id)

    return {
        "success": True,
        "entity_id": entity_id,
        "new_state": new_state,
        "speech": f"Done, {friendly_name} is now {new_state or action_name.replace('_', ' ')}.",
    }
