"""Timer-specific action execution.

In 0.26.0 the HA ``timer.*`` helper-pool model was removed entirely.
Plain timer requests are delegated to HA's native Assist engine when
the LLM picks ``delegate_native_plain_timer`` (handled in
``app.agents.timer``); every other timer-shaped action routes to the
AgentHub-managed ``TimerScheduler`` (``app.agents.timer_scheduler``).

This module retains:
- read-only handlers (``query_timer``, ``list_timers`` against the
  scheduler; ``list_alarms`` against ``input_datetime.*`` HA entities)
- ``set_datetime`` (HA ``input_datetime.set_datetime``)
- ``create_reminder`` / ``create_recurring_reminder`` (HA
  ``calendar.create_event``)

All HA ``timer.*`` service calls, the ``_TimerPool`` class, the
``_find_idle_timer`` allocator, the ``on_timer_finished`` WebSocket
handler, and the expired-timer tracking deque are deleted.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from app.agents.action_executor import (
    call_service_with_verification,
    filter_matches_by_domain,
)
from app.analytics.tracer import _optional_span

logger = logging.getLogger(__name__)


_TIMER_ACTION_MAP: dict[str, tuple[str, str]] = {
    "set_datetime": ("input_datetime", "set_datetime"),
}

_ACTION_PHRASES: dict[str, str] = {
    "set_datetime": "updated",
}

_ALLOWED_DOMAINS: frozenset[str] = frozenset({"input_datetime"})

_ACTION_DOMAINS: dict[str, frozenset[str]] = {
    "set_datetime": frozenset({"input_datetime"}),
}

_INPUT_DATETIME_DOMAINS: frozenset[str] = frozenset({"input_datetime"})
_CALENDAR_DOMAINS: frozenset[str] = frozenset({"calendar"})


def _validate_domain(entity_id: str) -> bool:
    domain = entity_id.split(".")[0] if "." in entity_id else ""
    return domain in _ALLOWED_DOMAINS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_timer_service_data(action: dict) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Scheduler accessor
# ---------------------------------------------------------------------------


def _get_scheduler() -> Any | None:
    """Return the process-wide ``TimerScheduler``, if available.

    Falls back to ``None`` in unit-test contexts that exercise this
    module without a running FastAPI app. Tests that need scheduler
    behaviour patch ``app.agents.timer_executor._get_scheduler``
    directly.
    """
    try:
        from app.main import app

        return getattr(app.state, "timer_scheduler", None)
    except Exception:
        return None


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
    *,
    area_id: str | None = None,
) -> dict:
    if action_name == "query_timer":
        return await _query_timer(entity_query, area_id=area_id)
    if action_name == "list_timers":
        return await _list_timers(area_id=area_id)
    if action_name == "list_alarms":
        return await _list_alarms(ha_client)
    return {"success": False, "entity_id": "", "new_state": None, "speech": f"Unknown read action: {action_name}"}


async def _query_timer(entity_query: str, *, area_id: str | None = None) -> dict:
    scheduler = _get_scheduler()
    if scheduler is None:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Timer scheduler is unavailable.",
            "cacheable": False,
        }
    rows = await scheduler.list(logical_name=entity_query or None, area=area_id)
    if not rows:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": f"No timer named '{entity_query}' is currently running.",
            "cacheable": False,
        }
    row = rows[0]
    remaining = max(0, int(row["fires_at"]) - int(datetime.now().timestamp()))
    human = _format_duration_human(remaining)
    return {
        "success": True,
        "entity_id": None,
        "new_state": "active",
        "speech": f"{row['logical_name']} has {human} remaining.",
        "cacheable": False,
    }


async def _list_timers(*, area_id: str | None = None) -> dict:
    scheduler = _get_scheduler()
    if scheduler is None:
        return {
            "success": True,
            "entity_id": "",
            "new_state": None,
            "speech": "No timers are currently running.",
            "cacheable": False,
        }
    rows = await scheduler.list(area=area_id)
    if not rows:
        return {
            "success": True,
            "entity_id": "",
            "new_state": None,
            "speech": "No timers are currently running.",
            "cacheable": False,
        }
    now = int(datetime.now().timestamp())
    parts: list[str] = []
    for row in rows:
        remaining = max(0, int(row["fires_at"]) - now)
        parts.append(f"{row['logical_name']} ({_format_duration_human(remaining)} remaining)")
    return {
        "success": True,
        "entity_id": "",
        "new_state": None,
        "speech": "Active: " + ", ".join(parts) + ".",
        "cacheable": False,
    }


async def _list_alarms(ha_client: Any) -> dict:
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
    return {
        "success": True,
        "entity_id": "",
        "new_state": None,
        "speech": "Alarms: " + "; ".join(lines) + ".",
        "cacheable": False,
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

    entity_id = None
    friendly_name = entity_query
    try:
        if entity_matcher:
            async with _optional_span(span_collector, "entity_match", agent_id=agent_id) as em_span:
                matches = await entity_matcher.match(entity_query, agent_id=agent_id)
                em_span["metadata"] = {"query": entity_query, "match_count": len(matches)}
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
    service_data["end_date_time"] = end_time or start_time
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

    entity_id = None
    friendly_name = entity_query
    try:
        if entity_matcher:
            async with _optional_span(span_collector, "entity_match", agent_id=agent_id) as em_span:
                matches = await entity_matcher.match(entity_query, agent_id=agent_id)
                em_span["metadata"] = {"query": entity_query, "match_count": len(matches)}
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
    service_data["end_date_time"] = end_time or start_time

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
# Scheduler-routed action handlers
# ---------------------------------------------------------------------------


_DEFAULT_SNOOZE_DURATION = "00:05:00"


async def _start_timer(
    action: dict,
    *,
    device_id: str | None,
    area_id: str | None,
    language: str | None,
) -> dict:
    entity_query = (action.get("entity") or "").strip()
    params = action.get("parameters") or {}
    duration = str(params.get("duration", ""))
    seconds = _parse_duration_seconds(duration)
    if not duration or seconds is None or seconds <= 0:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Duration is required for start_timer.",
        }
    scheduler = _get_scheduler()
    if scheduler is None:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Timer scheduler is unavailable.",
        }
    logical_name = entity_query or f"{seconds // 60}-minute timer"
    timer_id = await scheduler.schedule(
        logical_name=logical_name,
        kind="plain",
        duration_seconds=seconds,
        origin_device_id=device_id,
        origin_area=area_id,
        payload={"duration": duration, "language": language},
    )
    human = _format_duration_human(seconds)
    return {
        "success": True,
        "entity_id": None,
        "new_state": "active",
        "speech": f"Started {logical_name} for {human}.",
        "metadata": {"scheduler_timer_id": timer_id},
    }


async def _cancel_timer(
    action: dict,
    *,
    area_id: str | None,
) -> dict:
    entity_query = (action.get("entity") or "").strip()
    scheduler = _get_scheduler()
    if scheduler is None:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Timer scheduler is unavailable.",
        }
    if not entity_query:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Please specify which timer to cancel.",
        }
    count = await scheduler.cancel(logical_name=entity_query, area=area_id)
    if count == 0:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": f"No timer named '{entity_query}' is running.",
        }
    return {
        "success": True,
        "entity_id": None,
        "new_state": "idle",
        "speech": f"Cancelled {entity_query}.",
    }


async def _snooze_timer(
    action: dict,
    *,
    device_id: str | None,
    area_id: str | None,
    language: str | None,
) -> dict:
    entity_query = (action.get("entity") or "").strip()
    params = action.get("parameters") or {}
    snooze_duration = str(params.get("duration", _DEFAULT_SNOOZE_DURATION))
    seconds = _parse_duration_seconds(snooze_duration) or 0
    if seconds <= 0:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Invalid snooze duration.",
        }
    scheduler = _get_scheduler()
    if scheduler is None:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Timer scheduler is unavailable.",
        }
    if entity_query:
        await scheduler.cancel(logical_name=entity_query, area=area_id)
    logical_name = entity_query or "snoozed timer"
    await scheduler.schedule(
        logical_name=logical_name,
        kind="snooze",
        duration_seconds=seconds,
        origin_device_id=device_id,
        origin_area=area_id,
        payload={"snooze_seconds": seconds, "language": language},
    )
    human = _format_duration_human(seconds)
    return {
        "success": True,
        "entity_id": None,
        "new_state": "active",
        "speech": f"Snoozed {logical_name} for {human}.",
    }


async def _start_timer_with_notification(
    action: dict,
    *,
    device_id: str | None,
    area_id: str | None,
    language: str | None,
) -> dict:
    entity_query = (action.get("entity") or "").strip()
    params = action.get("parameters") or {}
    duration = str(params.get("duration", ""))
    notification_message = str(params.get("notification_message", "Timer finished!"))
    seconds = _parse_duration_seconds(duration)
    if not duration or seconds is None or seconds <= 0:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Duration is required for start_timer_with_notification.",
        }
    scheduler = _get_scheduler()
    if scheduler is None:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Timer scheduler is unavailable.",
        }
    logical_name = entity_query or f"{seconds // 60}-minute timer"
    await scheduler.schedule(
        logical_name=logical_name,
        kind="notification",
        duration_seconds=seconds,
        origin_device_id=device_id,
        origin_area=area_id,
        payload={"notification_message": notification_message, "duration": duration, "language": language},
    )
    human = _format_duration_human(seconds)
    return {
        "success": True,
        "entity_id": None,
        "new_state": "active",
        "speech": f'Started timer for {human} with notification: "{notification_message}".',
    }


async def _delayed_action(
    action: dict,
    *,
    device_id: str | None,
    area_id: str | None,
    language: str | None,
) -> dict:
    entity_query = (action.get("entity") or "delay timer").strip() or "delay timer"
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
    seconds = _parse_duration_seconds(delay_duration)
    if seconds is None or seconds <= 0:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Invalid delay_duration.",
        }
    scheduler = _get_scheduler()
    if scheduler is None:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Timer scheduler is unavailable.",
        }
    await scheduler.schedule(
        logical_name=entity_query,
        kind="delayed_action",
        duration_seconds=seconds,
        origin_device_id=device_id,
        origin_area=area_id,
        payload={"target_entity": target_entity, "target_action": target_action, "language": language},
    )
    human = _format_duration_human(seconds)
    return {
        "success": True,
        "entity_id": None,
        "new_state": "active",
        "speech": f"Scheduled {target_action.replace('/', ' ')} on {target_entity} in {human}.",
    }


async def _sleep_timer(
    action: dict,
    *,
    device_id: str | None,
    area_id: str | None,
    language: str | None,
) -> dict:
    entity_query = (action.get("entity") or "sleep timer").strip() or "sleep timer"
    params = action.get("parameters") or {}
    duration = str(params.get("duration", ""))
    media_player_entity = str(params.get("media_player", ""))
    seconds = _parse_duration_seconds(duration)
    if not duration or seconds is None or seconds <= 0:
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
    scheduler = _get_scheduler()
    if scheduler is None:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Timer scheduler is unavailable.",
        }
    await scheduler.schedule(
        logical_name=entity_query,
        kind="sleep",
        duration_seconds=seconds,
        origin_device_id=device_id,
        origin_area=area_id,
        payload={"media_player": media_player_entity, "duration": duration, "language": language},
    )
    human = _format_duration_human(seconds)
    return {
        "success": True,
        "entity_id": None,
        "new_state": "active",
        "speech": (f"Sleep timer set for {human}. Media on {media_player_entity} will stop when the timer ends."),
    }


async def _pause_or_resume_or_finish(
    action: dict,
    *,
    area_id: str | None,
) -> dict:
    """``pause_timer``/``resume_timer``/``finish_timer`` against the scheduler.

    The scheduler does not yet model true pause/resume; the simplest
    correct behaviour is: ``pause`` cancels the pending timer (so it
    will not fire), ``resume`` is rejected with a clear message
    (the user must restart), ``finish`` cancels and reports done.
    """
    action_name = action.get("action", "")
    entity_query = (action.get("entity") or "").strip()
    scheduler = _get_scheduler()
    if scheduler is None:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Timer scheduler is unavailable.",
        }
    if action_name == "resume_timer":
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": "Resume is not supported for AgentHub timers; please start a new timer.",
        }
    if not entity_query:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": f"Please specify which timer to {action_name.replace('_timer', '')}.",
        }
    count = await scheduler.cancel(logical_name=entity_query, area=area_id)
    if count == 0:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": f"No timer named '{entity_query}' is running.",
        }
    if action_name == "finish_timer":
        return {
            "success": True,
            "entity_id": None,
            "new_state": "idle",
            "speech": f"Finished {entity_query}.",
        }
    return {
        "success": True,
        "entity_id": None,
        "new_state": "paused",
        "speech": f"Paused {entity_query}.",
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
    language: str | None = None,
    span_collector=None,
) -> dict:
    """Dispatch a parsed timer action.

    All non-plain timer-shaped flows route to ``TimerScheduler``; HA
    ``timer.*`` services are no longer used. ``set_datetime``,
    ``list_alarms``, and the calendar reminders still go to HA.
    """
    action_name = action.get("action", "").lower()
    entity_query = action.get("entity", "")

    if action_name in ("query_timer", "list_timers", "list_alarms"):
        return await _handle_read_action(
            action_name,
            entity_query,
            ha_client,
            entity_index,
            entity_matcher,
            agent_id,
            span_collector=span_collector,
            area_id=area_id,
        )

    if action_name == "start_timer":
        return await _start_timer(action, device_id=device_id, area_id=area_id, language=language)
    if action_name == "cancel_timer":
        return await _cancel_timer(action, area_id=area_id)
    if action_name in ("pause_timer", "resume_timer", "finish_timer"):
        return await _pause_or_resume_or_finish(action, area_id=area_id)
    if action_name == "snooze_timer":
        return await _snooze_timer(action, device_id=device_id, area_id=area_id, language=language)
    if action_name == "start_timer_with_notification":
        return await _start_timer_with_notification(action, device_id=device_id, area_id=area_id, language=language)
    if action_name == "delayed_action":
        return await _delayed_action(action, device_id=device_id, area_id=area_id, language=language)
    if action_name == "sleep_timer":
        return await _sleep_timer(action, device_id=device_id, area_id=area_id, language=language)
    if action_name == "create_reminder":
        return await _create_reminder(
            action, ha_client, entity_index, entity_matcher, agent_id, span_collector=span_collector
        )
    if action_name == "create_recurring_reminder":
        return await _create_recurring_reminder(
            action, ha_client, entity_index, entity_matcher, agent_id, span_collector=span_collector
        )

    # Fall through: simple HA service-call actions (input_datetime only).
    mapping = _TIMER_ACTION_MAP.get(action_name)
    if not mapping:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": f"Unknown timer action: {action_name}",
        }

    domain, service = mapping
    entity_id = None
    friendly_name = entity_query

    try:
        if entity_matcher:
            async with _optional_span(span_collector, "entity_match", agent_id=agent_id) as em_span:
                matches = await entity_matcher.match(entity_query, agent_id=agent_id)
                em_span["metadata"] = {"query": entity_query, "match_count": len(matches)}
                required_domains = _ACTION_DOMAINS.get(action_name, _INPUT_DATETIME_DOMAINS)
                filtered = filter_matches_by_domain(matches, required_domains)
                if filtered:
                    entity_id = filtered[0].entity_id
                    friendly_name = filtered[0].friendly_name or entity_id
                    em_span["metadata"]["top_entity_id"] = entity_id
                    em_span["metadata"]["top_friendly_name"] = friendly_name
                    em_span["metadata"]["top_score"] = filtered[0].score
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
        }

    service_data = _build_timer_service_data(action)
    verify = await call_service_with_verification(
        ha_client,
        domain,
        service,
        entity_id,
        service_data=service_data,
    )
    if not verify["success"]:
        return {
            "success": False,
            "entity_id": entity_id,
            "new_state": None,
            "speech": f"Failed to execute {action_name} on {friendly_name}: {verify['error']}",
        }
    return {
        "success": True,
        "entity_id": entity_id,
        "new_state": verify.get("observed_state"),
        "speech": f"Updated {friendly_name}.",
    }
