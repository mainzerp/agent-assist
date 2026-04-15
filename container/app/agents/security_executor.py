"""Security-specific action execution via HA lock, alarm, and camera services."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.analytics.tracer import _optional_span

logger = logging.getLogger(__name__)

_SECURITY_ACTION_MAP: dict[str, tuple[str, str]] = {
    # Locks
    "lock":             ("lock", "lock"),
    "unlock":           ("lock", "unlock"),
    # Alarm control panels
    "alarm_arm_home":   ("alarm_control_panel", "alarm_arm_home"),
    "alarm_arm_away":   ("alarm_control_panel", "alarm_arm_away"),
    "alarm_arm_night":  ("alarm_control_panel", "alarm_arm_night"),
    "alarm_disarm":     ("alarm_control_panel", "alarm_disarm"),
    # Cameras
    "camera_turn_on":   ("camera", "turn_on"),
    "camera_turn_off":  ("camera", "turn_off"),
}

_ALLOWED_DOMAINS: frozenset[str] = frozenset({"alarm_control_panel", "lock", "camera", "binary_sensor", "sensor"})


def _validate_domain(entity_id: str) -> bool:
    """Check that entity_id belongs to an allowed domain for this executor."""
    domain = entity_id.split(".")[0] if "." in entity_id else ""
    return domain in _ALLOWED_DOMAINS


def _build_security_service_data(action: dict) -> dict[str, Any]:
    """Build HA service_data from a security action's parameters."""
    params = action.get("parameters") or {}
    data: dict[str, Any] = {}

    if "code" in params:
        data["code"] = str(params["code"])

    return data


async def execute_security_action(
    action: dict,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None = None,
    span_collector=None,
) -> dict:
    """Resolve an entity, call a security HA service, and verify the result.

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
    if action_name in ("query_security_state", "list_security"):
        return await _handle_security_read_action(
            action_name, entity_query, ha_client, entity_index, entity_matcher, agent_id,
            span_collector=span_collector,
        )

    # Validate action name
    mapping = _SECURITY_ACTION_MAP.get(action_name)
    if not mapping:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": f"Unknown action: {action_name}",
        }

    domain, service = mapping

    # Resolve entity via matcher first, then entity_index fallback
    entity_id = None
    friendly_name = entity_query
    try:
        if entity_matcher:
            async with _optional_span(span_collector, "entity_match", agent_id=agent_id) as em_span:
                matches = await entity_matcher.match(entity_query, agent_id=agent_id)
                em_span["metadata"] = {"query": entity_query, "match_count": len(matches)}
                if matches:
                    entity_id = matches[0].entity_id
                    friendly_name = matches[0].friendly_name or entity_id
                    em_span["metadata"]["top_entity_id"] = entity_id
                    em_span["metadata"]["top_friendly_name"] = friendly_name
                    em_span["metadata"]["top_score"] = matches[0].score
                    em_span["metadata"]["signal_scores"] = getattr(matches[0], "signal_scores", {})
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

    # Build service data
    service_data = _build_security_service_data(action)

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

    return {
        "success": True,
        "entity_id": entity_id,
        "new_state": new_state,
        "speech": f"Done, {friendly_name} is now {new_state or action_name.replace('_', ' ')}.",
    }


# ---------------------------------------------------------------------------
# Read-only security action handlers
# ---------------------------------------------------------------------------

_SECURITY_DEVICE_CLASSES = frozenset({
    "motion", "door", "window", "opening", "smoke", "gas",
    "carbon_monoxide", "tamper", "vibration",
})


def _format_security_state(entity_id: str, state_resp: dict) -> str:
    state = state_resp.get("state", "unknown")
    attrs = state_resp.get("attributes", {})
    friendly_name = attrs.get("friendly_name", entity_id)

    if entity_id.startswith("binary_sensor."):
        device_class = attrs.get("device_class", "")
        if device_class in ("door", "window", "opening"):
            label = "open" if state == "on" else "closed"
        elif device_class == "motion":
            label = "motion detected" if state == "on" else "clear"
        elif device_class in ("smoke", "gas", "carbon_monoxide"):
            label = "detected" if state == "on" else "clear"
        else:
            label = state
        return f"{friendly_name}: {label}."

    if entity_id.startswith("lock."):
        return f"{friendly_name} is {state}."

    if entity_id.startswith("alarm_control_panel."):
        return f"{friendly_name} is {state}."

    if entity_id.startswith("camera."):
        return f"{friendly_name} is {state}."

    return f"{friendly_name}: {state}."


async def _query_security_state(
    entity_query: str,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
    span_collector=None,
) -> dict:
    entity_id = None
    friendly_name = entity_query
    try:
        if entity_matcher:
            async with _optional_span(span_collector, "entity_match", agent_id=agent_id) as em_span:
                matches = await entity_matcher.match(entity_query, agent_id=agent_id)
                em_span["metadata"] = {"query": entity_query, "match_count": len(matches)}
                if matches:
                    entity_id = matches[0].entity_id
                    friendly_name = matches[0].friendly_name or entity_id
                    em_span["metadata"]["top_entity_id"] = entity_id
                    em_span["metadata"]["top_friendly_name"] = friendly_name
                    em_span["metadata"]["top_score"] = matches[0].score
                    em_span["metadata"]["signal_scores"] = getattr(matches[0], "signal_scores", {})
    except Exception:
        logger.warning("Entity resolution failed for '%s'", entity_query, exc_info=True)

    if entity_id and not _validate_domain(entity_id):
        logger.warning("Resolved entity %s not in allowed domains %s", entity_id, _ALLOWED_DOMAINS)
        entity_id = None

    if not entity_id:
        return {"success": False, "entity_id": None, "new_state": None,
                "speech": f"Could not find an entity matching '{entity_query}'.",
                "cacheable": False}

    try:
        state_resp = await ha_client.get_state(entity_id)
        if not state_resp:
            return {"success": False, "entity_id": entity_id, "new_state": None,
                    "speech": f"Could not retrieve state for {entity_id}.",
                    "cacheable": False}
        speech = _format_security_state(entity_id, state_resp)
        return {"success": True, "entity_id": entity_id,
                "new_state": state_resp.get("state"), "speech": speech,
                "cacheable": False}
    except Exception as exc:
        logger.error("State query failed for %s", entity_id, exc_info=True)
        return {"success": False, "entity_id": entity_id, "new_state": None,
                "speech": f"Failed to query security status: {exc}",
                "cacheable": False}


async def _list_security(ha_client: Any) -> dict:
    try:
        states = await ha_client.get_states()
    except Exception as exc:
        logger.error("Failed to fetch states for list_security", exc_info=True)
        return {"success": False, "entity_id": "", "new_state": None,
                "speech": f"Failed to list security devices: {exc}"}

    locks = []
    alarms = []
    cameras = []
    binary_sensors = []
    for s in states:
        eid = s.get("entity_id", "")
        if eid.startswith("lock."):
            locks.append(s)
        elif eid.startswith("alarm_control_panel."):
            alarms.append(s)
        elif eid.startswith("camera."):
            cameras.append(s)
        elif eid.startswith("binary_sensor."):
            dc = s.get("attributes", {}).get("device_class", "")
            if dc in _SECURITY_DEVICE_CLASSES:
                binary_sensors.append(s)

    if not locks and not alarms and not cameras and not binary_sensors:
        return {"success": True, "entity_id": "", "new_state": None,
                "speech": "No security devices found."}

    parts = []
    for label, entities in [("Locks", locks), ("Alarms", alarms), ("Cameras", cameras), ("Sensors", binary_sensors)]:
        if entities:
            items = []
            for e in entities:
                name = e.get("attributes", {}).get("friendly_name", e.get("entity_id", ""))
                state = e.get("state", "unknown")
                items.append(f"{name}: {state}")
            parts.append(f"{label}: {'; '.join(items)}")

    speech = ". ".join(parts) + "."
    return {"success": True, "entity_id": "", "new_state": None, "speech": speech,
            "cacheable": False}


async def _handle_security_read_action(
    action_name: str,
    entity_query: str,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
    span_collector=None,
) -> dict:
    if action_name == "query_security_state":
        return await _query_security_state(entity_query, ha_client, entity_index, entity_matcher, agent_id,
                                           span_collector=span_collector)
    if action_name == "list_security":
        return await _list_security(ha_client)
    return {"success": False, "entity_id": "", "new_state": None,
            "speech": f"Unknown read action: {action_name}"}
