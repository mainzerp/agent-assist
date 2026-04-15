"""Climate-specific action execution via HA climate services."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.analytics.tracer import _optional_span

logger = logging.getLogger(__name__)

_CLIMATE_ACTION_MAP: dict[str, tuple[str, str]] = {
    "set_temperature":  ("climate", "set_temperature"),
    "set_hvac_mode":    ("climate", "set_hvac_mode"),
    "set_fan_mode":     ("climate", "set_fan_mode"),
    "set_humidity":     ("climate", "set_humidity"),
    "turn_on":          ("climate", "turn_on"),
    "turn_off":         ("climate", "turn_off"),
}

_ALLOWED_DOMAINS: frozenset[str] = frozenset({"climate", "sensor"})


def _validate_domain(entity_id: str) -> bool:
    """Check that entity_id belongs to an allowed domain for this executor."""
    domain = entity_id.split(".")[0] if "." in entity_id else ""
    return domain in _ALLOWED_DOMAINS


def _build_climate_service_data(action: dict) -> dict[str, Any]:
    """Build HA service_data from a climate action's parameters."""
    params = action.get("parameters") or {}
    data: dict[str, Any] = {}

    if "temperature" in params:
        data["temperature"] = float(params["temperature"])
    if "target_temp_high" in params:
        data["target_temp_high"] = float(params["target_temp_high"])
    if "target_temp_low" in params:
        data["target_temp_low"] = float(params["target_temp_low"])
    if "hvac_mode" in params:
        data["hvac_mode"] = params["hvac_mode"]
    if "fan_mode" in params:
        data["fan_mode"] = params["fan_mode"]
    if "humidity" in params:
        data["humidity"] = int(params["humidity"])

    return data


async def execute_climate_action(
    action: dict,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None = None,
    span_collector=None,
) -> dict:
    """Resolve an entity, call a climate HA service, and verify the result.

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
    if action_name in ("query_climate_state", "list_climate"):
        return await _handle_climate_read_action(
            action_name, entity_query, ha_client, entity_index, entity_matcher, agent_id,
            span_collector=span_collector,
        )

    # Validate action name
    mapping = _CLIMATE_ACTION_MAP.get(action_name)
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
    service_data = _build_climate_service_data(action)

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
# Read-only climate action handlers
# ---------------------------------------------------------------------------

def _format_climate_state(entity_id: str, state_resp: dict) -> str:
    state = state_resp.get("state", "unknown")
    attrs = state_resp.get("attributes", {})
    friendly_name = attrs.get("friendly_name", entity_id)

    if entity_id.startswith("sensor."):
        unit = attrs.get("unit_of_measurement", "")
        return f"{friendly_name}: {state} {unit}".strip() + "."

    # climate.* entity
    parts = [f"{friendly_name} is in {state} mode"]
    current_temp = attrs.get("current_temperature")
    if current_temp is not None:
        parts.append(f"current temperature {current_temp}")
    target_temp = attrs.get("temperature")
    if target_temp is not None:
        parts.append(f"target {target_temp}")
    target_high = attrs.get("target_temp_high")
    target_low = attrs.get("target_temp_low")
    if target_high is not None and target_low is not None:
        parts.append(f"range {target_low}-{target_high}")
    humidity = attrs.get("current_humidity")
    if humidity is not None:
        parts.append(f"humidity {humidity}%")
    fan_mode = attrs.get("fan_mode")
    if fan_mode:
        parts.append(f"fan {fan_mode}")
    return ", ".join(parts) + "."


async def _query_climate_state(
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
        speech = _format_climate_state(entity_id, state_resp)
        return {"success": True, "entity_id": entity_id,
                "new_state": state_resp.get("state"), "speech": speech,
                "cacheable": False}
    except Exception as exc:
        logger.error("State query failed for %s", entity_id, exc_info=True)
        return {"success": False, "entity_id": entity_id, "new_state": None,
                "speech": f"Failed to query climate status: {exc}",
                "cacheable": False}


async def _list_climate(ha_client: Any) -> dict:
    try:
        states = await ha_client.get_states()
    except Exception as exc:
        logger.error("Failed to fetch states for list_climate", exc_info=True)
        return {"success": False, "entity_id": "", "new_state": None,
                "speech": f"Failed to list climate devices: {exc}"}

    climate_entities = []
    sensors = []
    _sensor_keywords = ("temperature", "humidity", "pressure", "dew_point")
    for s in states:
        eid = s.get("entity_id", "")
        if eid.startswith("climate."):
            climate_entities.append(s)
        elif eid.startswith("sensor.") and any(k in eid for k in _sensor_keywords):
            sensors.append(s)

    if not climate_entities and not sensors:
        return {"success": True, "entity_id": "", "new_state": None,
                "speech": "No climate devices or sensors found."}

    parts = []
    if climate_entities:
        lines = []
        for c in climate_entities:
            attrs = c.get("attributes", {})
            name = attrs.get("friendly_name", c.get("entity_id", ""))
            state = c.get("state", "unknown")
            current_temp = attrs.get("current_temperature")
            target_temp = attrs.get("temperature")
            info = f"{name}: {state}"
            if current_temp is not None:
                info += f", current {current_temp}"
            if target_temp is not None:
                info += f", target {target_temp}"
            lines.append(info)
        parts.append("Climate devices: " + "; ".join(lines))
    if sensors:
        lines = []
        for s in sensors:
            attrs = s.get("attributes", {})
            name = attrs.get("friendly_name", s.get("entity_id", ""))
            state = s.get("state", "unknown")
            unit = attrs.get("unit_of_measurement", "")
            lines.append(f"{name}: {state} {unit}".strip())
        parts.append("Sensors: " + "; ".join(lines))

    speech = ". ".join(parts) + "."
    return {"success": True, "entity_id": "", "new_state": None, "speech": speech,
            "cacheable": False}


async def _handle_climate_read_action(
    action_name: str,
    entity_query: str,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
    span_collector=None,
) -> dict:
    if action_name == "query_climate_state":
        return await _query_climate_state(entity_query, ha_client, entity_index, entity_matcher, agent_id,
                                          span_collector=span_collector)
    if action_name == "list_climate":
        return await _list_climate(ha_client)
    return {"success": False, "entity_id": "", "new_state": None,
            "speech": f"Unknown read action: {action_name}"}
