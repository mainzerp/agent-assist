"""Shared action parsing, execution, and verification for domain agents."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Regex to find JSON blocks in LLM output (fenced)
_JSON_FENCE_RE = re.compile(r"```json\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _try_parse_json_with_action(text: str) -> dict | None:
    """Try to parse a JSON object containing an 'action' key from text.

    Scans for '{' characters and attempts json.loads from each position.
    """
    start = 0
    while True:
        idx = text.find("{", start)
        if idx == -1:
            break
        try:
            data = json.loads(text[idx:])
            if isinstance(data, dict) and "action" in data:
                return data
        except json.JSONDecodeError:
            pass
        # Try to find end of object by scanning for balanced braces
        depth = 0
        end = idx
        for i, ch in enumerate(text[idx:], idx):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > idx:
            try:
                data = json.loads(text[idx:end])
                if isinstance(data, dict) and "action" in data:
                    return data
            except json.JSONDecodeError:
                pass
        start = idx + 1
    return None


def parse_action(llm_response: str) -> dict | None:
    """Extract a structured action dict from an LLM response.

    Looks for JSON in ```json``` fences first, then falls back to raw JSON
    objects containing an "action" key.

    Expected format:
        {"action": "turn_on", "entity": "kitchen light", "parameters": {}}

    Returns None if no valid action block is found.
    """
    # Try fenced JSON blocks first
    match = _JSON_FENCE_RE.search(llm_response)
    if match:
        result = _try_parse_json_with_action(match.group(1))
        if result:
            return result

    # Fallback: raw JSON object with "action" key
    return _try_parse_json_with_action(llm_response)


# Map action names to (service, extra_data_builder)
_ACTION_SERVICE_MAP: dict[str, str] = {
    "turn_on": "turn_on",
    "turn_off": "turn_off",
    "toggle": "toggle",
    "set_brightness": "turn_on",
    "set_color": "turn_on",
    "set_color_temp": "turn_on",
}


def _build_service_data(action: dict) -> dict[str, Any]:
    """Build HA service_data from action parameters."""
    params = action.get("parameters") or {}
    data: dict[str, Any] = {}

    if "brightness" in params:
        data["brightness"] = int(params["brightness"])
    if "color_name" in params:
        data["color_name"] = params["color_name"]
    if "rgb_color" in params:
        data["rgb_color"] = params["rgb_color"]
    if "color_temp" in params:
        data["color_temp"] = int(params["color_temp"])
    if "color_temp_kelvin" in params:
        data["color_temp_kelvin"] = int(params["color_temp_kelvin"])
    if "transition" in params:
        data["transition"] = float(params["transition"])

    return data


async def execute_action(
    action: dict,
    ha_client,
    entity_index,
    entity_matcher,
    agent_id: str | None = None,
) -> dict:
    """Resolve an entity, call a HA service, and verify the result.

    Args:
        action: Parsed action dict with "action", "entity", and optional "parameters".
        ha_client: HARestClient instance.
        entity_index: EntityIndex instance.
        entity_matcher: EntityMatcher instance.

    Returns:
        dict with "success", "entity_id", "new_state", and "speech".
    """
    action_name = action.get("action", "").lower()
    entity_query = action.get("entity", "")

    # Read-only actions (no service call)
    if action_name in ("query_light_state", "list_lights"):
        return await _handle_light_read_action(
            action_name, entity_query, ha_client, entity_index, entity_matcher, agent_id
        )

    # Validate action name
    service = _ACTION_SERVICE_MAP.get(action_name)
    if not service:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": f"Unknown action: {action_name}",
        }

    # Resolve entity via matcher
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
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": f"Could not find an entity matching '{entity_query}'.",
        }

    # Extract domain from entity_id
    domain = entity_id.split(".")[0] if "." in entity_id else "light"

    # Build service data
    service_data = _build_service_data(action)

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
# Read-only light/switch action handlers
# ---------------------------------------------------------------------------

def _format_light_state(entity_id: str, state_resp: dict) -> str:
    state = state_resp.get("state", "unknown")
    attrs = state_resp.get("attributes", {})
    friendly_name = attrs.get("friendly_name", entity_id)

    if entity_id.startswith("switch."):
        return f"{friendly_name} is {state}."

    if entity_id.startswith("sensor."):
        unit = attrs.get("unit_of_measurement", "")
        return f"{friendly_name}: {state} {unit}".strip() + "."

    # light.* entity
    parts = [f"{friendly_name} is {state}"]
    if state == "on":
        brightness = attrs.get("brightness")
        if brightness is not None:
            pct = round(int(brightness) / 255 * 100)
            parts.append(f"brightness {pct}%")
        color_name = attrs.get("color_name")
        if color_name:
            parts.append(f"color {color_name}")
        rgb = attrs.get("rgb_color")
        if rgb and not color_name:
            parts.append(f"RGB {rgb}")
        color_temp = attrs.get("color_temp")
        if color_temp:
            parts.append(f"color temp {color_temp} mireds")
    return ", ".join(parts) + "."


async def _query_light_state(
    entity_query: str,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
) -> dict:
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
        state_resp = await ha_client.get_state(entity_id)
        if not state_resp:
            return {"success": False, "entity_id": entity_id, "new_state": None,
                    "speech": f"Could not retrieve state for {entity_id}."}
        speech = _format_light_state(entity_id, state_resp)
        return {"success": True, "entity_id": entity_id,
                "new_state": state_resp.get("state"), "speech": speech}
    except Exception as exc:
        logger.error("State query failed for %s", entity_id, exc_info=True)
        return {"success": False, "entity_id": entity_id, "new_state": None,
                "speech": f"Failed to query light status: {exc}"}


async def _list_lights(ha_client: Any) -> dict:
    try:
        states = await ha_client.get_states()
    except Exception as exc:
        logger.error("Failed to fetch states for list_lights", exc_info=True)
        return {"success": False, "entity_id": "", "new_state": None,
                "speech": f"Failed to list lights: {exc}"}

    lights_on = []
    lights_off = []
    switches_on = []
    switches_off = []
    for s in states:
        eid = s.get("entity_id", "")
        state = s.get("state", "unknown")
        name = s.get("attributes", {}).get("friendly_name", eid)
        if eid.startswith("light."):
            if state == "on":
                lights_on.append(name)
            else:
                lights_off.append(name)
        elif eid.startswith("switch."):
            if state == "on":
                switches_on.append(name)
            else:
                switches_off.append(name)

    if not lights_on and not lights_off and not switches_on and not switches_off:
        return {"success": True, "entity_id": "", "new_state": None,
                "speech": "No light or switch entities found."}

    parts = []
    if lights_on:
        parts.append(f"Lights on: {', '.join(lights_on)}")
    if lights_off:
        parts.append(f"Lights off: {', '.join(lights_off)}")
    if switches_on:
        parts.append(f"Switches on: {', '.join(switches_on)}")
    if switches_off:
        parts.append(f"Switches off: {', '.join(switches_off)}")
    speech = ". ".join(parts) + "."
    return {"success": True, "entity_id": "", "new_state": None, "speech": speech}


async def _handle_light_read_action(
    action_name: str,
    entity_query: str,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
) -> dict:
    if action_name == "query_light_state":
        return await _query_light_state(entity_query, ha_client, entity_index, entity_matcher, agent_id)
    if action_name == "list_lights":
        return await _list_lights(ha_client)
    return {"success": False, "entity_id": "", "new_state": None,
            "speech": f"Unknown read action: {action_name}"}
