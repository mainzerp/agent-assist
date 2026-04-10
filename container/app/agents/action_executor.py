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
            matches = await entity_matcher.match(entity_query)
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
