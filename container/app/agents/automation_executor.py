"""Automation-specific action execution via HA automation services."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

_AUTOMATION_ACTION_MAP: dict[str, tuple[str, str]] = {
    "enable_automation":  ("automation", "turn_on"),
    "disable_automation": ("automation", "turn_off"),
    "trigger_automation": ("automation", "trigger"),
}


def _build_automation_service_data(action: dict) -> dict[str, Any]:
    """Build HA service_data from an automation action's parameters."""
    params = action.get("parameters") or {}
    data: dict[str, Any] = {}

    if "skip_condition" in params:
        data["skip_condition"] = bool(params["skip_condition"])
    if "variables" in params and isinstance(params["variables"], dict):
        data["variables"] = params["variables"]

    return data


async def execute_automation_action(
    action: dict,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None = None,
) -> dict:
    """Resolve an entity, call an automation HA service, and verify the result.

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
    if action_name in ("query_automation_state", "list_automations"):
        return await _handle_automation_read_action(
            action_name, entity_query, ha_client, entity_index, entity_matcher, agent_id
        )

    # Validate action name
    mapping = _AUTOMATION_ACTION_MAP.get(action_name)
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

    # Build service data
    service_data = _build_automation_service_data(action)

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
# Read-only automation action handlers
# ---------------------------------------------------------------------------

def _format_automation_state(entity_id: str, state_resp: dict) -> str:
    state = state_resp.get("state", "unknown")
    attrs = state_resp.get("attributes", {})
    friendly_name = attrs.get("friendly_name", entity_id)
    status = "enabled" if state == "on" else "disabled" if state == "off" else state
    parts = [f"{friendly_name} is {status}"]
    last_triggered = attrs.get("last_triggered")
    if last_triggered:
        parts.append(f"last triggered {last_triggered}")
    return ", ".join(parts) + "."


async def _query_automation_state(
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
        speech = _format_automation_state(entity_id, state_resp)
        return {"success": True, "entity_id": entity_id,
                "new_state": state_resp.get("state"), "speech": speech}
    except Exception as exc:
        logger.error("State query failed for %s", entity_id, exc_info=True)
        return {"success": False, "entity_id": entity_id, "new_state": None,
                "speech": f"Failed to query automation status: {exc}"}


async def _list_automations(ha_client: Any) -> dict:
    try:
        states = await ha_client.get_states()
    except Exception as exc:
        logger.error("Failed to fetch states for list_automations", exc_info=True)
        return {"success": False, "entity_id": "", "new_state": None,
                "speech": f"Failed to list automations: {exc}"}

    automations = [s for s in states if s.get("entity_id", "").startswith("automation.")]

    if not automations:
        return {"success": True, "entity_id": "", "new_state": None,
                "speech": "No automation entities found."}

    enabled = []
    disabled = []
    for a in automations:
        attrs = a.get("attributes", {})
        name = attrs.get("friendly_name", a.get("entity_id", ""))
        state = a.get("state", "unknown")
        last_triggered = attrs.get("last_triggered")
        info = name
        if last_triggered:
            info += f" (last triggered: {last_triggered})"
        if state == "on":
            enabled.append(info)
        else:
            disabled.append(info)

    parts = []
    if enabled:
        parts.append(f"Enabled ({len(enabled)}): {', '.join(enabled)}")
    if disabled:
        parts.append(f"Disabled ({len(disabled)}): {', '.join(disabled)}")
    speech = ". ".join(parts) + "."
    return {"success": True, "entity_id": "", "new_state": None, "speech": speech}


async def _handle_automation_read_action(
    action_name: str,
    entity_query: str,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
) -> dict:
    if action_name == "query_automation_state":
        return await _query_automation_state(entity_query, ha_client, entity_index, entity_matcher, agent_id)
    if action_name == "list_automations":
        return await _list_automations(ha_client)
    return {"success": False, "entity_id": "", "new_state": None,
            "speech": f"Unknown read action: {action_name}"}
