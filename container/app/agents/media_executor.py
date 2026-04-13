"""Media-player action execution via HA media_player services."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

_MEDIA_ACTION_MAP: dict[str, tuple[str, str]] = {
    "turn_on":           ("media_player", "turn_on"),
    "turn_off":          ("media_player", "turn_off"),
    "play":              ("media_player", "media_play"),
    "pause":             ("media_player", "media_pause"),
    "stop":              ("media_player", "media_stop"),
    "next_track":        ("media_player", "media_next_track"),
    "previous_track":    ("media_player", "media_previous_track"),
    "set_volume":        ("media_player", "volume_set"),
    "mute":              ("media_player", "volume_mute"),
    "select_source":     ("media_player", "select_source"),
    "play_media":        ("media_player", "play_media"),
}


def _build_media_service_data(action: dict) -> dict[str, Any]:
    """Build HA service_data from a media action's parameters."""
    params = action.get("parameters") or {}
    action_name = action.get("action", "")
    data: dict[str, Any] = {}

    if action_name == "set_volume":
        if "volume_level" in params:
            data["volume_level"] = float(params["volume_level"])
    elif action_name == "mute":
        if "is_volume_muted" in params:
            data["is_volume_muted"] = bool(params["is_volume_muted"])
    elif action_name == "select_source":
        if "source" in params:
            data["source"] = str(params["source"])
    elif action_name == "play_media":
        if "media_content_id" in params:
            data["media_content_id"] = str(params["media_content_id"])
        if "media_content_type" in params:
            data["media_content_type"] = str(params["media_content_type"])

    return data


async def execute_media_action(
    action: dict,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None = None,
) -> dict:
    """Resolve an entity, call a media_player HA service, and verify the result.

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

    mapping = _MEDIA_ACTION_MAP.get(action_name)
    if not mapping:
        return {
            "success": False,
            "entity_id": None,
            "new_state": None,
            "speech": f"Unknown action: {action_name}",
        }

    domain, service = mapping

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

    service_data = _build_media_service_data(action)

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
