"""Music-specific action execution via Music Assistant and media_player services."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

_MUSIC_ACTION_MAP: dict[str, tuple[str, str]] = {
    "play_media":           ("mass", "play_media"),
    "search":               ("mass", "search"),
    "volume_set":           ("media_player", "volume_set"),
    "media_play":           ("media_player", "media_play"),
    "media_pause":          ("media_player", "media_pause"),
    "media_next_track":     ("media_player", "media_next_track"),
    "media_previous_track": ("media_player", "media_previous_track"),
    "shuffle_set":          ("media_player", "shuffle_set"),
    "repeat_set":           ("media_player", "repeat_set"),
}


def _build_music_service_data(action: dict) -> dict[str, Any]:
    """Build HA service_data from a music action's parameters."""
    params = action.get("parameters") or {}
    action_name = action.get("action", "")
    data: dict[str, Any] = {}

    if action_name == "play_media":
        if "media_id" in params:
            data["media_id"] = params["media_id"]
        if "media_type" in params:
            data["media_type"] = params["media_type"]
        if "enqueue" in params:
            data["enqueue"] = params["enqueue"]
    elif action_name == "search":
        if "name" in params:
            data["name"] = params["name"]
        if "media_type" in params:
            data["media_type"] = params["media_type"]
        if "limit" in params:
            data["limit"] = int(params["limit"])
    elif action_name == "volume_set":
        if "volume_level" in params:
            data["volume_level"] = float(params["volume_level"])
    elif action_name == "shuffle_set":
        if "shuffle" in params:
            data["shuffle"] = bool(params["shuffle"])
    elif action_name == "repeat_set":
        if "repeat" in params:
            data["repeat"] = params["repeat"]

    return data


def _format_search_results(results: Any) -> str:
    """Format search results from mass.search into readable speech."""
    if not results:
        return "No results found for that search."

    if isinstance(results, dict):
        items = results.get("items") or results.get("result") or []
    elif isinstance(results, list):
        items = results
    else:
        return "No results found for that search."

    if not items:
        return "No results found for that search."

    lines = []
    for i, item in enumerate(items[:10], 1):
        if isinstance(item, dict):
            name = item.get("name") or item.get("title") or "Unknown"
            artist = item.get("artist") or item.get("artists") or ""
            if isinstance(artist, list):
                artist = ", ".join(str(a) for a in artist)
            if artist:
                lines.append(f"{i}. {name} by {artist}")
            else:
                lines.append(f"{i}. {name}")
        else:
            lines.append(f"{i}. {item}")

    return "I found: " + "; ".join(lines) + "."


async def execute_music_action(
    action: dict,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None = None,
) -> dict:
    """Resolve an entity, call a music HA service, and verify the result.

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
    mapping = _MUSIC_ACTION_MAP.get(action_name)
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
    service_data = _build_music_service_data(action)

    # Special case: search returns speech with results
    if action_name == "search":
        try:
            results = await ha_client.call_service(domain, service, entity_id, service_data or None)
            speech = _format_search_results(results)
            return {
                "success": True,
                "entity_id": entity_id,
                "new_state": None,
                "speech": speech,
            }
        except Exception as exc:
            logger.error("Search service call failed on %s", entity_id, exc_info=True)
            return {
                "success": False,
                "entity_id": entity_id,
                "new_state": None,
                "speech": f"Failed to search on {friendly_name}: {exc}",
            }

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
