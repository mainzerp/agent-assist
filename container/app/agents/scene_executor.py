"""Scene-specific action execution via HA scene services."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.analytics.tracer import _optional_span

logger = logging.getLogger(__name__)

_SCENE_ACTION_MAP: dict[str, tuple[str, str]] = {
    "activate_scene": ("scene", "turn_on"),
}

_ALLOWED_DOMAINS: frozenset[str] = frozenset({"scene"})


def _validate_domain(entity_id: str) -> bool:
    """Check that entity_id belongs to an allowed domain for this executor."""
    domain = entity_id.split(".")[0] if "." in entity_id else ""
    return domain in _ALLOWED_DOMAINS


def _build_scene_service_data(action: dict) -> dict[str, Any]:
    """Build HA service_data from a scene action's parameters."""
    params = action.get("parameters") or {}
    data: dict[str, Any] = {}

    if "transition" in params:
        data["transition"] = float(params["transition"])

    return data


async def execute_scene_action(
    action: dict,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None = None,
    span_collector=None,
) -> dict:
    """Resolve an entity, call a scene HA service, and verify the result.

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
    if action_name in ("query_scene", "list_scenes"):
        return await _handle_scene_read_action(
            action_name, entity_query, ha_client, entity_index, entity_matcher, agent_id,
            span_collector=span_collector,
        )

    # Validate action name
    mapping = _SCENE_ACTION_MAP.get(action_name)
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
    service_data = _build_scene_service_data(action)

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
        "speech": f"Done, {friendly_name} has been activated.",
    }


# ---------------------------------------------------------------------------
# Read-only scene action handlers
# ---------------------------------------------------------------------------

async def _query_scene(
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
                "speech": f"Could not find a scene matching '{entity_query}'.",
                "cacheable": False}

    return {"success": True, "entity_id": entity_id, "new_state": None,
            "speech": f"Scene found: {friendly_name} ({entity_id}).",
            "cacheable": False}


async def _list_scenes(ha_client: Any) -> dict:
    try:
        states = await ha_client.get_states()
    except Exception as exc:
        logger.error("Failed to fetch states for list_scenes", exc_info=True)
        return {"success": False, "entity_id": "", "new_state": None,
                "speech": f"Failed to list scenes: {exc}"}

    scenes = [s for s in states if s.get("entity_id", "").startswith("scene.")]

    if not scenes:
        return {"success": True, "entity_id": "", "new_state": None,
                "speech": "No scenes found."}

    names = []
    for s in scenes:
        name = s.get("attributes", {}).get("friendly_name", s.get("entity_id", ""))
        names.append(name)

    speech = f"Available scenes ({len(names)}): {', '.join(names)}."
    return {"success": True, "entity_id": "", "new_state": None, "speech": speech,
            "cacheable": False}


async def _handle_scene_read_action(
    action_name: str,
    entity_query: str,
    ha_client: Any,
    entity_index: Any,
    entity_matcher: Any,
    agent_id: str | None,
    span_collector=None,
) -> dict:
    if action_name == "query_scene":
        return await _query_scene(entity_query, ha_client, entity_index, entity_matcher, agent_id,
                                  span_collector=span_collector)
    if action_name == "list_scenes":
        return await _list_scenes(ha_client)
    return {"success": False, "entity_id": "", "new_state": None,
            "speech": f"Unknown read action: {action_name}"}
