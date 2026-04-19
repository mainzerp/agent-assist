"""Helpers for converting Home Assistant state payloads into entity index entries."""

from __future__ import annotations

from typing import Any

from app.models.entity_index import EntityIndexEntry


def state_to_entity_index_entry(
    state: dict[str, Any],
    *,
    entity_id: str | None = None,
) -> EntityIndexEntry:
    """Convert a Home Assistant state payload into an EntityIndexEntry."""
    resolved_entity_id = entity_id or state.get("entity_id", "")
    attrs = state.get("attributes", {}) or {}
    domain = resolved_entity_id.split(".")[0] if "." in resolved_entity_id else ""
    return EntityIndexEntry(
        entity_id=resolved_entity_id,
        friendly_name=attrs.get("friendly_name", ""),
        domain=domain,
        area=attrs.get("area_id"),
        device_class=attrs.get("device_class"),
        aliases=[],
    )


def parse_ha_states(states: list[dict[str, Any]]) -> list[EntityIndexEntry]:
    """Convert a Home Assistant states snapshot into index entries."""
    return [state_to_entity_index_entry(state) for state in states]