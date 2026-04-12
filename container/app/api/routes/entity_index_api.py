"""Entity index admin API endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from app.security.auth import require_admin_session
from app.cache.vector_store import COLLECTION_ENTITY_INDEX
from app.cache.embedding import get_embedding_info
from app.db.repository import SettingsRepository
from app.models.entity_index import EntityIndexEntry

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin/entity-index",
    tags=["admin-entity-index"],
    dependencies=[Depends(require_admin_session)],
)


@router.get("/stats")
async def get_entity_index_stats(request: Request):
    """Entity index stats with per-domain breakdown."""
    entity_index = request.app.state.entity_index
    if not entity_index:
        return {"count": 0, "status": "not_initialized", "domains": {}, "embedding": None, "sync": {}}

    try:
        stats = entity_index.get_stats()
        count = stats.get("count", 0)

        # Get per-domain breakdown
        domains: dict[str, int] = {}
        if count > 0:
            vector_store = entity_index._store
            data = vector_store.get(
                COLLECTION_ENTITY_INDEX,
                include=["metadatas"],
            )
            for meta in data.get("metadatas", []):
                domain = meta.get("domain", "unknown")
                domains[domain] = domains.get(domain, 0) + 1

        embedding_info = await get_embedding_info()

        sync_stats = stats.get("sync", {})
        sync_interval = await SettingsRepository.get_value(
            "entity_sync.interval_minutes", "30"
        )

        return {
            "count": count,
            "last_refresh": stats.get("last_refresh"),
            "domains": domains,
            "embedding": {
                **embedding_info,
                **stats.get("embedding_status", {}),
            },
            "sync": sync_stats,
            "sync_interval_minutes": int(sync_interval or 30),
        }
    except Exception as exc:
        logger.warning("Failed to get entity index stats", exc_info=True)
        return {"count": 0, "error": str(exc), "embedding": None, "sync": {}}


@router.post("/refresh")
async def refresh_entity_index(request: Request):
    """Force refresh entity index from Home Assistant."""
    entity_index = request.app.state.entity_index
    ha_client = request.app.state.ha_client
    if not entity_index or not ha_client:
        return {"status": "error", "detail": "Entity index or HA client not initialized"}

    try:
        states = await ha_client.get_states()
        entities = []
        for state in states:
            entity_id = state.get("entity_id", "")
            attrs = state.get("attributes", {})
            domain = entity_id.split(".")[0] if "." in entity_id else ""
            entities.append(EntityIndexEntry(
                entity_id=entity_id,
                friendly_name=attrs.get("friendly_name", ""),
                domain=domain,
                area=attrs.get("area_id"),
                device_class=attrs.get("device_class"),
                aliases=[],
            ))
        entity_index.refresh(entities)
        return {
            "status": "ok",
            "count": len(entities),
        }
    except Exception as exc:
        logger.warning("Failed to refresh entity index", exc_info=True)
        return {"status": "error", "detail": str(exc)}
