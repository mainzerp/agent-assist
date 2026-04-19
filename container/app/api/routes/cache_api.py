"""Cache management admin API endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from app.cache.vector_store import (
    COLLECTION_RESPONSE_CACHE,
    COLLECTION_ROUTING_CACHE,
)
from app.security.auth import require_admin_session

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin/cache",
    tags=["admin-cache"],
    dependencies=[Depends(require_admin_session)],
)


class FlushRequest(BaseModel):
    tier: str | None = None  # "routing", "response", or None for all


@router.get("/stats")
async def get_cache_stats(request: Request):
    """Cache stats per tier."""
    cache_manager = request.app.state.cache_manager
    if not cache_manager:
        return {"routing": {}, "response": {}, "status": "not_initialized"}
    try:
        stats = cache_manager.get_stats()
        return stats
    except Exception as exc:
        logger.warning("Failed to get cache stats", exc_info=True)
        return {"error": str(exc)}


@router.get("/entries")
async def browse_cache_entries(
    request: Request,
    tier: str = Query("routing", pattern="^(routing|response)$"),
    search: str | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
):
    """Browse/search cache entries by tier."""
    cache_manager = request.app.state.cache_manager
    if not cache_manager:
        return {"entries": [], "total": 0}

    vector_store = cache_manager._vector_store
    collection_name = COLLECTION_ROUTING_CACHE if tier == "routing" else COLLECTION_RESPONSE_CACHE

    try:
        total = vector_store.count(collection_name)
        if total == 0:
            return {"entries": [], "total": 0, "page": page, "per_page": per_page}

        # When not searching, use limit/offset to avoid loading all entries
        if not search:
            offset_val = (page - 1) * per_page
            data = vector_store.get(
                collection_name,
                include=["metadatas", "documents"],
                limit=per_page,
                offset=offset_val,
            )
            entries = []
            for i, doc_id in enumerate(data["ids"]):
                meta = data["metadatas"][i]
                document = data["documents"][i] if data.get("documents") else ""
                entry = {"id": doc_id, "document": document, **meta}
                entries.append(entry)

            return {
                "entries": entries,
                "total": total,
                "page": page,
                "per_page": per_page,
                "pages": (total + per_page - 1) // per_page if per_page else 0,
            }

        # Search requires loading all entries for text filtering
        data = vector_store.get(
            collection_name,
            include=["metadatas", "documents"],
        )
        entries = []
        for i, doc_id in enumerate(data["ids"]):
            meta = data["metadatas"][i]
            document = data["documents"][i] if data.get("documents") else ""
            entry = {"id": doc_id, "document": document, **meta}
            entries.append(entry)

        # Filter by search text
        if search:
            search_lower = search.lower()
            entries = [
                e
                for e in entries
                if search_lower in (e.get("document") or "").lower()
                or search_lower in str(e.get("agent_id", "")).lower()
            ]

        # Sort by last_accessed descending
        entries.sort(key=lambda e: e.get("last_accessed", ""), reverse=True)

        # Paginate
        filtered_total = len(entries)
        offset = (page - 1) * per_page
        entries = entries[offset : offset + per_page]

        return {
            "entries": entries,
            "total": filtered_total,
            "page": page,
            "per_page": per_page,
            "pages": (filtered_total + per_page - 1) // per_page if per_page else 0,
        }
    except Exception as exc:
        logger.warning("Failed to browse cache entries", exc_info=True)
        return {"entries": [], "total": 0, "error": str(exc)}


@router.post("/flush")
async def flush_cache(request: Request, payload: FlushRequest):
    """Flush cache tier(s)."""
    cache_manager = request.app.state.cache_manager
    if not cache_manager:
        return {"status": "error", "detail": "Cache not initialized"}

    tier = payload.tier
    if tier and tier not in ("routing", "response"):
        return {"status": "error", "detail": "Invalid tier. Use 'routing', 'response', or omit for all."}

    try:
        cache_manager.flush(tier)
        return {"status": "ok", "flushed": tier or "all"}
    except Exception as exc:
        logger.warning("Failed to flush cache", exc_info=True)
        return {"status": "error", "detail": str(exc)}
