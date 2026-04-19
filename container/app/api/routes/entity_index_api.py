"""Entity index admin API endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.security.auth import require_admin_session
from app.cache.vector_store import COLLECTION_ENTITY_INDEX
from app.cache.embedding import get_embedding_info
from app.db.repository import EntityVisibilityRepository, SettingsRepository
from app.entity.ingest import parse_ha_states
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


@router.get("/match-preview")
async def match_preview(
    request: Request,
    q: str = Query(..., min_length=1, max_length=200, description="Entity query"),
    agent_id: str | None = Query(
        default=None,
        description="Optional agent id for visibility/domain gating",
    ),
) -> dict[str, Any]:
    """Preview how the entity resolver + hybrid matcher handle a query.

    Surfaces exactly what each agent type receives:

    * ``deterministic`` -- the output of
      :func:`app.agents.action_executor._resolve_light_entity`, which is
      the path ``light-agent`` (and other light/switch/sensor executors)
      walk before reaching the hybrid matcher. Includes the chosen
      ``entity_id``, ``friendly_name``, ``resolution_path`` and whether
      the resolved id passes the light-executor domain gate.
    * ``hybrid`` -- the top candidates from
      :meth:`app.entity.matcher.EntityMatcher.match`, which is what
      non-light executors (climate / media / security / …) use directly.
      Each candidate carries its ``score`` and per-signal scores so the
      operator can see why a result was (or was not) picked.
    * ``visibility`` -- a compact summary of the visibility rules the
      selected ``agent_id`` is bound to, plus the live visible-entity
      count for that agent.

    The endpoint is read-only. No HA service calls are made.
    """
    entity_index = getattr(request.app.state, "entity_index", None)
    entity_matcher = getattr(request.app.state, "entity_matcher", None)
    if entity_index is None:
        raise HTTPException(
            status_code=503, detail="Entity index not initialized",
        )

    query = q.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be empty")

    agent = (agent_id or "").strip() or None

    # -----------------------------------------------------------------
    # Deterministic light-agent resolver (same code path as execute_action)
    # -----------------------------------------------------------------
    deterministic: dict[str, Any] = {
        "entity_id": None,
        "friendly_name": query,
        "speech": None,
        "metadata": {
            "query": query,
            "resolution_path": "not_attempted",
            "match_count": 0,
        },
        "domain_allowed": False,
        "error": None,
    }
    try:
        from app.agents.action_executor import (
            _resolve_light_entity,
            _validate_domain,
        )

        if entity_matcher is None:
            deterministic["error"] = "entity_matcher not initialized"
        else:
            resolution = await _resolve_light_entity(
                query, entity_index, entity_matcher, agent,
            )
            deterministic.update({
                "entity_id": resolution.get("entity_id"),
                "friendly_name": resolution.get("friendly_name") or query,
                "speech": resolution.get("speech"),
                "metadata": resolution.get("metadata") or deterministic["metadata"],
            })
            resolved_id = deterministic["entity_id"]
            deterministic["domain_allowed"] = bool(
                resolved_id and _validate_domain(resolved_id)
            )
    except Exception as exc:
        logger.warning("match-preview: deterministic resolution failed", exc_info=True)
        deterministic["error"] = str(exc)

    # -----------------------------------------------------------------
    # Hybrid matcher (what every non-light executor sees directly)
    # -----------------------------------------------------------------
    hybrid: list[dict[str, Any]] = []
    hybrid_error: str | None = None
    try:
        if entity_matcher is None:
            hybrid_error = "entity_matcher not initialized"
        else:
            matches = await entity_matcher.match(query, agent_id=agent)
            for match in matches:
                entity_id = getattr(match, "entity_id", "") or ""
                domain = entity_id.split(".", 1)[0] if "." in entity_id else ""
                entry = None
                try:
                    if entity_index is not None and hasattr(entity_index, "get_by_id"):
                        entry = entity_index.get_by_id(entity_id)
                except Exception:
                    entry = None
                hybrid.append({
                    "entity_id": entity_id,
                    "friendly_name": getattr(match, "friendly_name", "") or entity_id,
                    "domain": domain,
                    "area": getattr(entry, "area", None) if entry else None,
                    "score": round(float(getattr(match, "score", 0.0) or 0.0), 4),
                    "signal_scores": {
                        k: round(float(v), 4)
                        for k, v in (getattr(match, "signal_scores", {}) or {}).items()
                    },
                })
    except Exception as exc:
        logger.warning("match-preview: hybrid matcher failed", exc_info=True)
        hybrid_error = str(exc)

    # -----------------------------------------------------------------
    # Visibility summary for the selected agent
    # -----------------------------------------------------------------
    visibility: dict[str, Any] = {
        "agent_id": agent,
        "rules": [],
        "visible_entity_count": None,
        "total_entity_count": None,
    }
    try:
        if entity_index is not None and hasattr(entity_index, "list_entries"):
            total_entries = entity_index.list_entries()
            visibility["total_entity_count"] = len(total_entries)
            if agent and entity_matcher is not None and hasattr(
                entity_matcher, "filter_visible_results",
            ):
                from app.entity.matcher import MatchResult

                probe = [
                    MatchResult(
                        entity_id=e.entity_id,
                        friendly_name=e.friendly_name,
                        score=1.0,
                    )
                    for e in total_entries
                ]
                visible = await entity_matcher.filter_visible_results(agent, probe)
                visibility["visible_entity_count"] = len(visible)
        if agent:
            rules = await EntityVisibilityRepository.get_rules(agent)
            visibility["rules"] = rules
    except Exception:
        logger.debug(
            "match-preview: visibility summary failed for agent_id=%s",
            agent, exc_info=True,
        )

    return {
        "query": query,
        "agent_id": agent,
        "deterministic": deterministic,
        "hybrid": hybrid,
        "hybrid_error": hybrid_error,
        "visibility": visibility,
    }


@router.post("/refresh")
async def refresh_entity_index(request: Request):
    """Force refresh entity index from Home Assistant."""
    entity_index = request.app.state.entity_index
    ha_client = request.app.state.ha_client
    if not entity_index or not ha_client:
        return {"status": "error", "detail": "Entity index or HA client not initialized"}

    try:
        states = await ha_client.get_states()
        entities = parse_ha_states(states)
        entity_index.refresh(entities)
        return {
            "status": "ok",
            "count": len(entities),
        }
    except Exception as exc:
        logger.warning("Failed to refresh entity index", exc_info=True)
        return {"status": "error", "detail": str(exc)}
