"""Cache entry models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RoutingCacheEntry(BaseModel):
    """Entry in the routing cache tier."""

    query_text: str
    agent_id: str
    confidence: float
    hit_count: int = 0
    condensed_task: str | None = None
    created_at: str | None = None
    last_accessed: str | None = None
    # FLOW-HIGH-4: detected language at store time. Lookup filters on
    # this so cross-language hits cannot leak.
    language: str = "en"


class CachedAction(BaseModel):
    """A cached HA service call for direct execution on cache hit."""

    service: str = Field(..., description="HA service (e.g., light/turn_on)")
    entity_id: str
    service_data: dict = Field(default_factory=dict)


class ResponseCacheEntry(BaseModel):
    """Entry in the response cache tier."""

    query_text: str
    response_text: str
    agent_id: str
    cached_action: CachedAction | None = None
    confidence: float
    hit_count: int = 0
    entity_ids: list[str] = Field(default_factory=list)
    created_at: str | None = None
    last_accessed: str | None = None
    # FLOW-HIGH-4: detected language at store time. Lookup filters on
    # this so cross-language hits cannot leak.
    language: str = "en"


# Public alias (added in 0.21.0). The class keeps the name
# ResponseCacheEntry internally to avoid churn; new callers should use
# ActionCacheEntry. The alias will remain for at least one minor.
ActionCacheEntry = ResponseCacheEntry
