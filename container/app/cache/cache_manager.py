"""Unified cache manager with invalidation and threshold management."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.cache.routing_cache import RoutingCache
from app.cache.response_cache import ResponseCache
from app.cache.vector_store import VectorStore, get_vector_store, COLLECTION_ROUTING_CACHE, COLLECTION_RESPONSE_CACHE
from app.models.cache import RoutingCacheEntry, ResponseCacheEntry, CachedAction
from app.analytics.collector import track_cache_event, track_rewrite

logger = logging.getLogger(__name__)


@dataclass
class CacheResult:
    """Result from cache manager process() call."""
    hit_type: str  # "response_hit", "response_partial", "routing_hit", "miss"
    agent_id: str | None = None
    response_text: str | None = None
    cached_action: CachedAction | None = None
    entry: ResponseCacheEntry | RoutingCacheEntry | None = None
    condensed_task: str | None = None


class CacheManager:
    """Coordinates routing and response caches for the orchestrator flow."""

    def __init__(
        self,
        vector_store: VectorStore,
        rewrite_agent=None,
    ) -> None:
        self._vector_store = vector_store
        self._routing_cache = RoutingCache(vector_store)
        self._response_cache = ResponseCache(vector_store)
        self._rewrite_agent = rewrite_agent
        self._rewrite_enabled: bool = False

    async def initialize(self) -> None:
        """Load config for both cache tiers."""
        await self._routing_cache.load_config()
        await self._response_cache.load_config()
        from app.db.repository import SettingsRepository
        personality = await SettingsRepository.get_value("personality.prompt", "")
        self._rewrite_enabled = bool(personality.strip())

    async def reload_config(self) -> None:
        """Hot-reload thresholds and rewrite setting from DB."""
        await self._routing_cache.reload_config()
        await self._response_cache.reload_config()
        from app.db.repository import SettingsRepository
        if self._rewrite_agent:
            personality = await SettingsRepository.get_value("personality.prompt", "")
            self._rewrite_enabled = bool(personality.strip())

    async def process(self, query_text: str) -> CacheResult:
        """Check both cache tiers in order: response first, then routing.

        Returns a CacheResult indicating what was found.
        """
        try:
            result = self._process_inner(query_text)
            # Track cache event
            await track_cache_event(
                tier="response" if result.hit_type.startswith("response") else "routing",
                hit_type=result.hit_type,
                agent_id=result.agent_id,
            )
            # Apply rewrite on full response cache hits
            if (
                result.hit_type == "response_hit"
                and self._rewrite_enabled
                and self._rewrite_agent
                and result.response_text
            ):
                t0 = time.perf_counter()
                try:
                    result.response_text = await self._rewrite_agent.rewrite(result.response_text)
                    rewrite_ms = (time.perf_counter() - t0) * 1000
                    await track_rewrite(latency_ms=rewrite_ms, success=True)
                except Exception:
                    rewrite_ms = (time.perf_counter() - t0) * 1000
                    await track_rewrite(latency_ms=rewrite_ms, success=False)
                    logger.warning("Rewrite failed, using original cached text", exc_info=True)
            return result
        except Exception:
            logger.warning("Cache lookup failed, bypassing cache", exc_info=True)
            return CacheResult(hit_type="miss")

    def _process_inner(self, query_text: str) -> CacheResult:
        """Internal cache lookup logic."""
        # 1. Check response cache first (higher value hit)
        hit_type, resp_entry = self._response_cache.lookup(query_text)
        if hit_type == "hit":
            return CacheResult(
                hit_type="response_hit",
                agent_id=resp_entry.agent_id,
                response_text=resp_entry.response_text,
                cached_action=resp_entry.cached_action,
                entry=resp_entry,
            )
        if hit_type == "partial":
            return CacheResult(
                hit_type="response_partial",
                agent_id=resp_entry.agent_id,
                response_text=resp_entry.response_text,
                cached_action=resp_entry.cached_action,
                entry=resp_entry,
            )

        # 2. Check routing cache
        routing_entry = self._routing_cache.lookup(query_text)
        if routing_entry:
            return CacheResult(
                hit_type="routing_hit",
                agent_id=routing_entry.agent_id,
                entry=routing_entry,
                condensed_task=routing_entry.condensed_task,
            )

        # 3. Complete miss
        return CacheResult(hit_type="miss")

    def store_routing(self, query_text: str, agent_id: str, confidence: float, condensed_task: str = "") -> None:
        """Store a routing decision after an agent handles a request."""
        self._routing_cache.store(query_text, agent_id, confidence, condensed_task)

    def store_response(self, entry: ResponseCacheEntry) -> None:
        """Store a full response after successful execution."""
        self._response_cache.store(entry)

    def invalidate_response(self, entry_id: str) -> None:
        """Reactive invalidation -- remove a response entry on action failure."""
        self._response_cache.invalidate(entry_id)

    def flush(self, tier: str | None = None) -> None:
        """Clear one or both cache tiers. Used by admin UI.

        Args:
            tier: "routing", "response", or None (both).
        """
        if tier is None or tier == "routing":
            count = self._vector_store.count(COLLECTION_ROUTING_CACHE)
            if count > 0:
                all_data = self._vector_store.get(
                    COLLECTION_ROUTING_CACHE, include=[]
                )
                if all_data["ids"]:
                    self._vector_store.delete(COLLECTION_ROUTING_CACHE, ids=all_data["ids"])
            logger.info("Routing cache flushed")
        if tier is None or tier == "response":
            count = self._vector_store.count(COLLECTION_RESPONSE_CACHE)
            if count > 0:
                all_data = self._vector_store.get(
                    COLLECTION_RESPONSE_CACHE, include=[]
                )
                if all_data["ids"]:
                    self._vector_store.delete(COLLECTION_RESPONSE_CACHE, ids=all_data["ids"])
            logger.info("Response cache flushed")

    def get_stats(self) -> dict:
        """Return combined stats for both tiers."""
        return {
            "routing": self._routing_cache.get_stats(),
            "response": self._response_cache.get_stats(),
        }
