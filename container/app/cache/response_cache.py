"""Response cache tier for full responses and cached actions."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from app.cache.vector_store import VectorStore, COLLECTION_RESPONSE_CACHE
from app.db.repository import SettingsRepository
from app.models.cache import ResponseCacheEntry, CachedAction

logger = logging.getLogger(__name__)


class ResponseCache:
    """Response cache tier for full responses and cached actions.

    Threshold levels:
    - > 0.95 similarity: HIT (return cached response + action)
    - 0.80-0.95 similarity: PARTIAL (return cached response for rewrite consideration)
    - < 0.80 similarity: MISS
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store
        self._hit_threshold: float = 0.95
        self._partial_threshold: float = 0.80
        self._max_entries: int = 20000

    async def load_config(self) -> None:
        """Load thresholds from settings table."""
        self._hit_threshold = float(
            await SettingsRepository.get_value("cache.response.threshold", "0.95")
        )
        self._partial_threshold = float(
            await SettingsRepository.get_value("cache.response.partial_threshold", "0.80")
        )
        self._max_entries = int(
            await SettingsRepository.get_value("cache.response.max_entries", "20000")
        )

    async def reload_config(self) -> None:
        """Reload thresholds from DB without restart."""
        await self.load_config()

    def lookup(self, query_text: str) -> tuple[str, ResponseCacheEntry | None]:
        """Query response cache.

        Returns:
            (hit_type, entry) where hit_type is "hit", "partial", or "miss".
        """
        result = self._store.query(
            COLLECTION_RESPONSE_CACHE,
            query_texts=[query_text],
            n_results=1,
            include=["metadatas", "distances", "documents"],
        )
        if not result["ids"] or not result["ids"][0]:
            return ("miss", None)

        distance = result["distances"][0][0]
        similarity = 1.0 - distance

        if similarity < self._partial_threshold:
            return ("miss", None)

        meta = result["metadatas"][0][0]
        entry_id = result["ids"][0][0]

        # Update last_accessed and hit_count
        now = datetime.now(timezone.utc).isoformat()
        hit_count = int(meta.get("hit_count", 0)) + 1
        self._store.upsert(
            COLLECTION_RESPONSE_CACHE,
            ids=[entry_id],
            documents=[result["documents"][0][0]],
            metadatas=[{**meta, "last_accessed": now, "hit_count": str(hit_count)}],
        )

        # Parse cached action from metadata
        cached_action = None
        action_json = meta.get("cached_action")
        if action_json:
            cached_action = CachedAction.model_validate_json(action_json)

        # Parse entity_ids
        entity_ids_str = meta.get("entity_ids", "")
        entity_ids = entity_ids_str.split(",") if entity_ids_str else []

        entry = ResponseCacheEntry(
            query_text=result["documents"][0][0],
            response_text=meta.get("response_text", ""),
            agent_id=meta.get("agent_id", ""),
            cached_action=cached_action,
            confidence=similarity,
            hit_count=hit_count,
            entity_ids=entity_ids,
            created_at=meta.get("created_at"),
            last_accessed=now,
        )

        hit_type = "hit" if similarity >= self._hit_threshold else "partial"
        return (hit_type, entry)

    def store(self, entry: ResponseCacheEntry) -> None:
        """Store a new response cache entry."""
        self._enforce_lru()
        now = datetime.now(timezone.utc).isoformat()
        entry_id = str(uuid.uuid4())
        meta = {
            "response_text": entry.response_text,
            "agent_id": entry.agent_id,
            "confidence": str(entry.confidence),
            "hit_count": "0",
            "entity_ids": ",".join(entry.entity_ids),
            "created_at": now,
            "last_accessed": now,
        }
        if entry.cached_action:
            meta["cached_action"] = entry.cached_action.model_dump_json()
        else:
            meta["cached_action"] = ""

        self._store.upsert(
            COLLECTION_RESPONSE_CACHE,
            ids=[entry_id],
            documents=[entry.query_text],
            metadatas=[meta],
        )

    def invalidate(self, entry_id: str) -> None:
        """Remove a specific entry (reactive invalidation on action failure)."""
        self._store.delete(COLLECTION_RESPONSE_CACHE, ids=[entry_id])
        logger.info("Response cache entry invalidated: %s", entry_id)

    def _enforce_lru(self) -> None:
        """Evict oldest entries if collection exceeds max_entries."""
        count = self._store.count(COLLECTION_RESPONSE_CACHE)
        if count < self._max_entries:
            return
        overage = count - self._max_entries + int(self._max_entries * 0.1)
        all_data = self._store.get(
            COLLECTION_RESPONSE_CACHE,
            include=["metadatas"],
        )
        if not all_data["ids"]:
            return
        paired = list(zip(all_data["ids"], all_data["metadatas"]))
        paired.sort(key=lambda p: p[1].get("last_accessed", ""))
        to_delete = [p[0] for p in paired[:overage]]
        if to_delete:
            self._store.delete(COLLECTION_RESPONSE_CACHE, ids=to_delete)
            logger.info("Response cache LRU evicted %d entries", len(to_delete))

    def get_stats(self) -> dict:
        """Return response cache stats."""
        return {
            "count": self._store.count(COLLECTION_RESPONSE_CACHE),
            "max_entries": self._max_entries,
            "hit_threshold": self._hit_threshold,
            "partial_threshold": self._partial_threshold,
        }
