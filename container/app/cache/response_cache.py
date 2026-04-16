"""Response cache tier for full responses and cached actions."""

from __future__ import annotations

import hashlib
import logging
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
        self._store_count: int = 0
        self._eviction_interval: int = 100
        self._pending_updates: dict[str, tuple[str, dict]] = {}
        self._flush_interval: int = 5
        self._hit_since_flush: int = 0

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

    def lookup(self, query_text: str) -> tuple[str, ResponseCacheEntry | None, float | None]:
        """Query response cache.

        Returns:
            (hit_type, entry, similarity) where hit_type is "hit", "partial", or "miss".
            similarity is the best score found, even on a miss.
        """
        result = self._store.query(
            COLLECTION_RESPONSE_CACHE,
            query_texts=[query_text],
            n_results=1,
            include=["metadatas", "distances", "documents"],
        )
        if not result["ids"] or not result["ids"][0]:
            return ("miss", None, None)

        distance = result["distances"][0][0]
        similarity = 1.0 - distance

        if similarity < self._partial_threshold:
            return ("miss", None, similarity)

        meta = result["metadatas"][0][0]
        entry_id = result["ids"][0][0]

        # Buffer hit count update instead of immediate upsert
        now = datetime.now(timezone.utc).isoformat()
        hit_count = int(meta.get("hit_count", 0)) + 1
        self._pending_updates[entry_id] = (
            result["documents"][0][0],
            {**meta, "last_accessed": now, "hit_count": str(hit_count)},
        )
        self._hit_since_flush += 1
        if self._hit_since_flush >= self._flush_interval:
            self._flush_pending_updates()

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
        return (hit_type, entry, similarity)

    def store(self, entry: ResponseCacheEntry) -> None:
        """Store a new response cache entry."""
        self._store_count += 1
        if self._store_count >= self._eviction_interval:
            self._store_count = 0
            self._enforce_lru()
        now = datetime.now(timezone.utc).isoformat()
        entry_id = hashlib.sha256(entry.query_text.encode()).hexdigest()[:16]
        self._flush_pending_updates()
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
        self._flush_pending_updates()
        count = self._store.count(COLLECTION_RESPONSE_CACHE)
        if count <= self._max_entries:
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
            for i in range(0, len(to_delete), 500):
                self._store.delete(COLLECTION_RESPONSE_CACHE, ids=to_delete[i:i+500])
            logger.info("Response cache LRU evicted %d entries", len(to_delete))

    def _flush_pending_updates(self) -> None:
        """Batch-flush pending hit count updates to ChromaDB (metadata only)."""
        if not self._pending_updates:
            return
        ids = list(self._pending_updates.keys())
        metas = [self._pending_updates[i][1] for i in ids]
        try:
            self._store.update_metadata(COLLECTION_RESPONSE_CACHE, ids=ids, metadatas=metas)
        except Exception:
            logger.warning("Failed to flush response cache hit updates", exc_info=True)
        self._pending_updates.clear()
        self._hit_since_flush = 0

    def flush_pending(self) -> None:
        """Public flush for shutdown hook."""
        self._flush_pending_updates()

    def get_stats(self) -> dict:
        """Return response cache stats."""
        return {
            "count": self._store.count(COLLECTION_RESPONSE_CACHE),
            "max_entries": self._max_entries,
            "hit_threshold": self._hit_threshold,
            "partial_threshold": self._partial_threshold,
        }

    @staticmethod
    def _is_readonly_action(cached_action_str: str) -> bool:
        """Check if a cached_action JSON string represents a read-only service call."""
        if not cached_action_str:
            return True  # No action = read-only (sensor query, status check, etc.)
        try:
            import json
            data = json.loads(cached_action_str)
            service = data.get("service", "")
            # "sensor/query_status" -> action part is "query_status"
            action = service.split("/", 1)[1] if "/" in service else ""
            return action.startswith(("query_", "list_"))
        except (json.JSONDecodeError, IndexError, TypeError):
            return False

    def purge_readonly_entries(self) -> int:
        """Remove stale read-only response cache entries.

        Purges entries with no cached action OR whose cached action is a
        read-only service call (query_*, list_*).

        Returns the number of purged entries.
        """
        all_data = self._store.get(
            COLLECTION_RESPONSE_CACHE,
            include=["metadatas"],
        )
        if not all_data["ids"]:
            return 0
        to_delete = []
        for entry_id, meta in zip(all_data["ids"], all_data["metadatas"]):
            if self._is_readonly_action(meta.get("cached_action", "")):
                to_delete.append(entry_id)
        if to_delete:
            for i in range(0, len(to_delete), 500):
                self._store.delete(COLLECTION_RESPONSE_CACHE, ids=to_delete[i:i + 500])
        return len(to_delete)
