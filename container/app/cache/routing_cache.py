"""Routing cache tier for intent-to-agent routing decisions."""

from __future__ import annotations

import hashlib
import logging
import threading
from datetime import datetime, timezone

from app.cache.vector_store import VectorStore, COLLECTION_ROUTING_CACHE
from app.db.repository import SettingsRepository
from app.models.cache import RoutingCacheEntry

logger = logging.getLogger(__name__)


class RoutingCache:
    """Routing cache tier mapping user text to agent routing decisions."""

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store
        self._threshold: float = 0.92
        self._max_entries: int = 50000
        self._store_count: int = 0
        self._eviction_interval: int = 100
        self._pending_updates: dict[str, tuple[str, dict]] = {}
        self._flush_interval: int = 5
        self._hit_since_flush: int = 0
        # FLOW-MED-1: both lookup() and the flush path run via
        # asyncio.to_thread on worker threads; the in-memory counters
        # and pending map must be mutated under a lock. I/O against
        # the underlying vector store happens OUTSIDE the lock.
        self._mutation_lock = threading.Lock()

    async def load_config(self) -> None:
        """Load thresholds from settings table."""
        self._threshold = float(
            await SettingsRepository.get_value("cache.routing.threshold", "0.92")
        )
        self._max_entries = int(
            await SettingsRepository.get_value("cache.routing.max_entries", "50000")
        )

    async def reload_config(self) -> None:
        """Reload thresholds from DB without restart."""
        await self.load_config()

    def lookup(
        self, query_text: str, *, language: str = "en",
    ) -> tuple[RoutingCacheEntry | None, float | None]:
        """Query routing cache. Returns (entry, similarity).

        ChromaDB returns distance (0=identical). Similarity = 1 - distance.
        Returns the best similarity score even on a miss.

        FLOW-HIGH-4: scopes the vector query to entries with matching
        language metadata so cross-language hits cannot leak.
        """
        lang = (language or "en").lower()
        result = self._store.query(
            COLLECTION_ROUTING_CACHE,
            query_texts=[query_text],
            n_results=1,
            where={"language": lang},
            include=["metadatas", "distances", "documents"],
        )
        if not result["ids"] or not result["ids"][0]:
            return (None, None)

        distance = result["distances"][0][0]
        similarity = 1.0 - distance

        if similarity < self._threshold:
            return (None, similarity)

        meta = result["metadatas"][0][0]
        entry_id = result["ids"][0][0]
        now = datetime.now(timezone.utc).isoformat()
        hit_count = int(meta.get("hit_count", 0)) + 1
        with self._mutation_lock:
            self._pending_updates[entry_id] = (
                result["documents"][0][0],
                {**meta, "last_accessed": now, "hit_count": str(hit_count)},
            )
            self._hit_since_flush += 1
            should_flush = self._hit_since_flush >= self._flush_interval
        if should_flush:
            self._flush_pending_updates()

        return (RoutingCacheEntry(
            query_text=result["documents"][0][0],
            agent_id=meta["agent_id"],
            confidence=similarity,
            hit_count=hit_count,
            condensed_task=meta.get("condensed_task"),
            created_at=meta.get("created_at"),
            last_accessed=now,
            language=meta.get("language", "en"),
        ), similarity)

    def store(
        self, query_text: str, agent_id: str, confidence: float,
        condensed_task: str = "", *, language: str = "en",
    ) -> None:
        """Store a new routing decision in the cache."""
        self._store_count += 1
        if self._store_count >= self._eviction_interval:
            self._store_count = 0
            self._enforce_lru()
        now = datetime.now(timezone.utc).isoformat()
        lang = (language or "en").lower()
        # FLOW-HIGH-4: prefix the key with language so identical text
        # in different languages produces distinct entries.
        entry_id = hashlib.sha256(
            f"{lang}\n{query_text}".encode()
        ).hexdigest()[:16]
        self._flush_pending_updates()
        self._store.upsert(
            COLLECTION_ROUTING_CACHE,
            ids=[entry_id],
            documents=[query_text],
            metadatas=[{
                "agent_id": agent_id,
                "confidence": str(confidence),
                "hit_count": "0",
                "condensed_task": condensed_task,
                "created_at": now,
                "last_accessed": now,
                "language": lang,
            }],
        )

    def _enforce_lru(self) -> None:
        """Evict oldest entries if collection exceeds max_entries."""
        self._flush_pending_updates()
        count = self._store.count(COLLECTION_ROUTING_CACHE)
        if count <= self._max_entries:
            return
        overage = count - self._max_entries + int(self._max_entries * 0.1)
        all_data = self._store.get(
            COLLECTION_ROUTING_CACHE,
            include=["metadatas"],
        )
        if not all_data["ids"]:
            return
        paired = list(zip(all_data["ids"], all_data["metadatas"]))
        paired.sort(key=lambda p: p[1].get("last_accessed", ""))
        to_delete = [p[0] for p in paired[:overage]]
        if to_delete:
            for i in range(0, len(to_delete), 500):
                self._store.delete(COLLECTION_ROUTING_CACHE, ids=to_delete[i:i+500])
            logger.info("Routing cache LRU evicted %d entries", len(to_delete))

    def _flush_pending_updates(self) -> None:
        """Batch-flush pending hit count updates to ChromaDB (metadata only)."""
        with self._mutation_lock:
            if not self._pending_updates:
                return
            pending = self._pending_updates
            self._pending_updates = {}
            self._hit_since_flush = 0
        ids = list(pending.keys())
        metas = [pending[i][1] for i in ids]
        try:
            self._store.update_metadata(COLLECTION_ROUTING_CACHE, ids=ids, metadatas=metas)
        except Exception:
            logger.warning("Failed to flush routing cache hit updates", exc_info=True)

    def flush_pending(self) -> None:
        """Public flush for shutdown hook."""
        self._flush_pending_updates()

    def purge_entries_without_language(self) -> int:
        """Remove entries missing the ``language`` metadata field.

        FLOW-HIGH-4 migration: pre-0.18.0 entries have no ``language``
        field and their keys were not language-scoped. Since the new
        lookup filters on ``language`` they would be unreachable
        anyway; purge them to keep the collection tidy. Returns the
        number of purged entries.
        """
        all_data = self._store.get(
            COLLECTION_ROUTING_CACHE,
            include=["metadatas"],
        )
        if not all_data["ids"]:
            return 0
        to_delete = [
            eid
            for eid, meta in zip(all_data["ids"], all_data["metadatas"])
            if not (meta or {}).get("language")
        ]
        if to_delete:
            for i in range(0, len(to_delete), 500):
                self._store.delete(COLLECTION_ROUTING_CACHE, ids=to_delete[i:i+500])
            logger.info(
                "Routing cache: purged %d pre-0.18.0 entries without language metadata",
                len(to_delete),
            )
        return len(to_delete)

    def get_stats(self) -> dict:
        """Return routing cache stats."""
        return {
            "count": self._store.count(COLLECTION_ROUTING_CACHE),
            "max_entries": self._max_entries,
            "threshold": self._threshold,
        }
