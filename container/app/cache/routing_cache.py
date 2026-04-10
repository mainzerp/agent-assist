"""Routing cache tier for intent-to-agent routing decisions."""

from __future__ import annotations

import logging
import uuid
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

    def lookup(self, query_text: str) -> RoutingCacheEntry | None:
        """Query routing cache. Returns entry if cosine similarity > threshold.

        ChromaDB returns distance (0=identical). Similarity = 1 - distance.
        """
        result = self._store.query(
            COLLECTION_ROUTING_CACHE,
            query_texts=[query_text],
            n_results=1,
            include=["metadatas", "distances", "documents"],
        )
        if not result["ids"] or not result["ids"][0]:
            return None

        distance = result["distances"][0][0]
        similarity = 1.0 - distance

        if similarity < self._threshold:
            return None

        meta = result["metadatas"][0][0]
        # Update last_accessed and hit_count
        entry_id = result["ids"][0][0]
        now = datetime.now(timezone.utc).isoformat()
        hit_count = int(meta.get("hit_count", 0)) + 1
        self._store.upsert(
            COLLECTION_ROUTING_CACHE,
            ids=[entry_id],
            documents=[result["documents"][0][0]],
            metadatas=[{**meta, "last_accessed": now, "hit_count": str(hit_count)}],
        )

        return RoutingCacheEntry(
            query_text=result["documents"][0][0],
            agent_id=meta["agent_id"],
            confidence=similarity,
            hit_count=hit_count,
            created_at=meta.get("created_at"),
            last_accessed=now,
        )

    def store(self, query_text: str, agent_id: str, confidence: float) -> None:
        """Store a new routing decision in the cache."""
        self._enforce_lru()
        now = datetime.now(timezone.utc).isoformat()
        entry_id = str(uuid.uuid4())
        self._store.upsert(
            COLLECTION_ROUTING_CACHE,
            ids=[entry_id],
            documents=[query_text],
            metadatas=[{
                "agent_id": agent_id,
                "confidence": str(confidence),
                "hit_count": "0",
                "created_at": now,
                "last_accessed": now,
            }],
        )

    def _enforce_lru(self) -> None:
        """Evict oldest entries if collection exceeds max_entries."""
        count = self._store.count(COLLECTION_ROUTING_CACHE)
        if count < self._max_entries:
            return
        # Fetch all entries sorted by last_accessed, delete oldest 10%
        overage = count - self._max_entries + int(self._max_entries * 0.1)
        all_data = self._store.get(
            COLLECTION_ROUTING_CACHE,
            include=["metadatas"],
        )
        if not all_data["ids"]:
            return
        # Sort by last_accessed ascending
        paired = list(zip(all_data["ids"], all_data["metadatas"]))
        paired.sort(key=lambda p: p[1].get("last_accessed", ""))
        to_delete = [p[0] for p in paired[:overage]]
        if to_delete:
            self._store.delete(COLLECTION_ROUTING_CACHE, ids=to_delete)
            logger.info("Routing cache LRU evicted %d entries", len(to_delete))

    def get_stats(self) -> dict:
        """Return routing cache stats."""
        return {
            "count": self._store.count(COLLECTION_ROUTING_CACHE),
            "max_entries": self._max_entries,
            "threshold": self._threshold,
        }
