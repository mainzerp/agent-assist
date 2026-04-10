"""Pre-embedded entity index using ChromaDB."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.cache.vector_store import VectorStore, COLLECTION_ENTITY_INDEX
from app.models.entity_index import EntityIndexEntry

logger = logging.getLogger(__name__)


class EntityIndex:
    """Pre-embedded entity index backed by ChromaDB."""

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store
        self._last_refresh: str | None = None

    def populate(self, entities: list[EntityIndexEntry]) -> None:
        """Bulk upsert all HA entities into the entity_index collection.

        Called at startup after fetching GET /api/states.
        """
        if not entities:
            return
        ids = [e.entity_id for e in entities]
        documents = [e.embedding_text for e in entities]
        metadatas = [
            {
                "friendly_name": e.friendly_name,
                "domain": e.domain,
                "area": e.area or "",
                "device_class": e.device_class or "",
                "aliases": ",".join(e.aliases) if e.aliases else "",
            }
            for e in entities
        ]
        self._store.upsert(
            COLLECTION_ENTITY_INDEX,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        self._last_refresh = datetime.now(timezone.utc).isoformat()
        logger.info("Entity index populated with %d entities", len(entities))

    def search(self, query: str, n_results: int = 5) -> list[tuple[EntityIndexEntry, float]]:
        """Vector similarity search. Returns list of (entry, distance) tuples.

        Lower distance = more similar (cosine distance, 0.0 = identical).
        """
        result = self._store.query(
            COLLECTION_ENTITY_INDEX,
            query_texts=[query],
            n_results=n_results,
            include=["metadatas", "distances", "documents"],
        )
        entries: list[tuple[EntityIndexEntry, float]] = []
        if not result["ids"] or not result["ids"][0]:
            return entries
        for i, eid in enumerate(result["ids"][0]):
            meta = result["metadatas"][0][i]
            distance = result["distances"][0][i]
            aliases_str = meta.get("aliases", "")
            entry = EntityIndexEntry(
                entity_id=eid,
                friendly_name=meta.get("friendly_name", ""),
                domain=meta.get("domain", ""),
                area=meta.get("area", "") or None,
                device_class=meta.get("device_class", "") or None,
                aliases=aliases_str.split(",") if aliases_str else [],
            )
            entries.append((entry, distance))
        return entries

    def add(self, entry: EntityIndexEntry) -> None:
        """Add or update a single entity."""
        self._store.upsert(
            COLLECTION_ENTITY_INDEX,
            ids=[entry.entity_id],
            documents=[entry.embedding_text],
            metadatas=[{
                "friendly_name": entry.friendly_name,
                "domain": entry.domain,
                "area": entry.area or "",
                "device_class": entry.device_class or "",
                "aliases": ",".join(entry.aliases) if entry.aliases else "",
            }],
        )

    def remove(self, entity_id: str) -> None:
        """Remove an entity from the index."""
        self._store.delete(COLLECTION_ENTITY_INDEX, ids=[entity_id])

    def get_by_id(self, entity_id: str) -> EntityIndexEntry | None:
        """Retrieve a single entity by its ID, or None if not found."""
        data = self._store.get(
            COLLECTION_ENTITY_INDEX,
            ids=[entity_id],
            include=["metadatas"],
        )
        if not data["ids"]:
            return None
        meta = data["metadatas"][0]
        aliases_str = meta.get("aliases", "")
        return EntityIndexEntry(
            entity_id=entity_id,
            friendly_name=meta.get("friendly_name", ""),
            domain=meta.get("domain", ""),
            area=meta.get("area", "") or None,
            device_class=meta.get("device_class", "") or None,
            aliases=aliases_str.split(",") if aliases_str else [],
        )

    def clear(self) -> None:
        """Remove all entities from the index."""
        count = self._store.count(COLLECTION_ENTITY_INDEX)
        if count > 0:
            all_data = self._store.get(COLLECTION_ENTITY_INDEX, include=[])
            if all_data["ids"]:
                self._store.delete(COLLECTION_ENTITY_INDEX, ids=all_data["ids"])
        logger.info("Entity index cleared")

    def refresh(self, entities: list[EntityIndexEntry]) -> None:
        """Clear and re-populate from a fresh entity list."""
        self.clear()
        self.populate(entities)

    def get_stats(self) -> dict:
        """Return index statistics."""
        return {
            "count": self._store.count(COLLECTION_ENTITY_INDEX),
            "last_refresh": self._last_refresh,
        }
