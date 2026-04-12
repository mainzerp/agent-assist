"""Pre-embedded entity index using ChromaDB."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.cache.vector_store import VectorStore, COLLECTION_ENTITY_INDEX
from app.models.entity_index import EntityIndexEntry

logger = logging.getLogger(__name__)

BATCH_SIZE = 500


class EntityIndex:
    """Pre-embedded entity index backed by ChromaDB."""

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store
        self._last_refresh: str | None = None
        self._status: dict = {
            "state": "ready",
            "progress": 0,
            "total": 0,
            "processed": 0,
            "error": None,
        }
        self._sync_stats: dict = {
            "added": 0,
            "updated": 0,
            "removed": 0,
            "unchanged": 0,
            "last_sync": None,
            "last_sync_duration_ms": 0,
        }

    @staticmethod
    def _build_metadata(entry: EntityIndexEntry) -> dict:
        """Build ChromaDB metadata dict from an EntityIndexEntry."""
        return {
            "friendly_name": entry.friendly_name,
            "domain": entry.domain,
            "area": entry.area or "",
            "device_class": entry.device_class or "",
            "aliases": ",".join(entry.aliases) if entry.aliases else "",
        }

    def populate(self, entities: list[EntityIndexEntry]) -> None:
        """Bulk upsert all HA entities into the entity_index collection.

        Called at startup after fetching GET /api/states.
        """
        if not entities:
            return
        total = len(entities)
        self._status = {
            "state": "building",
            "progress": 0,
            "total": total,
            "processed": 0,
            "error": None,
        }
        try:
            for start in range(0, total, BATCH_SIZE):
                batch = entities[start : start + BATCH_SIZE]
                ids = [e.entity_id for e in batch]
                documents = [e.embedding_text for e in batch]
                metadatas = [self._build_metadata(e) for e in batch]
                self._store.upsert(
                    COLLECTION_ENTITY_INDEX,
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
                self._status["processed"] = min(start + len(batch), total)
                self._status["progress"] = int(self._status["processed"] / total * 100)
            self._last_refresh = datetime.now(timezone.utc).isoformat()
            self._status["state"] = "ready"
            self._status["progress"] = 100
            logger.info("Entity index populated with %d entities", total)
        except Exception as exc:
            self._status["state"] = "error"
            self._status["error"] = str(exc)
            logger.error("Entity index populate failed: %s", exc)
            raise

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
            metadatas=[self._build_metadata(entry)],
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
        self._status["state"] = "building"
        self._status["progress"] = 0
        self.clear()
        self.populate(entities)

    def sync(self, entities: list[EntityIndexEntry]) -> dict:
        """Smart diff sync: upsert changed/new, remove deleted, skip unchanged.

        Returns dict with counts: added, updated, removed, unchanged.
        """
        import time as _time
        start = _time.monotonic()

        if not entities:
            return {"added": 0, "updated": 0, "removed": 0, "unchanged": 0}

        prev_state = self._status["state"]
        self._status["state"] = "syncing"

        try:
            # Build map of incoming entities
            ha_map: dict[str, EntityIndexEntry] = {e.entity_id: e for e in entities}

            # Fetch all current entries from ChromaDB
            current_data = self._store.get(
                COLLECTION_ENTITY_INDEX,
                include=["documents", "metadatas"],
            )
            current_ids = current_data.get("ids", [])
            current_docs = current_data.get("documents", [])
            current_metas = current_data.get("metadatas", [])

            # Build lookup: entity_id -> (document, metadata)
            chroma_map: dict[str, tuple[str, dict]] = {}
            for i, eid in enumerate(current_ids):
                chroma_map[eid] = (current_docs[i], current_metas[i])

            to_upsert: list[EntityIndexEntry] = []
            added = 0
            updated = 0
            unchanged = 0

            for entity_id, entry in ha_map.items():
                if entity_id in chroma_map:
                    old_doc, old_meta = chroma_map[entity_id]
                    new_doc = entry.embedding_text
                    new_meta = self._build_metadata(entry)
                    if new_doc != old_doc or new_meta != old_meta:
                        to_upsert.append(entry)
                        updated += 1
                    else:
                        unchanged += 1
                else:
                    to_upsert.append(entry)
                    added += 1

            # Find entities to remove (in ChromaDB but not in HA)
            to_remove = [eid for eid in current_ids if eid not in ha_map]
            removed = len(to_remove)

            # Batch upsert changed/new entities
            if to_upsert:
                for start_idx in range(0, len(to_upsert), BATCH_SIZE):
                    batch = to_upsert[start_idx : start_idx + BATCH_SIZE]
                    ids = [e.entity_id for e in batch]
                    documents = [e.embedding_text for e in batch]
                    metadatas = [self._build_metadata(e) for e in batch]
                    self._store.upsert(
                        COLLECTION_ENTITY_INDEX,
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                    )

            # Batch delete removed entities
            if to_remove:
                self._store.delete(COLLECTION_ENTITY_INDEX, ids=to_remove)

            elapsed_ms = int((_time.monotonic() - start) * 1000)

            self._last_refresh = datetime.now(timezone.utc).isoformat()
            self._status["state"] = "ready"

            self._sync_stats = {
                "added": added,
                "updated": updated,
                "removed": removed,
                "unchanged": unchanged,
                "last_sync": self._last_refresh,
                "last_sync_duration_ms": elapsed_ms,
            }

            logger.info(
                "Entity sync complete: +%d ~%d -%d =%d (%dms)",
                added, updated, removed, unchanged, elapsed_ms,
            )
            return {"added": added, "updated": updated, "removed": removed, "unchanged": unchanged}

        except Exception as exc:
            self._status["state"] = prev_state if prev_state != "syncing" else "ready"
            self._status["error"] = str(exc)
            logger.error("Entity sync failed: %s", exc)
            raise

    def get_embedding_status(self) -> dict:
        """Return current embedding status."""
        return dict(self._status)

    def get_stats(self) -> dict:
        """Return index statistics."""
        return {
            "count": self._store.count(COLLECTION_ENTITY_INDEX),
            "last_refresh": self._last_refresh,
            "embedding_status": dict(self._status),
            "sync": dict(self._sync_stats),
        }
