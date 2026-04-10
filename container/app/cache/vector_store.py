"""ChromaDB wrapper managing cache and entity index collections."""

from __future__ import annotations

import logging

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import settings
from app.cache.embedding import ChromaEmbeddingFunction, get_embedding_engine

logger = logging.getLogger(__name__)

COLLECTION_ENTITY_INDEX = "entity_index"
COLLECTION_ROUTING_CACHE = "routing_cache"
COLLECTION_RESPONSE_CACHE = "response_cache"


class VectorStore:
    """Manages ChromaDB PersistentClient and all three collections."""

    def __init__(self) -> None:
        self._client: chromadb.ClientAPI | None = None
        self._embedding_fn: ChromaEmbeddingFunction | None = None
        self._collections: dict[str, Collection] = {}

    async def initialize(self) -> None:
        """Create PersistentClient and get/create all collections."""
        engine = await get_embedding_engine()
        self._embedding_fn = ChromaEmbeddingFunction(engine)
        self._client = chromadb.PersistentClient(path=settings.chromadb_persist_dir)
        for name in (COLLECTION_ENTITY_INDEX, COLLECTION_ROUTING_CACHE, COLLECTION_RESPONSE_CACHE):
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        logger.info(
            "VectorStore initialized with %d collections at %s",
            len(self._collections),
            settings.chromadb_persist_dir,
        )

    def get_collection(self, name: str) -> Collection:
        """Return a named collection. Must call initialize() first."""
        return self._collections[name]

    def add(
        self,
        collection_name: str,
        ids: list[str],
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add entries to a collection."""
        col = self.get_collection(collection_name)
        kwargs: dict = {"ids": ids}
        if documents is not None:
            kwargs["documents"] = documents
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        if metadatas is not None:
            kwargs["metadatas"] = metadatas
        col.add(**kwargs)

    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        """Upsert entries into a collection."""
        col = self.get_collection(collection_name)
        kwargs: dict = {"ids": ids}
        if documents is not None:
            kwargs["documents"] = documents
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        if metadatas is not None:
            kwargs["metadatas"] = metadatas
        col.upsert(**kwargs)

    def query(
        self,
        collection_name: str,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 5,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict:
        """Query a collection by text or embedding. Returns ChromaDB result dict."""
        col = self.get_collection(collection_name)
        kwargs: dict = {"n_results": n_results}
        if query_texts is not None:
            kwargs["query_texts"] = query_texts
        if query_embeddings is not None:
            kwargs["query_embeddings"] = query_embeddings
        if where is not None:
            kwargs["where"] = where
        if include is not None:
            kwargs["include"] = include
        return col.query(**kwargs)

    def delete(self, collection_name: str, ids: list[str]) -> None:
        """Delete entries by ID from a collection."""
        self.get_collection(collection_name).delete(ids=ids)

    def count(self, collection_name: str) -> int:
        """Return the number of entries in a collection."""
        return self.get_collection(collection_name).count()

    def get(
        self,
        collection_name: str,
        ids: list[str] | None = None,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict:
        """Get entries by ID or filter from a collection."""
        col = self.get_collection(collection_name)
        kwargs: dict = {}
        if ids is not None:
            kwargs["ids"] = ids
        if where is not None:
            kwargs["where"] = where
        if include is not None:
            kwargs["include"] = include
        return col.get(**kwargs)


_store: VectorStore | None = None


async def get_vector_store() -> VectorStore:
    """Return the singleton VectorStore, initializing on first call."""
    global _store
    if _store is None:
        _store = VectorStore()
        await _store.initialize()
    return _store
