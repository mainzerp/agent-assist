"""Unified embedding engine for local and external providers."""

from __future__ import annotations

import logging

import chromadb

from app.db.repository import SettingsRepository

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Unified embedding engine supporting local and external providers."""

    def __init__(self) -> None:
        self._provider: str | None = None
        self._model_name: str | None = None
        self._local_model = None  # SentenceTransformer instance, lazy-loaded

    async def _load_config(self) -> None:
        """Read embedding.provider and embedding.*_model from settings table."""
        self._provider = await SettingsRepository.get_value("embedding.provider", "local")
        if self._provider == "local":
            self._model_name = await SettingsRepository.get_value("embedding.local_model", "all-MiniLM-L6-v2")
        else:
            self._model_name = await SettingsRepository.get_value("embedding.external_model", "")

    def _get_local_model(self):
        """Lazy-load sentence-transformers model on first use."""
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(self._model_name)
            logger.info("Loaded local embedding model: %s", self._model_name)
        return self._local_model

    async def initialize(self) -> None:
        """Load config from DB and pre-load the model. Must call before embed/embed_batch."""
        await self._load_config()
        if self._provider == "local":
            self._get_local_model()

    def get_info(self) -> dict:
        """Return embedding model configuration info."""
        dimensions = None
        if self._provider == "local" and self._local_model is not None:
            dimensions = self._local_model.get_sentence_embedding_dimension()
        elif self._provider == "local":
            defaults = {"all-MiniLM-L6-v2": 384, "all-mpnet-base-v2": 768}
            dimensions = defaults.get(self._model_name)
        return {
            "provider": self._provider or "unknown",
            "model": self._model_name or "unknown",
            "dimensions": dimensions,
        }

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns 384-dim float list for local model."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        if self._provider == "local":
            return self._embed_local(texts)
        return self._embed_external(texts)

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Use sentence-transformers for local embedding."""
        model = self._get_local_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    def _embed_external(self, texts: list[str]) -> list[list[float]]:
        """Use litellm for external provider embedding."""
        import litellm

        response = litellm.embedding(model=self._model_name, input=texts)
        return [item["embedding"] for item in response.data]


class ChromaEmbeddingFunction(chromadb.EmbeddingFunction[list[str]]):
    """Adapter wrapping EmbeddingEngine for ChromaDB's EmbeddingFunction interface."""

    def __init__(self, engine: EmbeddingEngine) -> None:
        self._engine = engine

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._engine.embed_batch(input)


_engine: EmbeddingEngine | None = None


async def get_embedding_engine() -> EmbeddingEngine:
    """Return the singleton EmbeddingEngine, initializing on first call."""
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
        await _engine.initialize()
    return _engine


async def get_embedding_info() -> dict:
    """Return embedding config info from the singleton engine."""
    engine = await get_embedding_engine()
    return engine.get_info()
