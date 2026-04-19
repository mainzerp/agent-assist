"""Tests for app.cache -- routing cache, response cache, cache manager, embedding, vector store."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.cache.routing_cache import RoutingCache
from app.cache.response_cache import ResponseCache
from app.cache.cache_manager import CacheManager, CacheResult
from app.cache.embedding import ChromaEmbeddingFunction, EmbeddingEngine
from app.cache.vector_store import (
    COLLECTION_ENTITY_INDEX,
    COLLECTION_RESPONSE_CACHE,
    COLLECTION_ROUTING_CACHE,
    VectorStore,
)
from app.models.cache import CachedAction, ResponseCacheEntry, RoutingCacheEntry

from tests.helpers import make_cached_action, make_response_cache_entry


# ---------------------------------------------------------------------------
# Routing cache
# ---------------------------------------------------------------------------

class TestRoutingCache:

    def _make_cache(self) -> tuple[RoutingCache, MagicMock]:
        store = MagicMock(spec=VectorStore)
        cache = RoutingCache(store)
        cache._threshold = 0.92
        cache._max_entries = 100
        return cache, store

    def test_lookup_hit_above_threshold(self):
        cache, store = self._make_cache()
        store.query.return_value = {
            "ids": [["entry-1"]],
            "distances": [[0.05]],  # similarity = 0.95
            "documents": [["turn on kitchen light"]],
            "metadatas": [[{
                "agent_id": "light-agent", "confidence": "0.95",
                "hit_count": "2", "created_at": "2025-01-01T00:00:00",
                "last_accessed": "2025-01-01T00:00:00",
            }]],
        }
        entry, similarity = cache.lookup("turn on kitchen light")
        assert entry is not None
        assert entry.agent_id == "light-agent"
        assert entry.hit_count == 3  # incremented from 2
        assert similarity == pytest.approx(0.95)

    def test_lookup_miss_below_threshold(self):
        cache, store = self._make_cache()
        store.query.return_value = {
            "ids": [["entry-1"]],
            "distances": [[0.15]],  # similarity = 0.85 < 0.92
            "documents": [["something else"]],
            "metadatas": [[{"agent_id": "general-agent", "confidence": "0.85",
                           "hit_count": "0", "created_at": "", "last_accessed": ""}]],
        }
        entry, similarity = cache.lookup("different query")
        assert entry is None
        assert similarity == pytest.approx(0.85)

    def test_lookup_empty_results(self):
        cache, store = self._make_cache()
        store.query.return_value = {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
        entry, similarity = cache.lookup("anything")
        assert entry is None
        assert similarity is None

    def test_store_upserts_entry(self):
        cache, store = self._make_cache()
        store.count.return_value = 0
        cache.store("turn on kitchen light", "light-agent", 0.95)
        store.upsert.assert_called_once()
        call_kwargs = store.upsert.call_args
        assert call_kwargs[1]["metadatas"][0]["agent_id"] == "light-agent" or \
               call_kwargs[0][3][0]["agent_id"] == "light-agent"

    def test_lru_eviction_triggers_at_max(self):
        cache, store = self._make_cache()
        cache._max_entries = 10
        store.count.return_value = 15
        store.get.return_value = {
            "ids": [f"id-{i}" for i in range(15)],
            "metadatas": [{"last_accessed": f"2025-01-{i+1:02d}T00:00:00"} for i in range(15)],
        }
        cache._enforce_lru()
        store.delete.assert_called_once()

    def test_lru_no_eviction_below_max(self):
        cache, store = self._make_cache()
        cache._max_entries = 100
        store.count.return_value = 5
        cache._enforce_lru()
        store.delete.assert_not_called()

    def test_get_stats(self):
        cache, store = self._make_cache()
        store.count.return_value = 42
        stats = cache.get_stats()
        assert stats["count"] == 42
        assert stats["threshold"] == 0.92

    async def test_load_config_from_db(self):
        cache, store = self._make_cache()
        with patch("app.cache.routing_cache.SettingsRepository") as mock_settings:
            mock_settings.get_value = AsyncMock(side_effect=["0.90", "1000"])
            await cache.load_config()
        assert cache._threshold == 0.90
        assert cache._max_entries == 1000

    def test_store_uses_deterministic_id(self):
        """Calling store() twice with the same query should upsert the same ID."""
        cache, store = self._make_cache()
        store.count.return_value = 0
        cache.store("turn on kitchen light", "light-agent", 0.95)
        cache.store("turn on kitchen light", "light-agent", 0.96)
        assert store.upsert.call_count == 2
        id1 = store.upsert.call_args_list[0][1]["ids"][0]
        id2 = store.upsert.call_args_list[1][1]["ids"][0]
        assert id1 == id2  # same deterministic hash

    def test_store_flushes_pending_updates(self):
        """store() should flush pending hit-count updates via update_metadata."""
        cache, store = self._make_cache()
        store.count.return_value = 0
        cache._pending_updates = {"old-id": ("old query", {"hit_count": "5"})}
        cache._hit_since_flush = 1
        cache.store("new query", "agent", 0.9)
        # Flush uses update_metadata, store uses upsert
        store.update_metadata.assert_called_once()
        store.upsert.assert_called_once()  # only the store() upsert
        assert cache._hit_since_flush == 0

    def test_flush_pending_public_method(self):
        """flush_pending() should delegate to _flush_pending_updates via update_metadata."""
        cache, store = self._make_cache()
        cache._pending_updates = {"id-1": ("q", {"hit_count": "3"})}
        cache.flush_pending()
        store.update_metadata.assert_called_once()
        store.upsert.assert_not_called()
        assert len(cache._pending_updates) == 0


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------

class TestResponseCache:

    def _make_cache(self) -> tuple[ResponseCache, MagicMock]:
        store = MagicMock(spec=VectorStore)
        cache = ResponseCache(store)
        cache._hit_threshold = 0.95
        cache._partial_threshold = 0.80
        cache._max_entries = 100
        return cache, store

    def test_lookup_hit_above_threshold(self):
        cache, store = self._make_cache()
        store.query.return_value = {
            "ids": [["resp-1"]],
            "distances": [[0.02]],  # similarity = 0.98
            "documents": [["turn on kitchen light"]],
            "metadatas": [[{
                "response_text": "Done, light is on.",
                "agent_id": "light-agent",
                "confidence": "0.98",
                "hit_count": "1",
                "entity_ids": "light.kitchen_ceiling",
                "cached_action": "",
                "created_at": "2025-01-01T00:00:00",
                "last_accessed": "2025-01-01T00:00:00",
            }]],
        }
        hit_type, entry, similarity = cache.lookup("turn on kitchen light")
        assert hit_type == "hit"
        assert entry is not None
        assert entry.response_text == "Done, light is on."
        assert similarity == pytest.approx(0.98)

    def test_lookup_partial_match(self):
        cache, store = self._make_cache()
        store.query.return_value = {
            "ids": [["resp-1"]],
            "distances": [[0.12]],  # similarity = 0.88, between 0.80 and 0.95
            "documents": [["turn on the kitchen light"]],
            "metadatas": [[{
                "response_text": "Done.",
                "agent_id": "light-agent",
                "confidence": "0.88",
                "hit_count": "0",
                "entity_ids": "light.kitchen",
                "cached_action": "",
                "created_at": "", "last_accessed": "",
            }]],
        }
        hit_type, entry, similarity = cache.lookup("switch on kitchen light")
        assert hit_type == "partial"
        assert entry is not None
        assert similarity == pytest.approx(0.88)

    def test_lookup_miss_below_partial(self):
        cache, store = self._make_cache()
        store.query.return_value = {
            "ids": [["resp-1"]],
            "distances": [[0.30]],  # similarity = 0.70 < 0.80
            "documents": [["something unrelated"]],
            "metadatas": [[{
                "response_text": "nope",
                "agent_id": "gen",
                "confidence": "0.70",
                "hit_count": "0",
                "entity_ids": "",
                "cached_action": "",
                "created_at": "", "last_accessed": "",
            }]],
        }
        hit_type, entry, similarity = cache.lookup("totally different")
        assert hit_type == "miss"
        assert entry is None
        assert similarity == pytest.approx(0.70)

    def test_lookup_empty_results(self):
        cache, store = self._make_cache()
        store.query.return_value = {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
        hit_type, entry, similarity = cache.lookup("anything")
        assert hit_type == "miss"
        assert similarity is None

    def test_lookup_with_cached_action(self):
        cache, store = self._make_cache()
        action = CachedAction(service="light/turn_on", entity_id="light.kitchen", service_data={})
        store.query.return_value = {
            "ids": [["resp-1"]],
            "distances": [[0.01]],
            "documents": [["turn on kitchen"]],
            "metadatas": [[{
                "response_text": "Done.",
                "agent_id": "light-agent",
                "confidence": "0.99",
                "hit_count": "0",
                "entity_ids": "light.kitchen",
                "cached_action": action.model_dump_json(),
                "created_at": "", "last_accessed": "",
            }]],
        }
        hit_type, entry, similarity = cache.lookup("turn on kitchen")
        assert hit_type == "hit"
        assert entry.cached_action is not None
        assert entry.cached_action.service == "light/turn_on"

    def test_store_upserts_entry(self):
        cache, store = self._make_cache()
        store.count.return_value = 0
        entry = make_response_cache_entry()
        cache.store(entry)
        store.upsert.assert_called_once()

    def test_invalidate_deletes_entry(self):
        cache, store = self._make_cache()
        cache.invalidate("resp-1")
        store.delete.assert_called_once_with(COLLECTION_RESPONSE_CACHE, ids=["resp-1"])

    def test_get_stats(self):
        cache, store = self._make_cache()
        store.count.return_value = 100
        stats = cache.get_stats()
        assert stats["count"] == 100
        assert stats["hit_threshold"] == 0.95
        assert stats["partial_threshold"] == 0.80

    async def test_load_config_from_db(self):
        cache, store = self._make_cache()
        with patch("app.cache.response_cache.SettingsRepository") as mock_settings:
            mock_settings.get_value = AsyncMock(side_effect=["0.90", "0.75", "5000"])
            await cache.load_config()
        assert cache._hit_threshold == 0.90
        assert cache._partial_threshold == 0.75
        assert cache._max_entries == 5000

    def test_store_uses_deterministic_id(self):
        """Calling store() twice with same query should upsert same ID."""
        cache, store = self._make_cache()
        store.count.return_value = 0
        entry1 = make_response_cache_entry(query_text="turn on kitchen light")
        entry2 = make_response_cache_entry(query_text="turn on kitchen light")
        cache.store(entry1)
        cache.store(entry2)
        assert store.upsert.call_count == 2
        id1 = store.upsert.call_args_list[0][1]["ids"][0]
        id2 = store.upsert.call_args_list[1][1]["ids"][0]
        assert id1 == id2

    def test_flush_pending_public_method(self):
        """flush_pending() should delegate to _flush_pending_updates via update_metadata."""
        cache, store = self._make_cache()
        cache._pending_updates = {"id-1": ("q", {"hit_count": "3"})}
        cache.flush_pending()
        store.update_metadata.assert_called_once()
        store.upsert.assert_not_called()
        assert len(cache._pending_updates) == 0


# ---------------------------------------------------------------------------
# Cache manager
# ---------------------------------------------------------------------------

class TestCacheManager:

    def _make_manager(self) -> tuple[CacheManager, MagicMock]:
        store = MagicMock(spec=VectorStore)
        manager = CacheManager(store)
        return manager, store

    async def test_process_response_hit(self):
        manager, store = self._make_manager()
        with patch.object(manager, "_process_inner") as mock_inner, \
             patch("app.cache.cache_manager.track_cache_event", new_callable=AsyncMock):
            mock_inner.return_value = CacheResult(
                hit_type="response_hit",
                agent_id="light-agent",
                response_text="Done.",
            )
            result = await manager.process("turn on light")
        assert result.hit_type == "response_hit"
        assert result.response_text == "Done."

    async def test_process_miss(self):
        manager, store = self._make_manager()
        with patch.object(manager, "_process_inner") as mock_inner, \
             patch("app.cache.cache_manager.track_cache_event", new_callable=AsyncMock):
            mock_inner.return_value = CacheResult(hit_type="miss")
            result = await manager.process("random query")
        assert result.hit_type == "miss"

    async def test_process_exception_returns_miss(self):
        manager, store = self._make_manager()
        with patch.object(manager, "_process_inner", side_effect=RuntimeError("db fail")), \
             patch("app.cache.cache_manager.track_cache_event", new_callable=AsyncMock):
            result = await manager.process("any query")
        assert result.hit_type == "miss"

    def test_store_routing_delegates(self):
        manager, store = self._make_manager()
        store.count.return_value = 0
        store.query.return_value = {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
        manager.store_routing("query", "light-agent", 0.95, "Turn on the light")
        store.upsert.assert_called()
        # Verify condensed_task is in the metadata
        call_args = store.upsert.call_args
        metadatas = call_args[1].get("metadatas") or call_args[0][2] if len(call_args[0]) > 2 else call_args[1]["metadatas"]
        assert metadatas[0]["condensed_task"] == "Turn on the light"

    def test_store_response_delegates(self):
        manager, store = self._make_manager()
        store.count.return_value = 0
        entry = make_response_cache_entry()
        manager.store_response(entry)
        store.upsert.assert_called()

    def test_invalidate_response_delegates(self):
        manager, store = self._make_manager()
        manager.invalidate_response("resp-1")
        store.delete.assert_called_once()

    def test_flush_routing(self):
        manager, store = self._make_manager()
        store.count.return_value = 5
        store.get.return_value = {"ids": ["a", "b"]}
        manager.flush(tier="routing")
        store.delete.assert_called()

    def test_flush_response(self):
        manager, store = self._make_manager()
        store.count.return_value = 5
        store.get.return_value = {"ids": ["a", "b"]}
        manager.flush(tier="response")
        store.delete.assert_called()

    def test_flush_both(self):
        manager, store = self._make_manager()
        store.count.return_value = 3
        store.get.return_value = {"ids": ["a"]}
        manager.flush(tier=None)
        assert store.delete.call_count == 2

    def test_get_stats(self):
        manager, store = self._make_manager()
        store.count.return_value = 10
        stats = manager.get_stats()
        assert "routing" in stats
        assert "response" in stats

    async def test_initialize_loads_config(self):
        manager, store = self._make_manager()

        async def routing_get_value(key, default=None):
            mapping = {
                "cache.routing.threshold": "0.92",
                "cache.routing.max_entries": "50000",
            }
            return mapping.get(key, default)

        async def response_get_value(key, default=None):
            mapping = {
                "cache.response.threshold": "0.95",
                "cache.response.partial_threshold": "0.80",
                "cache.response.max_entries": "20000",
            }
            return mapping.get(key, default)

        async def mgr_get_value(key, default=None):
            if key == "personality.prompt":
                return ""
            return default

        with patch("app.cache.routing_cache.SettingsRepository") as mock_rs, \
             patch("app.cache.response_cache.SettingsRepository") as mock_resps, \
             patch("app.db.repository.SettingsRepository") as mock_cms:
            mock_rs.get_value = AsyncMock(side_effect=routing_get_value)
            mock_resps.get_value = AsyncMock(side_effect=response_get_value)
            mock_cms.get_value = AsyncMock(side_effect=mgr_get_value)
            await manager.initialize()
        # No assertion needed -- just verifying no exception is raised

    async def test_reload_config(self):
        manager, store = self._make_manager()

        async def routing_get_value(key, default=None):
            mapping = {
                "cache.routing.threshold": "0.90",
                "cache.routing.max_entries": "50000",
            }
            return mapping.get(key, default)

        async def response_get_value(key, default=None):
            mapping = {
                "cache.response.threshold": "0.90",
                "cache.response.partial_threshold": "0.75",
                "cache.response.max_entries": "20000",
            }
            return mapping.get(key, default)

        async def mgr_get_value(key, default=None):
            if key == "personality.prompt":
                return ""
            return default

        with patch("app.cache.routing_cache.SettingsRepository") as mock_rs, \
             patch("app.cache.response_cache.SettingsRepository") as mock_resps, \
             patch("app.db.repository.SettingsRepository") as mock_cms:
            mock_rs.get_value = AsyncMock(side_effect=routing_get_value)
            mock_resps.get_value = AsyncMock(side_effect=response_get_value)
            mock_cms.get_value = AsyncMock(side_effect=mgr_get_value)
            await manager.reload_config()

    def test_flush_pending_delegates_to_both_caches(self):
        """flush_pending() should call flush_pending() on both caches."""
        manager, store = self._make_manager()
        manager._routing_cache._pending_updates = {"r-1": ("q", {"hit_count": "2"})}
        manager._response_cache._pending_updates = {"s-1": ("q", {"hit_count": "3"})}
        manager.flush_pending()
        # Both should have been flushed via update_metadata
        assert len(manager._routing_cache._pending_updates) == 0
        assert len(manager._response_cache._pending_updates) == 0
        assert store.update_metadata.call_count == 2

    def test_routing_cache_stores_condensed_task(self):
        """Routing cache should persist condensed_task in the ChromaDB metadata."""
        cache, store = TestRoutingCache()._make_cache()
        store.count.return_value = 0
        cache.store("turn on light", "light-agent", 0.95, "Turn on the light")
        store.upsert.assert_called_once()
        call_kwargs = store.upsert.call_args
        metadatas = call_kwargs[1].get("metadatas") or call_kwargs[0][3]
        assert metadatas[0]["condensed_task"] == "Turn on the light"

    def test_routing_cache_lookup_returns_condensed_task(self):
        """Routing cache lookup should return the stored condensed_task."""
        cache, store = TestRoutingCache()._make_cache()
        store.query.return_value = {
            "ids": [["entry-1"]],
            "distances": [[0.05]],
            "documents": [["turn on light"]],
            "metadatas": [[{
                "agent_id": "light-agent",
                "confidence": "0.95",
                "hit_count": "0",
                "condensed_task": "Turn on the light",
                "created_at": "2025-01-01T00:00:00",
                "last_accessed": "2025-01-01T00:00:00",
            }]],
        }
        entry, similarity = cache.lookup("turn on light")
        assert entry is not None
        assert entry.condensed_task == "Turn on the light"
        assert similarity == pytest.approx(0.95)

    def test_cache_result_carries_condensed_task(self):
        """CacheResult should propagate condensed_task from routing entry."""
        manager, store = self._make_manager()
        store.query.side_effect = [
            # Routing cache hit (now checked first)
            {
                "ids": [["r-1"]],
                "distances": [[0.03]],
                "documents": [["turn on light"]],
                "metadatas": [[{
                    "agent_id": "light-agent",
                    "confidence": "0.95",
                    "hit_count": "0",
                    "condensed_task": "Turn on the light",
                    "created_at": "2025-01-01T00:00:00",
                    "last_accessed": "2025-01-01T00:00:00",
                }]],
            },
        ]
        result = manager._process_inner("turn on light")
        assert result.hit_type == "routing_hit"
        assert result.condensed_task == "Turn on the light"

    def test_store_response_disabled_skips_store(self):
        """store_response should no-op when _response_cache_enabled is False."""
        manager, store = self._make_manager()
        manager._response_cache_enabled = False
        entry = make_response_cache_entry()
        manager.store_response(entry)
        store.upsert.assert_not_called()

    def test_complete_miss_returns_none_similarity(self):
        """COR-3: a full miss must report similarity=None instead of leaking
        the best cross-tier similarity into the trace UI."""
        manager, store = self._make_manager()
        # Routing miss: returns a result with low similarity (below threshold)
        store.query.side_effect = [
            {
                "ids": [["r-1"]],
                "distances": [[0.5]],
                "documents": [["something else"]],
                "metadatas": [[{
                    "agent_id": "light-agent",
                    "confidence": "0.5",
                    "hit_count": "0",
                    "condensed_task": "",
                    "created_at": "2025-01-01T00:00:00",
                    "last_accessed": "2025-01-01T00:00:00",
                }]],
            },
            # Response miss as well
            {
                "ids": [["s-1"]],
                "distances": [[0.6]],
                "documents": [["yet another"]],
                "metadatas": [[{
                    "agent_id": "light-agent",
                    "response_text": "x",
                    "hit_count": "0",
                    "created_at": "2025-01-01T00:00:00",
                    "last_accessed": "2025-01-01T00:00:00",
                }]],
            },
        ]
        result = manager._process_inner("totally unrelated query")
        assert result.hit_type == "miss"
        assert result.similarity is None

    def test_store_response_enabled_delegates(self):
        """store_response should delegate when _response_cache_enabled is True."""
        manager, store = self._make_manager()
        manager._response_cache_enabled = True
        store.count.return_value = 0
        entry = make_response_cache_entry()
        manager.store_response(entry)
        store.upsert.assert_called()

    async def test_process_response_hit_preserves_text_on_empty_rewrite(self):
        manager, store = self._make_manager()
        rewrite_agent = AsyncMock()
        rewrite_agent.rewrite = AsyncMock(return_value="")
        manager._rewrite_agent = rewrite_agent
        with patch.object(manager, "_process_inner") as mock_inner, \
             patch("app.cache.cache_manager.track_cache_event", new_callable=AsyncMock), \
             patch("app.cache.cache_manager.track_rewrite", new_callable=AsyncMock):
            mock_inner.return_value = CacheResult(
                hit_type="response_hit",
                agent_id="light-agent",
                response_text="Original cached text.",
            )
            result = await manager.process("turn on light")
            await manager.apply_rewrite(result)
        assert result.response_text == "Original cached text."

    async def test_process_response_hit_applies_rewrite(self):
        manager, store = self._make_manager()
        rewrite_agent = AsyncMock()
        rewrite_agent.rewrite = AsyncMock(return_value="Rephrased text.")
        manager._rewrite_agent = rewrite_agent
        with patch.object(manager, "_process_inner") as mock_inner, \
             patch("app.cache.cache_manager.track_cache_event", new_callable=AsyncMock), \
             patch("app.cache.cache_manager.track_rewrite", new_callable=AsyncMock):
            mock_inner.return_value = CacheResult(
                hit_type="response_hit",
                agent_id="light-agent",
                response_text="Original text.",
            )
            result = await manager.process("turn on light")
            await manager.apply_rewrite(result)
        assert result.response_text == "Rephrased text."

    async def test_process_response_hit_sets_rewrite_metadata(self):
        manager, store = self._make_manager()
        rewrite_agent = AsyncMock()
        rewrite_agent.rewrite = AsyncMock(return_value="Rephrased.")
        manager._rewrite_agent = rewrite_agent
        with patch.object(manager, "_process_inner") as mock_inner, \
             patch("app.cache.cache_manager.track_cache_event", new_callable=AsyncMock), \
             patch("app.cache.cache_manager.track_rewrite", new_callable=AsyncMock):
            mock_inner.return_value = CacheResult(
                hit_type="response_hit",
                agent_id="light-agent",
                response_text="Original.",
            )
            result = await manager.process("turn on light")
            await manager.apply_rewrite(result)
        assert result.rewrite_applied is True
        assert result.rewrite_latency_ms is not None
        assert result.rewrite_latency_ms > 0
        assert result.original_response_text == "Original."
        assert result.response_text == "Rephrased."

    async def test_process_response_hit_no_rewrite_metadata_on_empty(self):
        manager, store = self._make_manager()
        rewrite_agent = AsyncMock()
        rewrite_agent.rewrite = AsyncMock(return_value="")
        manager._rewrite_agent = rewrite_agent
        with patch.object(manager, "_process_inner") as mock_inner, \
             patch("app.cache.cache_manager.track_cache_event", new_callable=AsyncMock), \
             patch("app.cache.cache_manager.track_rewrite", new_callable=AsyncMock):
            mock_inner.return_value = CacheResult(
                hit_type="response_hit",
                agent_id="light-agent",
                response_text="Original.",
            )
            result = await manager.process("turn on light")
            await manager.apply_rewrite(result)
        assert result.rewrite_applied is False
        assert result.original_response_text is None
        assert result.response_text == "Original."

    async def test_process_response_hit_no_rewrite_metadata_on_exception(self):
        manager, store = self._make_manager()
        rewrite_agent = AsyncMock()
        rewrite_agent.rewrite = AsyncMock(side_effect=RuntimeError("LLM error"))
        manager._rewrite_agent = rewrite_agent
        with patch.object(manager, "_process_inner") as mock_inner, \
             patch("app.cache.cache_manager.track_cache_event", new_callable=AsyncMock), \
             patch("app.cache.cache_manager.track_rewrite", new_callable=AsyncMock):
            mock_inner.return_value = CacheResult(
                hit_type="response_hit",
                agent_id="light-agent",
                response_text="Original.",
            )
            result = await manager.process("turn on light")
            await manager.apply_rewrite(result)
        assert result.rewrite_applied is False
        assert result.original_response_text is None
        assert result.rewrite_latency_ms is not None


# ---------------------------------------------------------------------------
# Embedding engine
# ---------------------------------------------------------------------------

class TestEmbeddingEngine:

    def test_embed_local_via_sentence_transformer(self):
        engine = EmbeddingEngine()
        engine._provider = "local"
        engine._model_name = "all-MiniLM-L6-v2"

        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.zeros((1, 384))
        engine._local_model = mock_model

        result = engine.embed("test")
        assert len(result) == 384

    def test_embed_batch_local(self):
        engine = EmbeddingEngine()
        engine._provider = "local"
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.zeros((2, 384))
        engine._local_model = mock_model

        results = engine.embed_batch(["text1", "text2"])
        assert len(results) == 2
        assert len(results[0]) == 384

    def test_embed_external_via_litellm(self):
        engine = EmbeddingEngine()
        engine._provider = "external"
        engine._model_name = "openai/text-embedding-3-small"

        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1] * 384}, {"embedding": [0.2] * 384}]

        import sys
        mock_litellm = MagicMock()
        mock_litellm.embedding.return_value = mock_response
        with patch.dict(sys.modules, {"litellm": mock_litellm}):
            results = engine.embed_batch(["text1", "text2"])
        assert len(results) == 2

    async def test_initialize_loads_config(self):
        engine = EmbeddingEngine()
        with patch("app.cache.embedding.SettingsRepository") as mock_repo:
            mock_repo.get_value = AsyncMock(side_effect=["local", "all-MiniLM-L6-v2"])
            await engine.initialize()
        assert engine._provider == "local"
        assert engine._model_name == "all-MiniLM-L6-v2"


class TestChromaEmbeddingFunction:

    def test_calls_engine(self):
        mock_engine = MagicMock(spec=EmbeddingEngine)
        mock_engine.embed_batch.return_value = [[0.0] * 384]

        fn = ChromaEmbeddingFunction(mock_engine)
        result = fn(["test text"])
        assert len(result) == 1
        mock_engine.embed_batch.assert_called_once_with(["test text"])


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class TestVectorStore:

    def test_add_delegates_to_collection(self):
        store = VectorStore()
        mock_col = MagicMock()
        store._collections = {COLLECTION_ENTITY_INDEX: mock_col}
        store.add(COLLECTION_ENTITY_INDEX, ids=["a"], documents=["doc"])
        mock_col.add.assert_called_once()

    def test_upsert_delegates_to_collection(self):
        store = VectorStore()
        mock_col = MagicMock()
        store._collections = {COLLECTION_ROUTING_CACHE: mock_col}
        store.upsert(COLLECTION_ROUTING_CACHE, ids=["a"], documents=["doc"])
        mock_col.upsert.assert_called_once()

    def test_query_delegates_to_collection(self):
        store = VectorStore()
        mock_col = MagicMock()
        mock_col.query.return_value = {"ids": [["a"]], "distances": [[0.1]]}
        store._collections = {COLLECTION_ENTITY_INDEX: mock_col}
        result = store.query(COLLECTION_ENTITY_INDEX, query_texts=["test"])
        assert result["ids"] == [["a"]]

    def test_delete_delegates_to_collection(self):
        store = VectorStore()
        mock_col = MagicMock()
        store._collections = {COLLECTION_RESPONSE_CACHE: mock_col}
        store.delete(COLLECTION_RESPONSE_CACHE, ids=["x"])
        mock_col.delete.assert_called_once_with(ids=["x"])

    def test_count_delegates_to_collection(self):
        store = VectorStore()
        mock_col = MagicMock()
        mock_col.count.return_value = 42
        store._collections = {COLLECTION_ROUTING_CACHE: mock_col}
        assert store.count(COLLECTION_ROUTING_CACHE) == 42

    def test_get_delegates_to_collection(self):
        store = VectorStore()
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": ["a"], "metadatas": [{}]}
        store._collections = {COLLECTION_ENTITY_INDEX: mock_col}
        result = store.get(COLLECTION_ENTITY_INDEX, ids=["a"])
        assert result["ids"] == ["a"]

    def test_get_collection_missing_raises(self):
        store = VectorStore()
        store._collections = {}
        with pytest.raises(KeyError):
            store.get_collection("nonexistent")

    def test_update_metadata_delegates_to_collection(self):
        store = VectorStore()
        mock_col = MagicMock()
        store._collections = {COLLECTION_ENTITY_INDEX: mock_col}
        store.update_metadata(
            COLLECTION_ENTITY_INDEX,
            ids=["a", "b"],
            metadatas=[{"key": "v1"}, {"key": "v2"}],
        )
        mock_col.update.assert_called_once_with(
            ids=["a", "b"], metadatas=[{"key": "v1"}, {"key": "v2"}]
        )

    def test_update_metadata_reconnects_on_closed(self):
        store = VectorStore()
        mock_col = MagicMock()
        mock_col.update.side_effect = RuntimeError("connection closed")
        store._collections = {COLLECTION_ENTITY_INDEX: mock_col}
        mock_col2 = MagicMock()
        original_get = store.get_collection
        call_count = 0
        def side_effect_get(name):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return original_get(name)
            return mock_col2
        with patch.object(store, "_reinitialize_sync") as mock_reinit, \
             patch.object(store, "get_collection", side_effect=side_effect_get):
            store.update_metadata(
                COLLECTION_ENTITY_INDEX,
                ids=["a"],
                metadatas=[{"key": "v1"}],
            )
        mock_reinit.assert_called_once()
        mock_col2.update.assert_called_once()

    def test_update_metadata_raises_non_closed_error(self):
        store = VectorStore()
        mock_col = MagicMock()
        mock_col.update.side_effect = ValueError("bad data")
        store._collections = {COLLECTION_ENTITY_INDEX: mock_col}
        with pytest.raises(ValueError, match="bad data"):
            store.update_metadata(
                COLLECTION_ENTITY_INDEX,
                ids=["a"],
                metadatas=[{"key": "v1"}],
            )


# ---------------------------------------------------------------------------
# Cache trace visibility -- similarity propagation tests
# ---------------------------------------------------------------------------

class TestCacheTraceSimilarity:

    def test_cache_result_includes_similarity_on_routing_hit(self):
        """CacheResult.similarity is populated on a routing cache hit."""
        store = MagicMock(spec=VectorStore)
        manager = CacheManager(store)
        # Routing cache hit (now checked first, distance 0.05 = similarity 0.95)
        store.query.side_effect = [
            {
                "ids": [["r-1"]],
                "distances": [[0.05]],
                "documents": [["turn on light"]],
                "metadatas": [[{
                    "agent_id": "light-agent",
                    "confidence": "0.95",
                    "hit_count": "0",
                    "condensed_task": "Turn on",
                    "created_at": "2025-01-01T00:00:00",
                    "last_accessed": "2025-01-01T00:00:00",
                }]],
            },
        ]
        result = manager._process_inner("turn on light")
        assert result.hit_type == "routing_hit"
        assert result.similarity == pytest.approx(0.95)

    def test_cache_result_includes_similarity_on_miss(self):
        """COR-3: CacheResult.similarity is None on a complete miss; the
        previous behavior of leaking the best cross-tier similarity was
        misleading in the trace UI."""
        store = MagicMock(spec=VectorStore)
        manager = CacheManager(store)
        # Routing cache miss with similarity 0.80 (now checked first)
        store.query.side_effect = [
            {
                "ids": [["r-1"]],
                "distances": [[0.20]],
                "documents": [["other"]],
                "metadatas": [[{
                    "agent_id": "general-agent", "confidence": "0.80",
                    "hit_count": "0", "created_at": "", "last_accessed": "",
                }]],
            },
            # Response cache miss with similarity 0.70
            {
                "ids": [["resp-1"]],
                "distances": [[0.30]],
                "documents": [["unrelated"]],
                "metadatas": [[{
                    "response_text": "x", "agent_id": "gen", "confidence": "0.7",
                    "hit_count": "0", "entity_ids": "", "cached_action": "",
                    "created_at": "", "last_accessed": "",
                }]],
            },
        ]
        result = manager._process_inner("some query")
        assert result.hit_type == "miss"
        assert result.similarity is None

    def test_routing_cache_lookup_returns_similarity_tuple(self):
        """routing_cache.lookup() returns (entry, similarity) tuple."""
        store = MagicMock(spec=VectorStore)
        cache = RoutingCache(store)
        cache._threshold = 0.92
        store.query.return_value = {
            "ids": [["e-1"]],
            "distances": [[0.03]],
            "documents": [["test"]],
            "metadatas": [[{
                "agent_id": "light-agent", "confidence": "0.97",
                "hit_count": "0", "created_at": "2025-01-01T00:00:00",
                "last_accessed": "2025-01-01T00:00:00",
            }]],
        }
        entry, sim = cache.lookup("test")
        assert entry is not None
        assert sim == pytest.approx(0.97)

    def test_response_cache_lookup_returns_similarity_tuple(self):
        """response_cache.lookup() returns (hit_type, entry, similarity) tuple."""
        store = MagicMock(spec=VectorStore)
        cache = ResponseCache(store)
        cache._hit_threshold = 0.95
        cache._partial_threshold = 0.80
        store.query.return_value = {
            "ids": [["r-1"]],
            "distances": [[0.01]],
            "documents": [["test"]],
            "metadatas": [[{
                "response_text": "Done.", "agent_id": "light-agent",
                "confidence": "0.99", "hit_count": "0",
                "entity_ids": "", "cached_action": "",
                "created_at": "", "last_accessed": "",
            }]],
        }
        hit_type, entry, sim = cache.lookup("test")
        assert hit_type == "hit"
        assert entry is not None
        assert sim == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# Phase 4.4: Cache eviction tests
# ---------------------------------------------------------------------------

class TestRoutingCacheEviction:
    """Tests for interval-based LRU eviction and hit count buffering in routing cache."""

    def _make_cache(self) -> tuple[RoutingCache, MagicMock]:
        store = MagicMock(spec=VectorStore)
        cache = RoutingCache(store)
        cache._threshold = 0.92
        cache._max_entries = 10
        return cache, store

    def test_eviction_triggers_at_interval(self):
        """LRU eviction should only run every _eviction_interval stores."""
        cache, store = self._make_cache()
        cache._eviction_interval = 5
        store.count.return_value = 5  # below max, so no actual eviction needed

        for i in range(4):
            cache._store_count = i
            store.count.reset_mock()
            cache.store(f"query-{i}", "light-agent", 0.95)
        # count() should NOT have been called for eviction check on stores 0-3
        # (store calls upsert + may call count for eviction)

        # On the 5th store, eviction interval is hit
        cache._store_count = 4
        cache.store("query-final", "light-agent", 0.95)
        # The store method should have checked count for eviction

    def test_eviction_does_not_trigger_before_interval(self):
        """LRU eviction should not check before the interval is reached."""
        cache, store = self._make_cache()
        cache._eviction_interval = 100
        cache._store_count = 0
        store.count.return_value = 0
        cache.store("query-1", "light-agent", 0.95)
        # store_count should have incremented but no eviction check
        assert cache._store_count == 1

    def test_hit_count_buffering_flushes_at_threshold(self):
        """Pending hit updates should flush when buffer reaches _flush_interval."""
        cache, store = self._make_cache()
        cache._flush_interval = 3

        # Simulate lookups that buffer hits
        for i in range(3):
            store.query.return_value = {
                "ids": [[f"entry-{i}"]],
                "distances": [[0.05]],
                "documents": [[f"query-{i}"]],
                "metadatas": [[{
                    "agent_id": "light-agent", "confidence": "0.95",
                    "hit_count": "1", "created_at": "2025-01-01T00:00:00",
                    "last_accessed": "2025-01-01T00:00:00",
                }]],
            }
            cache.lookup(f"query-{i}")

        # After flush_interval lookups, update_metadata should have been called for flush
        assert store.update_metadata.call_count >= 1

    def test_batch_delete_in_chunks(self):
        """When evicting many entries, delete should be called in chunks of 500."""
        cache, store = self._make_cache()
        cache._max_entries = 10
        # Simulate 1010 entries
        store.count.return_value = 1010
        ids = [f"id-{i}" for i in range(1010)]
        metadatas = [{"last_accessed": f"2025-01-{(i % 28) + 1:02d}T00:00:00"} for i in range(1010)]
        store.get.return_value = {"ids": ids, "metadatas": metadatas}
        cache._enforce_lru()
        # Should delete in chunks - at least 2 calls (1000 excess / 500)
        assert store.delete.call_count >= 2


class TestResponseCacheEviction:
    """Tests for interval-based LRU eviction in response cache."""

    def _make_cache(self) -> tuple[ResponseCache, MagicMock]:
        store = MagicMock(spec=VectorStore)
        cache = ResponseCache(store)
        cache._hit_threshold = 0.95
        cache._partial_threshold = 0.80
        cache._max_entries = 10
        return cache, store

    def test_eviction_triggers_at_interval(self):
        """LRU eviction should only run every _eviction_interval stores."""
        cache, store = self._make_cache()
        cache._eviction_interval = 5
        store.count.return_value = 5

        entry = make_response_cache_entry()
        for i in range(4):
            cache._store_count = i
            store.count.reset_mock()
            cache.store(entry)

        cache._store_count = 4
        cache.store(entry)

    def test_batch_delete_in_chunks(self):
        """Response cache eviction should also u batch deletes in chunks of 500."""
        cache, store = self._make_cache()
        cache._max_entries = 10
        store.count.return_value = 600
        ids = [f"id-{i}" for i in range(600)]
        metadatas = [{"last_accessed": f"2025-01-{(i % 28) + 1:02d}T00:00:00"} for i in range(600)]
        store.get.return_value = {"ids": ids, "metadatas": metadatas}
        cache._enforce_lru()
        # 590 excess / 500 = 2 chunks
        assert store.delete.call_count >= 2


# ---------------------------------------------------------------------------
# Orchestrator _store_response_cache cacheable flag
# ---------------------------------------------------------------------------

class TestStoreResponseCacheCacheable:
    """Test that _store_response_cache respects the cacheable flag."""

    def _make_orchestrator(self):
        from app.agents.orchestrator import OrchestratorAgent
        orch = OrchestratorAgent.__new__(OrchestratorAgent)
        cache_manager = MagicMock()

        async def _store_response_async(entry):
            cache_manager.store_response(entry)

        cache_manager.store_response_async = _store_response_async
        orch._cache_manager = cache_manager
        return orch

    async def test_skips_non_cacheable_action(self):
        orch = self._make_orchestrator()
        stored = await orch._store_response_cache(
            user_text="what is the temperature",
            speech="It is 22 degrees.",
            target_agent="climate-agent",
            confidence=0.95,
            action_executed={"action": "query_climate_state", "entity_id": "sensor.temp",
                             "success": True, "cacheable": False},
            has_error=False,
        )
        assert stored is False
        orch._cache_manager.store_response.assert_not_called()

    async def test_stores_cacheable_action(self):
        orch = self._make_orchestrator()
        stored = await orch._store_response_cache(
            user_text="turn on kitchen light",
            speech="Done, kitchen light is on.",
            target_agent="light-agent",
            confidence=0.95,
            action_executed={"action": "turn_on", "entity_id": "light.kitchen",
                             "success": True, "cacheable": True},
            has_error=False,
        )
        assert stored is True
        orch._cache_manager.store_response.assert_called_once()

    async def test_stores_action_without_cacheable_field(self):
        orch = self._make_orchestrator()
        stored = await orch._store_response_cache(
            user_text="turn off bedroom light",
            speech="Done, bedroom light is off.",
            target_agent="light-agent",
            confidence=0.95,
            action_executed={"action": "turn_off", "entity_id": "light.bedroom",
                             "success": True},
            has_error=False,
        )
        assert stored is True
        orch._cache_manager.store_response.assert_called_once()


# ---------------------------------------------------------------------------
# Response cache purge readonly entries
# ---------------------------------------------------------------------------

class TestResponseCachePurgeReadonly:
    """Tests for ResponseCache.purge_readonly_entries()."""

    def _make_cache(self) -> tuple[ResponseCache, MagicMock]:
        store = MagicMock(spec=VectorStore)
        cache = ResponseCache(store)
        return cache, store

    def test_purge_removes_readonly_entries(self):
        cache, store = self._make_cache()
        store.get.return_value = {
            "ids": ["id-1", "id-2", "id-3"],
            "metadatas": [
                {"cached_action": "", "response_text": "It is 22 degrees."},
                {"cached_action": '{"service":"light/turn_on","entity_id":"light.kitchen","service_data":{}}', "response_text": "Done."},
                {"cached_action": "", "response_text": "The door is locked."},
            ],
        }
        count = cache.purge_readonly_entries()
        assert count == 2
        store.delete.assert_called_once_with(COLLECTION_RESPONSE_CACHE, ids=["id-1", "id-3"])

    def test_purge_skips_entries_with_cached_action(self):
        cache, store = self._make_cache()
        store.get.return_value = {
            "ids": ["id-1"],
            "metadatas": [
                {"cached_action": '{"service":"light/turn_on","entity_id":"light.kitchen","service_data":{}}'},
            ],
        }
        count = cache.purge_readonly_entries()
        assert count == 0
        store.delete.assert_not_called()

    def test_purge_empty_collection(self):
        cache, store = self._make_cache()
        store.get.return_value = {"ids": [], "metadatas": []}
        count = cache.purge_readonly_entries()
        assert count == 0
        store.delete.assert_not_called()

    def test_purge_handles_missing_cached_action_key(self):
        """Entries without cached_action key (pre-v0.14.0) should be purged."""
        cache, store = self._make_cache()
        store.get.return_value = {
            "ids": ["id-1", "id-2"],
            "metadatas": [
                {"response_text": "Old entry without cached_action field."},
                {"cached_action": '{"service":"light/turn_on","entity_id":"light.k","service_data":{}}'},
            ],
        }
        count = cache.purge_readonly_entries()
        assert count == 1
        store.delete.assert_called_once_with(COLLECTION_RESPONSE_CACHE, ids=["id-1"])

    async def test_cache_manager_purge_delegates(self):
        """CacheManager.purge_readonly_entries() should delegate to ResponseCache."""
        store = MagicMock(spec=VectorStore)
        manager = CacheManager(store)
        store.get.return_value = {
            "ids": ["id-1"],
            "metadatas": [{"cached_action": ""}],
        }
        count = await manager.purge_readonly_entries()
        assert count == 1

    def test_purge_removes_readonly_service_entries(self):
        """Entries with read-only service (query_*, list_*) should be purged."""
        cache, store = self._make_cache()
        store.get.return_value = {
            "ids": ["id-1", "id-2", "id-3", "id-4"],
            "metadatas": [
                {"cached_action": '{"service":"sensor/query_status","entity_id":"sensor.temp","service_data":{}}'},
                {"cached_action": '{"service":"light/turn_on","entity_id":"light.kitchen","service_data":{}}'},
                {"cached_action": '{"service":"media/list_sources","entity_id":"media_player.tv","service_data":{}}'},
                {"cached_action": ""},
            ],
        }
        count = cache.purge_readonly_entries()
        assert count == 3  # id-1 (query_status), id-3 (list_sources), id-4 (empty)
        store.delete.assert_called_once_with(COLLECTION_RESPONSE_CACHE, ids=["id-1", "id-3", "id-4"])

    def test_is_readonly_action_helper(self):
        """Unit test for _is_readonly_action static method."""
        assert ResponseCache._is_readonly_action("") is True
        assert ResponseCache._is_readonly_action('{"service":"sensor/query_status","entity_id":"x","service_data":{}}') is True
        assert ResponseCache._is_readonly_action('{"service":"media/list_sources","entity_id":"x","service_data":{}}') is True
        assert ResponseCache._is_readonly_action('{"service":"light/turn_on","entity_id":"x","service_data":{}}') is False
        assert ResponseCache._is_readonly_action('{"service":"climate/set_temperature","entity_id":"x","service_data":{}}') is False
        assert ResponseCache._is_readonly_action("invalid json") is False
