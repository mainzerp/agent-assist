"""Regression tests for P1-3 / P1-4 / P1-5 cache findings.

P1-3: thread-safety + LRU pagination + re-queue on flush failure.
P1-4: classify confidence gating + ``None`` for old-format lines.
P1-5: response-cache replays service_data parameters.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

from app.cache._state import _CacheState
from app.cache.response_cache import ResponseCache
from app.cache.routing_cache import RoutingCache
from app.cache.vector_store import VectorStore
from app.models.cache import CachedAction, ResponseCacheEntry

# ---------------------------------------------------------------------------
# P1-3: threadsafety
# ---------------------------------------------------------------------------


class TestCacheStateConcurrency:
    def test_record_pending_update_is_threadsafe(self):
        state = _CacheState()
        errors: list[BaseException] = []

        def worker(i: int) -> None:
            try:
                state.record_pending_update(f"id-{i}", f"q-{i}", {"hit_count": str(i)}, flush_interval=10_000)
            except BaseException as exc:  # pragma: no cover - defensive
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        pending = state.snapshot_pending()
        assert len(pending) == 100

    def test_swap_pending_returns_consistent_snapshot(self):
        state = _CacheState()
        for i in range(5):
            state.record_pending_update(f"id-{i}", "q", {}, flush_interval=10_000)

        snap = state.swap_pending()
        assert len(snap) == 5
        # After swap the internal buffer must be empty so the next
        # flush does not double-write the same rows.
        assert not state.has_pending()

    def test_requeue_failed_restores_pending(self):
        state = _CacheState()
        state.record_pending_update("id-1", "q", {"hit_count": "1"}, flush_interval=10_000)
        snap = state.swap_pending()
        assert not state.has_pending()
        state.requeue_failed(snap)
        assert state.has_pending()
        restored = state.snapshot_pending()
        assert restored["id-1"][1]["hit_count"] == "1"


class TestRoutingCacheStoreConcurrency:
    def _make_cache(self) -> tuple[RoutingCache, MagicMock]:
        store = MagicMock(spec=VectorStore)
        store.count.return_value = 0
        cache = RoutingCache(store)
        cache._threshold = 0.92
        cache._max_entries = 10_000
        cache._eviction_interval = 10_000
        return cache, store

    def test_parallel_stores_do_not_lose_updates(self):
        """50 concurrent store() calls -- no crash, upserts on every call."""
        cache, store = self._make_cache()

        async def spawn_all() -> None:
            await asyncio.gather(
                *(asyncio.to_thread(cache.store, f"query-{i}", "light-agent", 0.95) for i in range(50))
            )

        asyncio.run(spawn_all())
        # Each store() produces exactly one upsert regardless of
        # scheduling. If the shared counter or the pending map were
        # corrupted, the MagicMock would either raise or the test
        # would deadlock before reaching the assertion.
        assert store.upsert.call_count == 50


class TestRoutingCacheFlushRequeue:
    def test_flush_failure_requeues_pending(self):
        """P1-3: when update_metadata raises, pending rows stay pending."""
        store = MagicMock(spec=VectorStore)
        store.update_metadata.side_effect = RuntimeError("chroma down")
        cache = RoutingCache(store)
        cache._state.record_pending_update("id-1", "q", {"hit_count": "2"}, flush_interval=10_000)
        assert cache._state.has_pending()

        cache._flush_pending_updates()

        # Buffer must be re-populated so the next flush can retry. The
        # previous implementation dropped the pending rows on the
        # floor whenever Chroma raised, silently losing hit counts.
        assert cache._state.has_pending()
        restored = cache._state.snapshot_pending()
        assert "id-1" in restored


# ---------------------------------------------------------------------------
# P1-3: LRU pagination
# ---------------------------------------------------------------------------


class TestRoutingCacheLRUPagination:
    def test_enforce_lru_paginates_get_calls(self):
        """``_enforce_lru`` must issue paginated ``get`` calls, not one
        fat ``get`` that loads the whole collection into memory."""
        store = MagicMock(spec=VectorStore)
        cache = RoutingCache(store)
        cache._max_entries = 10
        store.count.return_value = 1100

        # Two pages: 1000 entries, then a 100-entry tail that also
        # signals the end of pagination (len < PAGE_SIZE).
        ids_page1 = [f"id-{i}" for i in range(1000)]
        metas_page1 = [{"last_accessed": f"2025-01-{(i % 28) + 1:02d}T00:00:00"} for i in range(1000)]
        ids_page2 = [f"id-{i}" for i in range(1000, 1100)]
        metas_page2 = [{"last_accessed": f"2025-02-{(i % 28) + 1:02d}T00:00:00"} for i in range(100)]
        store.get.side_effect = [
            {"ids": ids_page1, "metadatas": metas_page1},
            {"ids": ids_page2, "metadatas": metas_page2},
        ]

        cache._enforce_lru()

        # Assert that get() was called with a ``limit`` kwarg at least
        # twice -- this is the distinguishing signal between the old
        # "load everything" behaviour and the new paginated sweep.
        assert store.get.call_count == 2
        for call in store.get.call_args_list:
            assert "limit" in call.kwargs
            assert "offset" in call.kwargs
        # And deletions actually happened for the overage.
        assert store.delete.call_count >= 1


# ---------------------------------------------------------------------------
# P1-4: classify confidence gating
# ---------------------------------------------------------------------------


class TestClassifyNoConfidence:
    async def test_old_format_line_yields_none_confidence(self):
        from app.agents.orchestrator import OrchestratorAgent
        from app.models.agent import AgentCard

        orch = OrchestratorAgent(dispatcher=AsyncMock())
        orch._registry = AsyncMock()
        orch._registry.list_agents = AsyncMock(
            return_value=[
                AgentCard(agent_id="light-agent", name="", description="", skills=[]),
            ]
        )
        results = await orch._parse_classification("light-agent: turn on bedroom", "turn on bedroom")
        assert len(results) == 1
        assert results[0][0] == "light-agent"
        assert results[0][2] is None


# ---------------------------------------------------------------------------
# P1-5: response cache replay with parameters
# ---------------------------------------------------------------------------


def _make_response_cache() -> tuple[ResponseCache, MagicMock]:
    store = MagicMock(spec=VectorStore)
    cache = ResponseCache(store)
    cache._hit_threshold = 0.95
    cache._partial_threshold = 0.80
    cache._max_entries = 1000
    cache._eviction_interval = 1000
    return cache, store


class TestResponseCacheReplayServiceData:
    def test_store_and_lookup_round_trips_service_data(self):
        cache, store = _make_response_cache()
        store.count.return_value = 0

        cached_action = CachedAction(
            service="light/turn_on",
            entity_id="light.bedroom",
            service_data={"brightness_pct": 30, "transition": 2},
        )
        entry = ResponseCacheEntry(
            query_text="dim the bedroom lights",
            response_text="Done.",
            agent_id="light-agent",
            confidence=0.97,
            cached_action=cached_action,
            entity_ids=["light.bedroom"],
        )
        cache.store(entry)

        # Inspect what was persisted -- this is the contract the cache
        # manager relies on for replay.
        store.upsert.assert_called_once()
        kwargs = store.upsert.call_args.kwargs
        meta = kwargs["metadatas"][0]
        assert meta["schema_version"] == "2"
        restored = CachedAction.model_validate_json(meta["cached_action"])
        assert restored.service_data == {"brightness_pct": 30, "transition": 2}

        # Now simulate a lookup hit on the stored entry.
        store.query.return_value = {
            "ids": [[kwargs["ids"][0]]],
            "distances": [[0.02]],
            "documents": [[entry.query_text]],
            "metadatas": [[meta]],
        }
        hit_type, looked_up, _similarity = cache.lookup(entry.query_text)
        assert hit_type == "hit"
        assert looked_up is not None
        assert looked_up.cached_action is not None
        assert looked_up.cached_action.service_data == {
            "brightness_pct": 30,
            "transition": 2,
        }

    def test_lookup_of_legacy_entry_without_schema_version(self):
        """Backward compat: entries persisted before P1-5 lack
        ``schema_version`` and ``service_data``. They must still decode
        into a ``CachedAction`` with an empty ``service_data`` dict."""
        cache, store = _make_response_cache()
        legacy_action_json = '{"service":"light/turn_on","entity_id":"light.bedroom"}'
        store.query.return_value = {
            "ids": [["legacy-1"]],
            "distances": [[0.02]],
            "documents": [["dim the bedroom lights"]],
            "metadatas": [
                [
                    {
                        "response_text": "Done.",
                        "agent_id": "light-agent",
                        "confidence": "0.97",
                        "hit_count": "1",
                        "entity_ids": "light.bedroom",
                        "created_at": "2025-01-01T00:00:00",
                        "last_accessed": "2025-01-01T00:00:00",
                        "language": "en",
                        "cached_action": legacy_action_json,
                    }
                ]
            ],
        }
        hit_type, looked_up, _ = cache.lookup("dim the bedroom lights")
        assert hit_type == "hit"
        assert looked_up is not None
        assert looked_up.cached_action is not None
        assert looked_up.cached_action.service_data == {}


class TestStoreResponseCacheWhitelist:
    async def test_non_whitelisted_keys_dropped(self):
        """Orchestrator must only persist whitelisted service_data keys."""
        from app.agents.orchestrator import OrchestratorAgent

        orch = OrchestratorAgent.__new__(OrchestratorAgent)
        stored: list[ResponseCacheEntry] = []

        async def fake_store(entry: ResponseCacheEntry) -> None:
            stored.append(entry)

        orch._cache_manager = MagicMock()
        orch._cache_manager.store_response_async = AsyncMock(side_effect=fake_store)

        action_executed = {
            "success": True,
            "action": "turn_on",
            "entity_id": "light.kitchen",
            "cacheable": True,
            "service_data": {
                "brightness_pct": 50,
                "transition": 2,
                "evil_key": "drop me",
                "__proto__": "drop me too",
            },
        }
        await orch._store_response_cache(
            user_text="turn on kitchen",
            speech="Done.",
            target_agent="light-agent",
            confidence=0.97,
            action_executed=action_executed,
            has_error=False,
            language="en",
        )

        assert len(stored) == 1
        entry = stored[0]
        assert entry.cached_action is not None
        # Only whitelisted keys survive; unknown keys are dropped.
        assert entry.cached_action.service_data == {
            "brightness_pct": 50,
            "transition": 2,
        }


# Async tests rely on the repo-wide asyncio_mode = auto (see pytest config).
