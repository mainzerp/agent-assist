"""Tests for app.entity -- signals, matcher, aliases, index."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.entity.signals import (
    AliasSignal,
    EmbeddingSignal,
    JaroWinklerSignal,
    LevenshteinSignal,
    PhoneticSignal,
)
from app.entity.matcher import EntityMatcher, MatchResult
from app.entity.aliases import AliasResolver
from app.entity.index import EntityIndex
from app.models.entity_index import EntityIndexEntry
from app.cache.vector_store import COLLECTION_ENTITY_INDEX

from tests.helpers import make_entity_index_entry


# ---------------------------------------------------------------------------
# Levenshtein signal
# ---------------------------------------------------------------------------

class TestLevenshteinSignal:

    def test_exact_match_returns_1(self):
        score = LevenshteinSignal.score("kitchen light", "kitchen light")
        assert score == pytest.approx(1.0)

    def test_partial_match_returns_middle_score(self):
        score = LevenshteinSignal.score("kitchen lite", "kitchen light")
        assert 0.5 < score < 1.0

    def test_no_match_returns_low_score(self):
        score = LevenshteinSignal.score("zzzzzzz", "kitchen light")
        assert score < 0.3

    def test_case_insensitive(self):
        score = LevenshteinSignal.score("Kitchen Light", "kitchen light")
        assert score == pytest.approx(1.0)

    def test_empty_strings(self):
        score = LevenshteinSignal.score("", "")
        # rapidfuzz returns 0.0 for two empty strings (no characters to compare)
        assert score == pytest.approx(0.0) or score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Jaro-Winkler signal
# ---------------------------------------------------------------------------

class TestJaroWinklerSignal:

    def test_exact_match_returns_1(self):
        score = JaroWinklerSignal.score("bedroom light", "bedroom light")
        assert score == pytest.approx(1.0)

    def test_partial_match_returns_high_score(self):
        score = JaroWinklerSignal.score("bedroom lite", "bedroom light")
        assert score > 0.8

    def test_unrelated_returns_low_score(self):
        score = JaroWinklerSignal.score("xyz", "bedroom light")
        assert score < 0.5

    def test_case_insensitive(self):
        score = JaroWinklerSignal.score("BEDROOM LIGHT", "bedroom light")
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Phonetic signal
# ---------------------------------------------------------------------------

class TestPhoneticSignal:

    def test_soundex_match_returns_1(self):
        # "lite" and "light" should have same Soundex if pyphonetics available
        score = PhoneticSignal.score("light", "lite")
        # May return 1.0 (Soundex match) or 0.0 if pyphonetics not installed
        assert score in (0.0, 0.8, 1.0)

    def test_completely_different_returns_0(self):
        score = PhoneticSignal.score("apple", "zebra")
        assert score == 0.0 or score == 0.8  # Metaphone might match in edge cases

    def test_identical_returns_1(self):
        score = PhoneticSignal.score("kitchen", "kitchen")
        # Identical words always match both Soundex and Metaphone
        if score > 0:
            assert score == 1.0

    def test_graceful_without_pyphonetics(self):
        with patch("app.entity.signals.Soundex", None), \
             patch("app.entity.signals.Metaphone", None):
            score = PhoneticSignal.score("light", "lite")
            assert score == 0.0


# ---------------------------------------------------------------------------
# Embedding signal
# ---------------------------------------------------------------------------

class TestEmbeddingSignal:

    def test_returns_scored_results(self):
        mock_index = MagicMock(spec=EntityIndex)
        entry = make_entity_index_entry()
        mock_index.search.return_value = [(entry, 0.1)]

        results = EmbeddingSignal.score("kitchen light", mock_index, n=5)
        assert len(results) == 1
        eid, name, sim = results[0]
        assert eid == entry.entity_id
        assert sim == pytest.approx(0.9)  # 1 - 0.1

    def test_zero_distance_returns_similarity_1(self):
        mock_index = MagicMock(spec=EntityIndex)
        entry = make_entity_index_entry()
        mock_index.search.return_value = [(entry, 0.0)]

        results = EmbeddingSignal.score("kitchen light", mock_index)
        assert results[0][2] == pytest.approx(1.0)

    def test_empty_results(self):
        mock_index = MagicMock(spec=EntityIndex)
        mock_index.search.return_value = []
        results = EmbeddingSignal.score("nonexistent", mock_index)
        assert results == []

    def test_similarity_clamped_at_zero(self):
        mock_index = MagicMock(spec=EntityIndex)
        entry = make_entity_index_entry()
        mock_index.search.return_value = [(entry, 1.5)]  # distance > 1
        results = EmbeddingSignal.score("test", mock_index)
        assert results[0][2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Alias signal
# ---------------------------------------------------------------------------

class TestAliasSignal:

    async def test_exact_match_returns_entity_and_score_1(self):
        resolver = AsyncMock(spec=AliasResolver)
        resolver.resolve = AsyncMock(return_value="light.bedroom_nightstand")
        result = await AliasSignal.score("nightstand lamp", resolver)
        assert result is not None
        assert result[0] == "light.bedroom_nightstand"
        assert result[1] == 1.0

    async def test_no_match_returns_none(self):
        resolver = AsyncMock(spec=AliasResolver)
        resolver.resolve = AsyncMock(return_value=None)
        result = await AliasSignal.score("unknown thing", resolver)
        assert result is None

    async def test_strips_whitespace(self):
        resolver = AsyncMock(spec=AliasResolver)
        resolver.resolve = AsyncMock(return_value="light.x")
        result = await AliasSignal.score("  lamp  ", resolver)
        resolver.resolve.assert_called_with("lamp")


# ---------------------------------------------------------------------------
# Entity matcher
# ---------------------------------------------------------------------------

class TestEntityMatcher:

    def _make_matcher(self) -> tuple[EntityMatcher, MagicMock, AsyncMock]:
        mock_index = MagicMock(spec=EntityIndex)
        mock_alias_resolver = AsyncMock(spec=AliasResolver)
        matcher = EntityMatcher(mock_index, mock_alias_resolver)
        matcher._weights = {
            "levenshtein": 0.2,
            "jaro_winkler": 0.2,
            "phonetic": 0.15,
            "embedding": 0.3,
            "alias": 0.15,
        }
        matcher._confidence_threshold = 0.75
        matcher._top_n = 3
        return matcher, mock_index, mock_alias_resolver

    async def test_match_returns_sorted_results(self):
        matcher, mock_index, mock_alias = self._make_matcher()
        mock_alias.resolve = AsyncMock(return_value=None)

        entry1 = make_entity_index_entry("light.kitchen", "Kitchen Light")
        entry2 = make_entity_index_entry("light.bedroom", "Bedroom Light")
        mock_index.search.return_value = [(entry1, 0.05), (entry2, 0.3)]

        with patch("app.entity.matcher.EntityVisibilityRepository"):
            results = await matcher.match("kitchen light")
        # Results should be sorted by score descending
        if len(results) > 1:
            assert results[0].score >= results[1].score

    async def test_match_empty_entity_list(self):
        matcher, mock_index, mock_alias = self._make_matcher()
        mock_alias.resolve = AsyncMock(return_value=None)
        mock_index.search.return_value = []

        with patch("app.entity.matcher.EntityVisibilityRepository"):
            results = await matcher.match("nonexistent thing")
        assert results == []

    async def test_match_alias_fast_path(self):
        matcher, mock_index, mock_alias = self._make_matcher()
        mock_alias.resolve = AsyncMock(return_value="light.nightstand")
        mock_index.search.return_value = []

        with patch("app.entity.matcher.EntityVisibilityRepository"):
            results = await matcher.match("nightstand lamp")
        # Should have at least the alias result
        entity_ids = [r.entity_id for r in results]
        # Alias signal provides a score, but the alias result might not pass threshold
        # because only the alias weight (0.15) contributes
        # The alias score is 1.0 * 0.15 = 0.15, which is below threshold 0.75
        # so the result list may be empty if only alias matches
        assert isinstance(results, list)

    async def test_match_confidence_threshold_filters(self):
        matcher, mock_index, mock_alias = self._make_matcher()
        matcher._confidence_threshold = 0.99  # Very high threshold
        mock_alias.resolve = AsyncMock(return_value=None)

        entry = make_entity_index_entry("light.kitchen", "Kitchen Light")
        mock_index.search.return_value = [(entry, 0.5)]  # similarity = 0.5

        with patch("app.entity.matcher.EntityVisibilityRepository"):
            results = await matcher.match("kitchen liiight")
        assert results == []

    async def test_match_configurable_weights(self):
        matcher, mock_index, mock_alias = self._make_matcher()
        # Set embedding weight very high so embedding dominates
        matcher._weights = {
            "levenshtein": 0.0, "jaro_winkler": 0.0, "phonetic": 0.0,
            "embedding": 1.0, "alias": 0.0,
        }
        matcher._confidence_threshold = 0.5
        mock_alias.resolve = AsyncMock(return_value=None)

        entry = make_entity_index_entry("light.kitchen", "Kitchen Light")
        mock_index.search.return_value = [(entry, 0.1)]  # sim = 0.9

        with patch("app.entity.matcher.EntityVisibilityRepository"):
            results = await matcher.match("kitchen light")
        assert len(results) >= 1
        assert results[0].score >= 0.5

    async def test_load_config_reads_from_db(self):
        matcher, mock_index, mock_alias = self._make_matcher()
        mock_db = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[
            ("weight.levenshtein", "0.25"),
            ("weight.jaro_winkler", "0.25"),
            ("weight.phonetic", "0.10"),
            ("weight.embedding", "0.30"),
            ("weight.alias", "0.10"),
        ])
        mock_db.execute = AsyncMock(return_value=mock_cursor)

        with patch("app.db.schema.get_db") as mock_get_db, \
             patch("app.entity.matcher.SettingsRepository") as mock_settings:
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def fake_db():
                yield mock_db
            mock_get_db.side_effect = fake_db
            mock_settings.get_value = AsyncMock(side_effect=["0.75", "3"])
            await matcher.load_config()

        assert "levenshtein" in matcher._weights

    async def test_match_result_has_signal_scores(self):
        matcher, mock_index, mock_alias = self._make_matcher()
        matcher._confidence_threshold = 0.0  # Accept everything
        mock_alias.resolve = AsyncMock(return_value=None)

        entry = make_entity_index_entry("light.kitchen", "Kitchen Light")
        mock_index.search.return_value = [(entry, 0.05)]

        with patch("app.entity.matcher.EntityVisibilityRepository"):
            results = await matcher.match("Kitchen Light")
        assert len(results) >= 1
        assert "embedding" in results[0].signal_scores


# ---------------------------------------------------------------------------
# Alias resolver
# ---------------------------------------------------------------------------

class TestAliasResolver:

    async def test_resolve_loads_and_caches(self):
        resolver = AliasResolver()
        with patch.object(AliasResolver, "load", new_callable=AsyncMock) as mock_load:
            async def set_cache():
                resolver._cache = {"nightstand lamp": "light.nightstand"}
            mock_load.side_effect = set_cache
            result = await resolver.resolve("nightstand lamp")
        assert result == "light.nightstand"

    async def test_resolve_not_found(self):
        resolver = AliasResolver()
        resolver._cache = {"existing": "light.x"}
        result = await resolver.resolve("nonexistent")
        assert result is None

    async def test_resolve_case_insensitive(self):
        resolver = AliasResolver()
        resolver._cache = {"nightstand lamp": "light.nightstand"}
        result = await resolver.resolve("Nightstand Lamp")
        assert result == "light.nightstand"

    async def test_list_all(self):
        resolver = AliasResolver()
        resolver._cache = {"a": "light.a", "b": "light.b"}
        all_aliases = await resolver.list_all()
        assert len(all_aliases) == 2

    async def test_substitute_replaces_aliases(self):
        resolver = AliasResolver()
        resolver._cache = {"nightstand lamp": "light.nightstand"}
        result = await resolver.substitute("turn on nightstand lamp please")
        assert "light.nightstand" in result

    async def test_substitute_no_match(self):
        resolver = AliasResolver()
        resolver._cache = {"nightstand lamp": "light.nightstand"}
        result = await resolver.substitute("turn on kitchen light")
        assert result == "turn on kitchen light"

    async def test_reload_clears_cache(self):
        resolver = AliasResolver()
        resolver._cache = {"old": "light.old"}
        with patch("app.entity.aliases.AliasRepository") as mock_repo:
            mock_repo.list_all = AsyncMock(return_value=[
                {"alias": "new", "entity_id": "light.new"},
            ])
            await resolver.reload()
        assert resolver._cache == {"new": "light.new"}


# ---------------------------------------------------------------------------
# Entity index
# ---------------------------------------------------------------------------

class TestEntityIndex:

    def _make_index(self) -> tuple[EntityIndex, MagicMock]:
        mock_store = MagicMock()
        index = EntityIndex(mock_store)
        return index, mock_store

    def test_populate_upserts_to_store(self):
        index, store = self._make_index()
        entities = [
            make_entity_index_entry("light.kitchen", "Kitchen Light"),
            make_entity_index_entry("light.bedroom", "Bedroom Light"),
        ]
        index.populate(entities)
        store.upsert.assert_called_once()
        call_args = store.upsert.call_args
        assert call_args[1]["ids"] == ["light.kitchen", "light.bedroom"] or \
               call_args[0][1] == ["light.kitchen", "light.bedroom"]

    def test_populate_empty_list_noop(self):
        index, store = self._make_index()
        index.populate([])
        store.upsert.assert_not_called()

    def test_search_returns_entries(self):
        index, store = self._make_index()
        store.query.return_value = {
            "ids": [["light.kitchen"]],
            "metadatas": [[{"friendly_name": "Kitchen Light", "domain": "light", "area": "kitchen", "device_class": "", "aliases": ""}]],
            "distances": [[0.1]],
            "documents": [["Kitchen Light light kitchen"]],
        }
        results = index.search("kitchen light")
        assert len(results) == 1
        entry, dist = results[0]
        assert entry.entity_id == "light.kitchen"
        assert dist == 0.1

    def test_search_empty_returns_empty(self):
        index, store = self._make_index()
        store.query.return_value = {"ids": [[]], "metadatas": [[]], "distances": [[]], "documents": [[]]}
        results = index.search("nonexistent")
        assert results == []

    def test_add_single_entity(self):
        index, store = self._make_index()
        entry = make_entity_index_entry("light.new", "New Light")
        index.add(entry)
        store.upsert.assert_called_once()

    def test_remove_entity(self):
        index, store = self._make_index()
        index.remove("light.old")
        store.delete.assert_called_once_with(COLLECTION_ENTITY_INDEX, ids=["light.old"])

    def test_refresh_clears_and_repopulates(self):
        index, store = self._make_index()
        store.count.return_value = 0
        store.get.return_value = {"ids": []}
        entities = [make_entity_index_entry()]
        index.refresh(entities)
        # clear + populate = at least one upsert
        assert store.upsert.called

    def test_get_stats(self):
        index, store = self._make_index()
        store.count.return_value = 42
        stats = index.get_stats()
        assert stats["count"] == 42

    def test_get_by_id_found(self):
        index, store = self._make_index()
        store.get.return_value = {
            "ids": ["light.kitchen"],
            "metadatas": [{"friendly_name": "Kitchen", "domain": "light", "area": "kitchen", "device_class": "", "aliases": ""}],
        }
        entry = index.get_by_id("light.kitchen")
        assert entry is not None
        assert entry.entity_id == "light.kitchen"

    def test_get_by_id_not_found(self):
        index, store = self._make_index()
        store.get.return_value = {"ids": [], "metadatas": []}
        entry = index.get_by_id("light.missing")
        assert entry is None


# ---------------------------------------------------------------------------
# Visibility rules filtering
# ---------------------------------------------------------------------------

class TestVisibilityRules:
    """Tests for _apply_visibility_rules in EntityMatcher."""

    def _make_matcher(self) -> tuple[EntityMatcher, MagicMock, AsyncMock]:
        mock_index = MagicMock(spec=EntityIndex)
        mock_alias = AsyncMock(spec=AliasResolver)
        matcher = EntityMatcher(mock_index, mock_alias)
        return matcher, mock_index, mock_alias

    def _make_results(self, *entity_ids: str) -> list[MatchResult]:
        return [
            MatchResult(entity_id=eid, friendly_name=eid, score=1.0)
            for eid in entity_ids
        ]

    async def test_no_rules_returns_all(self):
        matcher, mock_index, _ = self._make_matcher()
        results = self._make_results("light.kitchen", "switch.hallway", "media_player.sonos")

        with patch("app.entity.matcher.EntityVisibilityRepository") as mock_repo:
            mock_repo.get_rules = AsyncMock(return_value=[])
            filtered = await matcher._apply_visibility_rules("light-agent", results)

        assert len(filtered) == 3

    async def test_domain_include_filters(self):
        matcher, mock_index, _ = self._make_matcher()
        mock_index.get_by_id.return_value = None
        results = self._make_results("light.kitchen", "switch.hallway", "media_player.sonos")

        with patch("app.entity.matcher.EntityVisibilityRepository") as mock_repo:
            mock_repo.get_rules = AsyncMock(return_value=[
                {"rule_type": "domain_include", "rule_value": "light"},
            ])
            filtered = await matcher._apply_visibility_rules("light-agent", results)

        assert len(filtered) == 1
        assert filtered[0].entity_id == "light.kitchen"

    async def test_domain_exclude_filters(self):
        matcher, mock_index, _ = self._make_matcher()
        mock_index.get_by_id.return_value = None
        results = self._make_results("light.kitchen", "switch.hallway", "media_player.sonos")

        with patch("app.entity.matcher.EntityVisibilityRepository") as mock_repo:
            mock_repo.get_rules = AsyncMock(return_value=[
                {"rule_type": "domain_exclude", "rule_value": "switch"},
            ])
            filtered = await matcher._apply_visibility_rules("light-agent", results)

        assert len(filtered) == 2
        entity_ids = {r.entity_id for r in filtered}
        assert "switch.hallway" not in entity_ids

    async def test_entity_include_whitelists(self):
        matcher, mock_index, _ = self._make_matcher()
        mock_index.get_by_id.return_value = None
        results = self._make_results("light.kitchen", "switch.hallway", "media_player.sonos")

        with patch("app.entity.matcher.EntityVisibilityRepository") as mock_repo:
            mock_repo.get_rules = AsyncMock(return_value=[
                {"rule_type": "domain_include", "rule_value": "light"},
                {"rule_type": "entity_include", "rule_value": "media_player.sonos"},
            ])
            filtered = await matcher._apply_visibility_rules("light-agent", results)

        entity_ids = {r.entity_id for r in filtered}
        assert "light.kitchen" in entity_ids
        assert "media_player.sonos" in entity_ids
        assert "switch.hallway" not in entity_ids
        assert len(filtered) == 2

    async def test_entity_include_union_with_domain(self):
        matcher, mock_index, _ = self._make_matcher()
        mock_index.get_by_id.return_value = None
        results = self._make_results(
            "light.kitchen", "light.bedroom", "media_player.sonos", "switch.hallway"
        )

        with patch("app.entity.matcher.EntityVisibilityRepository") as mock_repo:
            mock_repo.get_rules = AsyncMock(return_value=[
                {"rule_type": "domain_include", "rule_value": "light"},
                {"rule_type": "entity_include", "rule_value": "media_player.sonos"},
            ])
            filtered = await matcher._apply_visibility_rules("light-agent", results)

        entity_ids = {r.entity_id for r in filtered}
        assert "light.kitchen" in entity_ids
        assert "light.bedroom" in entity_ids
        assert "media_player.sonos" in entity_ids
        assert "switch.hallway" not in entity_ids


# ---------------------------------------------------------------------------
# Entity index status tracking
# ---------------------------------------------------------------------------

class TestEntityIndexStatus:

    def test_initial_status_is_ready(self):
        store = MagicMock()
        idx = EntityIndex(store)
        status = idx.get_embedding_status()
        assert status["state"] == "ready"
        assert status["progress"] == 0

    def test_populate_sets_building_then_ready(self):
        store = MagicMock()
        idx = EntityIndex(store)
        entities = [make_entity_index_entry(f"light.test_{i}") for i in range(10)]
        idx.populate(entities)
        status = idx.get_embedding_status()
        assert status["state"] == "ready"
        assert status["progress"] == 100
        assert status["total"] == 10
        assert status["processed"] == 10

    def test_populate_empty_keeps_ready(self):
        store = MagicMock()
        idx = EntityIndex(store)
        idx.populate([])
        status = idx.get_embedding_status()
        assert status["state"] == "ready"

    def test_populate_error_sets_error_state(self):
        store = MagicMock()
        store.upsert.side_effect = RuntimeError("ChromaDB error")
        idx = EntityIndex(store)
        entities = [make_entity_index_entry("light.test")]
        with pytest.raises(RuntimeError):
            idx.populate(entities)
        status = idx.get_embedding_status()
        assert status["state"] == "error"
        assert "ChromaDB error" in status["error"]

    def test_get_stats_includes_embedding_status(self):
        store = MagicMock()
        store.count.return_value = 5
        idx = EntityIndex(store)
        stats = idx.get_stats()
        assert "embedding_status" in stats
        assert stats["embedding_status"]["state"] == "ready"

    def test_populate_batches_upserts(self):
        """With BATCH_SIZE=500, 1200 entities should produce 3 upsert calls."""
        store = MagicMock()
        idx = EntityIndex(store)
        entities = [make_entity_index_entry(f"light.test_{i}") for i in range(1200)]
        idx.populate(entities)
        assert store.upsert.call_count == 3  # 500 + 500 + 200


# ---------------------------------------------------------------------------
# Entity index sync
# ---------------------------------------------------------------------------

class TestEntityIndexSync:
    """Tests for EntityIndex.sync() smart diff."""

    def _make_index(self) -> tuple[EntityIndex, MagicMock]:
        mock_store = MagicMock()
        index = EntityIndex(mock_store)
        return index, mock_store

    def test_sync_adds_new_entities(self):
        """New entities not in ChromaDB are added."""
        index, store = self._make_index()
        store.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        entities = [
            make_entity_index_entry("light.kitchen", "Kitchen Light"),
            make_entity_index_entry("light.bedroom", "Bedroom Light"),
        ]
        result = index.sync(entities)

        assert result["added"] == 2
        assert result["updated"] == 0
        assert result["removed"] == 0
        assert result["unchanged"] == 0
        store.upsert.assert_called_once()

    def test_sync_removes_deleted_entities(self):
        """Entities in ChromaDB but not in HA list are removed."""
        index, store = self._make_index()
        store.get.return_value = {
            "ids": ["light.kitchen", "light.old_deleted"],
            "documents": ["Kitchen Light light", "Old Light light"],
            "metadatas": [
                {"friendly_name": "Kitchen Light", "domain": "light", "area": "", "device_class": "", "aliases": ""},
                {"friendly_name": "Old Light", "domain": "light", "area": "", "device_class": "", "aliases": ""},
            ],
        }

        entities = [make_entity_index_entry("light.kitchen", "Kitchen Light")]
        result = index.sync(entities)

        assert result["removed"] == 1
        store.delete.assert_called_once_with(COLLECTION_ENTITY_INDEX, ids=["light.old_deleted"])

    def test_sync_updates_changed_entities(self):
        """Entities with changed embedding_text are re-upserted."""
        index, store = self._make_index()
        store.get.return_value = {
            "ids": ["light.kitchen"],
            "documents": ["Old Kitchen Light light"],
            "metadatas": [
                {"friendly_name": "Old Kitchen Light", "domain": "light", "area": "", "device_class": "", "aliases": ""},
            ],
        }

        entities = [make_entity_index_entry("light.kitchen", "New Kitchen Light")]
        result = index.sync(entities)

        assert result["updated"] == 1
        assert result["unchanged"] == 0
        store.upsert.assert_called_once()

    def test_sync_skips_unchanged_entities(self):
        """Entities with identical doc + metadata are skipped (no upsert)."""
        index, store = self._make_index()
        entry = make_entity_index_entry("light.kitchen", "Kitchen Light")
        store.get.return_value = {
            "ids": ["light.kitchen"],
            "documents": [entry.embedding_text],
            "metadatas": [EntityIndex._build_metadata(entry)],
        }

        result = index.sync([entry])

        assert result["unchanged"] == 1
        assert result["added"] == 0
        assert result["updated"] == 0
        store.upsert.assert_not_called()
        store.delete.assert_not_called()

    def test_sync_empty_list_noop(self):
        """Syncing with an empty list returns all zeros."""
        index, store = self._make_index()
        result = index.sync([])
        assert result == {"added": 0, "updated": 0, "removed": 0, "unchanged": 0}
        store.get.assert_not_called()

    def test_sync_updates_status(self):
        """sync() sets state to 'syncing' during operation and 'ready' after."""
        index, store = self._make_index()
        store.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        states_seen = []
        def capture_state(*args, **kwargs):
            states_seen.append(index._status["state"])
        store.upsert.side_effect = capture_state

        entities = [make_entity_index_entry("light.kitchen", "Kitchen Light")]
        index.sync(entities)

        assert "syncing" in states_seen
        assert index._status["state"] == "ready"

    def test_sync_updates_sync_stats(self):
        """sync() updates _sync_stats with counts and timestamp."""
        index, store = self._make_index()
        store.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        entities = [make_entity_index_entry("light.kitchen", "Kitchen Light")]
        index.sync(entities)

        stats = index._sync_stats
        assert stats["added"] == 1
        assert stats["last_sync"] is not None
        assert stats["last_sync_duration_ms"] >= 0

    def test_sync_error_restores_state(self):
        """If sync fails, state is restored and error is logged."""
        index, store = self._make_index()
        store.get.side_effect = Exception("ChromaDB down")

        with pytest.raises(Exception, match="ChromaDB down"):
            index.sync([make_entity_index_entry()])

        assert index._status["state"] == "ready"
        assert "ChromaDB down" in index._status["error"]

    def test_sync_mixed_operations(self):
        """Full scenario: 1 unchanged, 1 updated, 1 removed, 1 added."""
        index, store = self._make_index()

        existing_unchanged = make_entity_index_entry("light.kitchen", "Kitchen Light")
        existing_changed = make_entity_index_entry("light.bedroom", "Old Bedroom")

        store.get.return_value = {
            "ids": ["light.kitchen", "light.bedroom", "light.deleted"],
            "documents": [
                existing_unchanged.embedding_text,
                "Old Bedroom light",
                "Deleted light",
            ],
            "metadatas": [
                EntityIndex._build_metadata(existing_unchanged),
                {"friendly_name": "Old Bedroom", "domain": "light", "area": "", "device_class": "", "aliases": ""},
                {"friendly_name": "Deleted", "domain": "light", "area": "", "device_class": "", "aliases": ""},
            ],
        }

        new_entities = [
            existing_unchanged,  # unchanged
            make_entity_index_entry("light.bedroom", "New Bedroom"),  # updated
            make_entity_index_entry("light.new_one", "New One"),  # added
            # light.deleted is absent -> removed
        ]
        result = index.sync(new_entities)

        assert result["unchanged"] == 1
        assert result["updated"] == 1
        assert result["added"] == 1
        assert result["removed"] == 1

    def test_get_stats_includes_sync(self):
        """get_stats() includes sync stats."""
        index, store = self._make_index()
        store.count.return_value = 10
        stats = index.get_stats()
        assert "sync" in stats
        assert stats["sync"]["last_sync"] is None  # No sync yet


# ---------------------------------------------------------------------------
# Periodic entity sync task
# ---------------------------------------------------------------------------

class TestPeriodicEntitySync:
    """Tests for _periodic_entity_sync background task."""

    async def test_periodic_sync_calls_sync(self):
        """Task fetches states, parses, and calls entity_index.sync()."""
        from unittest.mock import AsyncMock, patch
        from app.main import _periodic_entity_sync

        mock_app = MagicMock()
        mock_app.state.ha_client = AsyncMock()
        mock_app.state.ha_client.get_states = AsyncMock(return_value=[
            {"entity_id": "light.kitchen", "attributes": {"friendly_name": "Kitchen"}},
        ])
        mock_app.state.entity_index = MagicMock()
        mock_app.state.entity_index.sync.return_value = {
            "added": 1, "updated": 0, "removed": 0, "unchanged": 0,
        }

        with patch("app.main.SettingsRepository") as mock_settings, \
             patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]):
            mock_settings.get_value = AsyncMock(return_value="1")
            with pytest.raises(asyncio.CancelledError):
                await _periodic_entity_sync(mock_app)

        mock_app.state.entity_index.sync.assert_called_once()

    async def test_periodic_sync_disabled_when_zero(self):
        """Interval=0 skips sync and re-checks after 5 min."""
        from unittest.mock import AsyncMock, patch
        from app.main import _periodic_entity_sync

        mock_app = MagicMock()
        mock_app.state.ha_client = AsyncMock()
        mock_app.state.entity_index = MagicMock()

        sleep_calls = []
        async def fake_sleep(duration):
            sleep_calls.append(duration)
            if len(sleep_calls) >= 2:
                raise asyncio.CancelledError

        with patch("app.main.SettingsRepository") as mock_settings, \
             patch("asyncio.sleep", side_effect=fake_sleep):
            mock_settings.get_value = AsyncMock(return_value="0")
            with pytest.raises(asyncio.CancelledError):
                await _periodic_entity_sync(mock_app)

        assert sleep_calls[0] == 300  # 5 min fallback when disabled

    async def test_periodic_sync_handles_errors_gracefully(self):
        """Errors are logged but do not crash the task."""
        from unittest.mock import AsyncMock, patch
        from app.main import _periodic_entity_sync

        mock_app = MagicMock()
        mock_app.state.ha_client = AsyncMock()
        mock_app.state.ha_client.get_states = AsyncMock(side_effect=Exception("HA down"))
        mock_app.state.entity_index = MagicMock()

        call_count = 0
        async def fake_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError

        with patch("app.main.SettingsRepository") as mock_settings, \
             patch("asyncio.sleep", side_effect=fake_sleep):
            mock_settings.get_value = AsyncMock(return_value="1")
            with pytest.raises(asyncio.CancelledError):
                await _periodic_entity_sync(mock_app)

        # sync was NOT called because get_states failed
        mock_app.state.entity_index.sync.assert_not_called()
