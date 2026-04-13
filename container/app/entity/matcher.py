"""Hybrid entity matching engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.db.repository import SettingsRepository, EntityVisibilityRepository
from app.entity.index import EntityIndex
from app.entity.aliases import AliasResolver
from app.entity.signals import LevenshteinSignal, JaroWinklerSignal, PhoneticSignal, EmbeddingSignal, AliasSignal
from app.models.entity_index import EntityIndexEntry

logger = logging.getLogger(__name__)

# Domains where device_class filtering applies (sensor-like domains)
DEVICE_CLASS_DOMAINS = {"sensor", "binary_sensor", "cover", "number"}


@dataclass
class MatchResult:
    """Result of entity matching with per-signal scores."""
    entity_id: str
    friendly_name: str
    score: float
    signal_scores: dict[str, float] = field(default_factory=dict)


class EntityMatcher:
    """Hybrid entity matcher combining fuzzy, alias, and embedding signals.

    Uses all 5 signals (Levenshtein, Jaro-Winkler, Phonetic, Embedding, Alias).
    Weights are loaded from entity_matching_config table.
    """

    def __init__(
        self,
        entity_index: EntityIndex,
        alias_resolver: AliasResolver,
    ) -> None:
        self._entity_index = entity_index
        self._alias_resolver = alias_resolver
        self._weights: dict[str, float] = {}
        self._confidence_threshold: float = 0.75
        self._top_n: int = 3

    async def load_config(self) -> None:
        """Load matching weights and thresholds from DB."""
        from app.db.schema import get_db
        async with get_db() as db:
            cursor = await db.execute("SELECT key, value FROM entity_matching_config")
            rows = await cursor.fetchall()
            raw_weights = {row[0]: float(row[1]) for row in rows}

        # All 5 active signals
        active_keys = [
            "weight.levenshtein", "weight.jaro_winkler", "weight.phonetic",
            "weight.embedding", "weight.alias",
        ]
        active_raw = {k: raw_weights.get(k, 0.0) for k in active_keys}
        total = sum(active_raw.values())
        if total > 0:
            self._weights = {k.split(".")[-1]: v / total for k, v in active_raw.items()}
        else:
            self._weights = {
                "levenshtein": 0.2, "jaro_winkler": 0.2, "phonetic": 0.2,
                "embedding": 0.2, "alias": 0.2,
            }

        self._confidence_threshold = float(
            await SettingsRepository.get_value("entity_matching.confidence_threshold", "0.75")
        )
        self._top_n = int(
            await SettingsRepository.get_value("entity_matching.top_n_candidates", "3")
        )
        logger.info("Entity matcher config: weights=%s threshold=%s", self._weights, self._confidence_threshold)

    async def match(
        self,
        query: str,
        candidates: list[EntityIndexEntry] | None = None,
        agent_id: str | None = None,
    ) -> list[MatchResult]:
        """Match a query against entities using all active signals.

        Args:
            query: User text (e.g. "kitchen light", "living room lamp").
            candidates: Optional pre-filtered candidates. If None, uses entity_index search.
            agent_id: Optional agent ID for entity visibility filtering.

        Returns:
            Sorted list of MatchResult (highest score first), filtered by confidence threshold.
        """
        results: dict[str, MatchResult] = {}

        # 1. Alias signal (fast path -- exact match)
        alias_result = await AliasSignal.score(query, self._alias_resolver)
        if alias_result:
            entity_id, alias_score = alias_result
            results[entity_id] = MatchResult(
                entity_id=entity_id,
                friendly_name="",
                score=0.0,
                signal_scores={"alias": alias_score},
            )

        # 2. Embedding signal -- vector search
        try:
            embedding_results = EmbeddingSignal.score(query, self._entity_index, n=self._top_n * 2)
        except Exception:
            logger.warning("Embedding signal unavailable, proceeding with remaining signals")
            embedding_results = []
        for entity_id, friendly_name, emb_score in embedding_results:
            if entity_id in results:
                results[entity_id].signal_scores["embedding"] = emb_score
                results[entity_id].friendly_name = friendly_name
            else:
                results[entity_id] = MatchResult(
                    entity_id=entity_id,
                    friendly_name=friendly_name,
                    score=0.0,
                    signal_scores={"embedding": emb_score},
                )

        # 3. Levenshtein signal -- compare query against each candidate friendly_name
        for entity_id, result in results.items():
            if result.friendly_name:
                lev_score = LevenshteinSignal.score(query, result.friendly_name)
                result.signal_scores["levenshtein"] = lev_score

        # 3b. Jaro-Winkler signal
        for entity_id, result in results.items():
            if result.friendly_name:
                jw_score = JaroWinklerSignal.score(query, result.friendly_name)
                result.signal_scores["jaro_winkler"] = jw_score

        # 3c. Phonetic signal
        for entity_id, result in results.items():
            if result.friendly_name:
                ph_score = PhoneticSignal.score(query, result.friendly_name)
                result.signal_scores["phonetic"] = ph_score

        # Compute weighted score for each candidate
        for result in results.values():
            weighted_sum = 0.0
            for signal_name, weight in self._weights.items():
                signal_score = result.signal_scores.get(signal_name, 0.0)
                weighted_sum += weight * signal_score
            result.score = weighted_sum

        # Filter by confidence and sort
        filtered = [r for r in results.values() if r.score >= self._confidence_threshold]
        filtered.sort(key=lambda r: r.score, reverse=True)

        top_results = filtered[:self._top_n]

        # Apply entity visibility filtering if agent_id is provided
        if agent_id and top_results:
            top_results = await self._apply_visibility_rules(agent_id, top_results)

        return top_results

    async def _apply_visibility_rules(
        self,
        agent_id: str,
        results: list[MatchResult],
    ) -> list[MatchResult]:
        """Filter match results by agent entity visibility rules.

        Rule types: domain_include, domain_exclude, area_include, area_exclude,
        device_class_include, device_class_exclude.
        No rules = no filtering (full access).
        """
        rules = await EntityVisibilityRepository.get_rules(agent_id)
        if not rules:
            return results

        domain_include = set()
        domain_exclude = set()
        area_include = set()
        area_exclude = set()
        entity_include = set()
        device_class_include = set()
        device_class_exclude = set()
        for rule in rules:
            rt = rule["rule_type"]
            rv = rule["rule_value"]
            if rt == "domain_include":
                domain_include.add(rv)
            elif rt == "domain_exclude":
                domain_exclude.add(rv)
            elif rt == "area_include":
                area_include.add(rv)
            elif rt == "area_exclude":
                area_exclude.add(rv)
            elif rt == "entity_include":
                entity_include.add(rv)
            elif rt == "device_class_include":
                device_class_include.add(rv)
            elif rt == "device_class_exclude":
                device_class_exclude.add(rv)

        filtered = []
        for result in results:
            entity_id = result.entity_id
            domain = entity_id.split(".")[0] if "." in entity_id else ""

            if domain_include and domain not in domain_include:
                continue
            if domain_exclude and domain in domain_exclude:
                continue

            # Entity index lookup for area and device_class checks
            entry = self._entity_index.get_by_id(entity_id)
            area = entry.area if entry else None
            if area_include and (area is None or area not in area_include):
                continue
            if area_exclude and area is not None and area in area_exclude:
                continue

            # Device class filtering (only for sensor-like domains)
            if device_class_include and domain in DEVICE_CLASS_DOMAINS:
                entity_dc = entry.device_class if entry else None
                if not entity_dc or entity_dc not in device_class_include:
                    continue
            if device_class_exclude:
                entity_dc = entry.device_class if entry else None
                if entity_dc and entity_dc in device_class_exclude:
                    continue

            filtered.append(result)

        # entity_include: union with domain/area-filtered results
        if entity_include:
            filtered_ids = {r.entity_id for r in filtered}
            for r in results:
                if r.entity_id in entity_include and r.entity_id not in filtered_ids:
                    filtered.append(r)

        return filtered
