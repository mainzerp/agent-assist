"""Export and import helpers for the routing and response cache tiers.

This module owns serialisation, envelope validation and the upsert
loop. It is deliberately framework-agnostic (no FastAPI imports) so the
admin API routes in :mod:`app.api.routes.cache_api` can stay thin and
unit tests can exercise the behaviour without spinning up a Chroma
backend or HTTP layer.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.cache.vector_store import (
    COLLECTION_RESPONSE_CACHE,
    COLLECTION_ROUTING_CACHE,
)
from app.models.cache import CachedAction

if TYPE_CHECKING:
    from app.cache.cache_manager import CacheManager

logger = logging.getLogger(__name__)


SUPPORTED_FORMAT_VERSION: int = 2  # bumped in 0.21.0 for the action/response rename
_LEGACY_FORMAT_VERSION: int = 1
ROUTING_SCHEMA_VERSION: int = 1
ACTION_SCHEMA_VERSION: int = 2  # mirrors response_cache._RESPONSE_CACHE_SCHEMA_VERSION
# Legacy alias retained for one minor; new code should use ACTION_SCHEMA_VERSION.
RESPONSE_SCHEMA_VERSION: int = ACTION_SCHEMA_VERSION

# Canonical tier names accepted by the API/envelope layer. ``response`` is
# kept as a legacy alias for ``action`` and is normalised away by
# :func:`_canonical_tier` before any further dispatch.
ALLOWED_TIERS: tuple[str, ...] = ("routing", "action", "response")
EXPORT_PAGE_SIZE: int = 1000
IMPORT_BATCH_SIZE: int = 500
MAX_IMPORT_BYTES: int = 50 * 1024 * 1024  # 50 MiB
EXPORT_FORMAT_TAG: str = "agent-assist.cache"

_TIER_ALIASES: dict[str, str] = {"response": "action"}


def _canonical_tier(name: str) -> str:
    """Return the canonical tier name (collapses legacy ``response`` -> ``action``)."""

    return _TIER_ALIASES.get(name, name)


_TIER_TO_COLLECTION: dict[str, str] = {
    "routing": COLLECTION_ROUTING_CACHE,
    "action": COLLECTION_RESPONSE_CACHE,
}

_TIER_SCHEMA_VERSION: dict[str, int] = {
    "routing": ROUTING_SCHEMA_VERSION,
    "action": ACTION_SCHEMA_VERSION,
}


class ImportValidationError(ValueError):
    """Raised when the envelope itself is malformed or unsupported.

    Per-entry problems are collected as warnings; only envelope-level
    rejection raises this. The route layer maps it to HTTP 400.
    """


@dataclass
class TierImportResult:
    imported: int = 0
    skipped: int = 0
    re_embedded: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ImportSummary:
    mode: str  # "merge" | "replace"
    format_version: int
    tiers: dict[str, TierImportResult] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def build_export_filename(tiers: list[str], now: datetime) -> str:
    """Return the suggested attachment filename for an export."""

    canonical = [_canonical_tier(t) for t in tiers]
    canonical_set = set(canonical)
    all_canonical = {t for t in ALLOWED_TIERS if t != "response"}
    tag = "all" if canonical_set == all_canonical else "-".join(canonical) or "empty"
    ts = now.strftime("%Y%m%d%H%M%S")
    return f"agent-assist-cache-{tag}-{ts}.json"


def _detect_embedding_dim(vector_store, collection_name: str) -> int | None:
    """Best-effort detection of the current embedding dimension.

    Returns ``None`` for empty collections or when Chroma omits the
    embedding payload.
    """

    try:
        page = vector_store.get(
            collection_name,
            include=["embeddings"],
            limit=1,
        )
    except Exception:
        logger.warning("Failed to detect embedding dim for %s", collection_name, exc_info=True)
        return None
    embeddings = page.get("embeddings") if page else None
    if not embeddings:
        return None
    first = embeddings[0]
    if first is None:
        return None
    try:
        return len(first)
    except TypeError:
        return None


def _export_tier_pages(vector_store, collection_name: str) -> Iterator[dict]:
    """Yield one normalised entry dict at a time for ``collection_name``."""

    offset = 0
    while True:
        page = vector_store.get(
            collection_name,
            include=["embeddings", "documents", "metadatas"],
            limit=EXPORT_PAGE_SIZE,
            offset=offset,
        )
        ids = (page.get("ids") if page else None) or []
        if not ids:
            return
        documents = page.get("documents") or [None] * len(ids)
        metadatas = page.get("metadatas") or [None] * len(ids)
        embeddings = page.get("embeddings")
        for i, entry_id in enumerate(ids):
            entry: dict = {
                "id": entry_id,
                "document": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
            }
            if embeddings is not None and i < len(embeddings):
                emb = embeddings[i]
                if emb is not None:
                    with contextlib.suppress(TypeError):
                        entry["embedding"] = list(emb)
            yield entry
        if len(ids) < EXPORT_PAGE_SIZE:
            return
        offset += EXPORT_PAGE_SIZE


def _embedding_model_name() -> str | None:
    """Return the configured embedding model name, or ``None`` on error.

    Reads the singleton without forcing initialisation; export must not
    block on embedding-engine setup.
    """

    try:
        from app.cache import embedding as _embedding_mod

        engine = getattr(_embedding_mod, "_engine", None)
        if engine is None:
            return None
        return getattr(engine, "_model_name", None)
    except Exception:
        return None


def iter_export_chunks(
    cache_manager: CacheManager,
    tiers: list[str],
    *,
    app_version: str,
) -> Iterator[bytes]:
    """Yield UTF-8 JSON bytes that together form one export envelope.

    Streams pages of ``EXPORT_PAGE_SIZE`` per tier so memory stays
    bounded. The envelope shape is documented in the plan and the
    backup module README.
    """

    requested = [_canonical_tier(t) for t in tiers if _canonical_tier(t) in _TIER_TO_COLLECTION]
    vector_store = cache_manager._vector_store

    embedding_model = _embedding_model_name()
    embedding_dim: int | None = None
    for tier in requested:
        dim = _detect_embedding_dim(vector_store, _TIER_TO_COLLECTION[tier])
        if dim is not None:
            embedding_dim = dim
            break

    header = {
        "export_format": EXPORT_FORMAT_TAG,
        "format_version": SUPPORTED_FORMAT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "app_version": app_version,
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim,
        },
    }
    header_json = json.dumps(header)
    # Strip the closing brace so we can append ``"tiers": {...}`` after it.
    assert header_json.endswith("}")
    yield (header_json[:-1] + ',"tiers":{').encode("utf-8")

    for tier_index, tier in enumerate(requested):
        collection = _TIER_TO_COLLECTION[tier]
        try:
            count = vector_store.count(collection)
        except Exception:
            logger.warning(
                "Failed to count collection %s during export",
                collection,
                exc_info=True,
            )
            count = 0

        prefix = ""
        if tier_index > 0:
            prefix = ","
        yield (f'{prefix}"{tier}":{{"schema_version":{_TIER_SCHEMA_VERSION[tier]},"count":{count},"entries":[').encode()

        first = True
        for entry in _export_tier_pages(vector_store, collection):
            sep = "" if first else ","
            first = False
            yield (sep + json.dumps(entry)).encode("utf-8")
        yield b"]}"

    yield b"}}"


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def parse_envelope(raw: bytes) -> dict:
    """Parse + validate an export envelope.

    Raises :class:`ImportValidationError` for envelope-level problems:
    non-JSON payload, oversized payload, wrong ``export_format``,
    ``format_version`` newer than supported, missing ``tiers`` block, or
    a tier with a ``schema_version`` newer than supported.
    """

    if len(raw) > MAX_IMPORT_BYTES:
        raise ImportValidationError("payload too large")
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ImportValidationError(f"invalid JSON: {exc}") from exc
    if not isinstance(envelope, dict):
        raise ImportValidationError("envelope must be a JSON object")

    if envelope.get("export_format") != EXPORT_FORMAT_TAG:
        raise ImportValidationError(f"unsupported export_format; expected {EXPORT_FORMAT_TAG!r}")
    fmt_version = envelope.get("format_version")
    if not isinstance(fmt_version, int):
        raise ImportValidationError("format_version must be an integer")
    if fmt_version not in (_LEGACY_FORMAT_VERSION, SUPPORTED_FORMAT_VERSION):
        raise ImportValidationError(f"unsupported format_version {fmt_version}")

    tiers_block = envelope.get("tiers")
    if not isinstance(tiers_block, dict) or not tiers_block:
        raise ImportValidationError("tiers block missing or empty")

    # Canonicalise legacy ``response`` tier key so downstream code only
    # sees ``action``. This is what makes v1 envelopes round-trip.
    canonical_tiers: dict = {}
    for tier_name, tier_data in tiers_block.items():
        canonical_name = _canonical_tier(tier_name)
        if canonical_name in canonical_tiers:
            raise ImportValidationError(f"duplicate tier {canonical_name!r} after canonicalisation")
        canonical_tiers[canonical_name] = tier_data
    envelope["tiers"] = canonical_tiers
    tiers_block = canonical_tiers

    for tier_name, tier_data in tiers_block.items():
        if tier_name not in _TIER_TO_COLLECTION:
            # Unknown tier names are tolerated; they are simply skipped
            # during import. Only structural issues raise.
            continue
        if not isinstance(tier_data, dict):
            raise ImportValidationError(f"tier {tier_name!r} must be an object")
        schema = tier_data.get("schema_version")
        if not isinstance(schema, int):
            raise ImportValidationError(f"tier {tier_name!r} schema_version must be an integer")
        if schema > _TIER_SCHEMA_VERSION[tier_name]:
            raise ImportValidationError(f"tier {tier_name!r} schema_version {schema} is newer than supported")
        entries = tier_data.get("entries")
        if not isinstance(entries, list):
            raise ImportValidationError(f"tier {tier_name!r} entries must be a list")

    return envelope


def _coerce_metadata(meta: dict) -> dict:
    """Force every metadata value to ``str`` (Chroma stores strings)."""

    out: dict[str, str] = {}
    for k, v in meta.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, str):
            out[k] = v
        elif isinstance(v, bool):
            out[k] = "true" if v else "false"
        else:
            out[k] = str(v)
    return out


def _validate_routing_entry(
    entry: dict,
    expected_dim: int | None,
) -> tuple[dict | None, str | None, bool]:
    """Validate a routing entry. Returns ``(normalised, warning, force_re_embed)``.

    ``normalised`` is ``None`` when the entry must be skipped.
    ``force_re_embed`` is True when the embedding payload was dropped
    because of a dimension mismatch; the caller must then upsert with
    ``embeddings=None`` so Chroma re-embeds the document.
    """

    return _validate_entry(entry, expected_dim, tier="routing")


def _validate_response_entry(
    entry: dict,
    expected_dim: int | None,
) -> tuple[dict | None, str | None, bool]:
    return _validate_action_entry(entry, expected_dim)


def _validate_action_entry(
    entry: dict,
    expected_dim: int | None,
) -> tuple[dict | None, str | None, bool]:
    return _validate_entry(entry, expected_dim, tier="action")


def _validate_entry(
    entry: dict,
    expected_dim: int | None,
    *,
    tier: str,
) -> tuple[dict | None, str | None, bool]:
    if not isinstance(entry, dict):
        return None, f"{tier} entry: not a JSON object", False
    entry_id = entry.get("id")
    if not isinstance(entry_id, str) or not entry_id or len(entry_id) > 64:
        return None, f"{tier} entry: missing or invalid id", False
    document = entry.get("document")
    if not isinstance(document, str) or not document:
        return None, f"{tier} entry {entry_id}: missing or empty document", False
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        return None, f"{tier} entry {entry_id}: metadata missing or invalid", False

    agent_id = metadata.get("agent_id")
    if not isinstance(agent_id, str) or not agent_id:
        return None, f"{tier} entry {entry_id}: missing or empty agent_id", False

    warnings: list[str] = []
    if not metadata.get("language"):
        metadata = {**metadata, "language": "en"}
        warnings.append(f"{tier} entry {entry_id}: missing language; defaulted to en")

    if tier == "action":
        cached_action = metadata.get("cached_action")
        if isinstance(cached_action, str) and cached_action:
            try:
                CachedAction.model_validate_json(cached_action)
            except Exception as exc:
                return (
                    None,
                    f"action entry {entry_id}: invalid cached_action ({exc})",
                    False,
                )

    coerced_meta = _coerce_metadata(metadata)
    normalised: dict = {
        "id": entry_id,
        "document": document,
        "metadata": coerced_meta,
    }

    embedding = entry.get("embedding")
    force_re_embed = False
    if embedding is not None:
        if not isinstance(embedding, list):
            warnings.append(f"{tier} entry {entry_id}: embedding not a list; re-embedded")
            force_re_embed = True
        elif expected_dim is not None and len(embedding) != expected_dim:
            warnings.append(
                f"{tier} entry {entry_id}: embedding dim mismatch "
                f"(got {len(embedding)}, expected {expected_dim}); re-embedded"
            )
            force_re_embed = True
        else:
            normalised["embedding"] = embedding

    warning = "; ".join(warnings) if warnings else None
    return normalised, warning, force_re_embed


def _apply_tier_import(
    cache_manager: CacheManager,
    tier: str,
    entries: list[dict],
    *,
    mode: str,
    re_embed: bool,
    expected_dim: int | None,
) -> TierImportResult:
    result = TierImportResult()
    vector_store = cache_manager._vector_store
    collection = _TIER_TO_COLLECTION[tier]
    tier_cache = cache_manager._routing_cache if tier == "routing" else cache_manager._response_cache

    if mode == "replace":
        cache_manager.flush(tier)
    else:
        tier_cache.prepare_for_flush()

    validator = _validate_routing_entry if tier == "routing" else _validate_action_entry

    valid: list[tuple[dict, bool]] = []  # (entry, force_re_embed)
    for raw_entry in entries:
        normalised, warning, force = validator(raw_entry, expected_dim)
        if warning:
            result.warnings.append(warning)
        if normalised is None:
            result.skipped += 1
            continue
        valid.append((normalised, force))

    # Batch upserts. Mixing entries that need re-embedding with entries
    # that ship explicit embeddings is awkward (Chroma wants either all
    # or none per call), so we split into two batch streams.
    explicit: list[dict] = []
    re_embed_entries: list[dict] = []
    for entry, force in valid:
        if re_embed or force or "embedding" not in entry:
            re_embed_entries.append(entry)
        else:
            explicit.append(entry)

    def _flush_batch(batch: list[dict], with_embeddings: bool) -> None:
        if not batch:
            return
        ids = [e["id"] for e in batch]
        documents = [e["document"] for e in batch]
        metadatas = [e["metadata"] for e in batch]
        kwargs: dict = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        if with_embeddings:
            kwargs["embeddings"] = [e["embedding"] for e in batch]
        else:
            kwargs["embeddings"] = None
        vector_store.upsert(collection, **kwargs)

    for chunk_start in range(0, len(explicit), IMPORT_BATCH_SIZE):
        chunk = explicit[chunk_start : chunk_start + IMPORT_BATCH_SIZE]
        _flush_batch(chunk, with_embeddings=True)
        result.imported += len(chunk)

    for chunk_start in range(0, len(re_embed_entries), IMPORT_BATCH_SIZE):
        chunk = re_embed_entries[chunk_start : chunk_start + IMPORT_BATCH_SIZE]
        _flush_batch(chunk, with_embeddings=False)
        result.imported += len(chunk)
        result.re_embedded += len(chunk)

    try:
        tier_cache._enforce_lru()
    except Exception:
        logger.warning("LRU enforcement after import failed for %s", tier, exc_info=True)

    return result


async def import_envelope(
    cache_manager: CacheManager,
    envelope: dict,
    *,
    mode: str,
    tiers: list[str],
    re_embed: bool,
) -> ImportSummary:
    """Apply a parsed envelope to the cache.

    Envelope-level validation must already have been done via
    :func:`parse_envelope`. Per-entry validation runs here and is
    surfaced in :class:`TierImportResult` warnings.
    """

    import asyncio

    if mode not in ("merge", "replace"):
        raise ImportValidationError(f"unsupported import mode {mode!r}")
    requested = [_canonical_tier(t) for t in tiers if _canonical_tier(t) in _TIER_TO_COLLECTION]
    if not requested:
        raise ImportValidationError("no supported tiers in import request")

    summary = ImportSummary(
        mode=mode,
        format_version=int(envelope.get("format_version", SUPPORTED_FORMAT_VERSION)),
    )
    raw_tiers_block = envelope.get("tiers") or {}
    # Defensive canonicalisation in case the caller did not run the
    # envelope through :func:`parse_envelope` first.
    tiers_block = {_canonical_tier(k): v for k, v in raw_tiers_block.items()}

    for tier in requested:
        tier_data = tiers_block.get(tier)
        if not isinstance(tier_data, dict):
            summary.warnings.append(f"tier {tier!r} not present in envelope")
            summary.tiers[tier] = TierImportResult()
            continue
        entries = tier_data.get("entries") or []
        expected_dim = await asyncio.to_thread(
            _detect_embedding_dim, cache_manager._vector_store, _TIER_TO_COLLECTION[tier]
        )
        tier_result = await asyncio.to_thread(
            _apply_tier_import,
            cache_manager,
            tier,
            entries,
            mode=mode,
            re_embed=re_embed,
            expected_dim=expected_dim,
        )
        summary.tiers[tier] = tier_result

    return summary
