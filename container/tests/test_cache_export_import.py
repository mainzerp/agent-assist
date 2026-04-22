"""Tests for :mod:`app.cache.export_import` and the cache export/import API."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from app.cache.cache_manager import CacheManager
from app.cache.export_import import (
    ALLOWED_TIERS,
    EXPORT_FORMAT_TAG,
    EXPORT_PAGE_SIZE,
    SUPPORTED_FORMAT_VERSION,
    ImportSummary,
    ImportValidationError,
    TierImportResult,
    build_export_filename,
    import_envelope,
    iter_export_chunks,
    parse_envelope,
)
from app.cache.vector_store import (
    COLLECTION_RESPONSE_CACHE,
    COLLECTION_ROUTING_CACHE,
    VectorStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_routing_entry(entry_id: str = "abc123") -> dict:
    return {
        "id": entry_id,
        "document": "turn on the kitchen light",
        "embedding": [0.1, 0.2, 0.3],
        "metadata": {
            "agent_id": "light-agent",
            "confidence": "0.95",
            "hit_count": "1",
            "condensed_task": "turn on light kitchen",
            "created_at": "2026-04-20T08:00:00+00:00",
            "last_accessed": "2026-04-22T07:55:12+00:00",
            "language": "en",
        },
    }


def _make_response_entry(entry_id: str = "def456") -> dict:
    return {
        "id": entry_id,
        "document": "what is the temperature",
        "embedding": [0.4, 0.5, 0.6],
        "metadata": {
            "response_text": "It is 21 degrees.",
            "agent_id": "climate-agent",
            "confidence": "0.97",
            "hit_count": "0",
            "entity_ids": "sensor.living_room_temperature",
            "created_at": "2026-04-21T18:00:00+00:00",
            "last_accessed": "2026-04-22T09:00:00+00:00",
            "language": "en",
            "schema_version": "2",
            "cached_action": "",
        },
    }


def _make_envelope(
    routing_entries: list[dict] | None = None,
    response_entries: list[dict] | None = None,
) -> dict:
    tiers: dict = {}
    if routing_entries is not None:
        tiers["routing"] = {
            "schema_version": 1,
            "count": len(routing_entries),
            "entries": routing_entries,
        }
    if response_entries is not None:
        tiers["response"] = {
            "schema_version": 2,
            "count": len(response_entries),
            "entries": response_entries,
        }
    return {
        "export_format": EXPORT_FORMAT_TAG,
        "format_version": 1,
        "generated_at": "2026-04-22T10:15:00+00:00",
        "source": {
            "app_version": "0.20.0",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": 3,
        },
        "tiers": tiers,
    }


def _make_cache_manager(vector_store: MagicMock) -> MagicMock:
    """Return a MagicMock(spec=CacheManager) wired with the given vector store."""

    cm = MagicMock(spec=CacheManager)
    cm._vector_store = vector_store
    cm._routing_cache = MagicMock()
    cm._response_cache = MagicMock()
    cm.flush = MagicMock()
    return cm


def _vector_store_with_pages(
    pages_by_collection: dict[str, list[dict]],
) -> MagicMock:
    """Build a MagicMock(spec=VectorStore) that returns the given pages.

    ``pages_by_collection[name]`` is a list of pages; each page is a
    dict with ``ids`` / ``documents`` / ``metadatas`` / ``embeddings``
    keys mirroring Chroma. Pagination cursors only advance for full
    pagination calls (limit > 1); single-row probe calls used by
    ``_detect_embedding_dim`` are answered from page 0 without
    consuming it.
    """

    store = MagicMock(spec=VectorStore)
    counts = {name: sum(len(p.get("ids", [])) for p in pages) for name, pages in pages_by_collection.items()}
    cursors: dict[str, int] = {name: 0 for name in pages_by_collection}

    def _count(name):
        return counts.get(name, 0)

    def _get(name, **kwargs):
        pages = pages_by_collection.get(name, [])
        # Single-row probe (dim detection) does not consume the cursor.
        if kwargs.get("limit") == 1:
            if not pages or not pages[0].get("ids"):
                return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
            first = pages[0]
            return {
                "ids": first["ids"][:1],
                "documents": (first.get("documents") or [None])[:1],
                "metadatas": (first.get("metadatas") or [None])[:1],
                "embeddings": (first.get("embeddings") or [None])[:1],
            }
        idx = cursors[name]
        cursors[name] += 1
        if idx >= len(pages):
            return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        return pages[idx]

    store.count.side_effect = _count
    store.get.side_effect = _get
    return store


# ---------------------------------------------------------------------------
# 7.1 Helper unit tests
# ---------------------------------------------------------------------------


def test_build_export_filename_uses_all_tag_for_full_export():
    from datetime import datetime

    name = build_export_filename(["routing", "response"], datetime(2026, 4, 22, 10, 15, 30))
    assert name == "agent-assist-cache-all-20260422101530.json"


def test_build_export_filename_uses_single_tier_tag():
    from datetime import datetime

    name = build_export_filename(["routing"], datetime(2026, 4, 22, 10, 15, 30))
    assert name == "agent-assist-cache-routing-20260422101530.json"


def test_iter_export_chunks_emits_valid_envelope():
    routing_pages = [
        {
            "ids": ["r1"],
            "documents": ["doc1"],
            "metadatas": [{"agent_id": "a1", "language": "en"}],
            "embeddings": [[0.1, 0.2]],
        }
    ]
    response_pages = [
        {
            "ids": ["s1"],
            "documents": ["doc2"],
            "metadatas": [{"agent_id": "b1", "language": "en"}],
            "embeddings": [[0.3, 0.4]],
        }
    ]
    store = _vector_store_with_pages(
        {COLLECTION_ROUTING_CACHE: routing_pages, COLLECTION_RESPONSE_CACHE: response_pages}
    )
    cm = _make_cache_manager(store)

    chunks = list(iter_export_chunks(cm, ["routing", "response"], app_version="0.20.0"))
    payload = b"".join(chunks).decode("utf-8")
    envelope = json.loads(payload)

    assert envelope["export_format"] == EXPORT_FORMAT_TAG
    assert envelope["format_version"] == SUPPORTED_FORMAT_VERSION
    assert envelope["source"]["app_version"] == "0.20.0"
    assert set(envelope["tiers"].keys()) == {"routing", "action"}
    assert envelope["tiers"]["routing"]["count"] == 1
    assert envelope["tiers"]["routing"]["entries"][0]["id"] == "r1"
    assert envelope["tiers"]["action"]["entries"][0]["id"] == "s1"

def test_iter_export_chunks_skips_unrequested_tier():
    routing_pages = [
        {
            "ids": ["r1"],
            "documents": ["doc1"],
            "metadatas": [{"agent_id": "a1"}],
            "embeddings": [[0.1]],
        }
    ]
    store = _vector_store_with_pages({COLLECTION_ROUTING_CACHE: routing_pages})
    cm = _make_cache_manager(store)

    payload = b"".join(iter_export_chunks(cm, ["routing"], app_version="0.20.0"))
    envelope = json.loads(payload)
    assert "routing" in envelope["tiers"]
    assert "action" not in envelope["tiers"]
    assert "response" not in envelope["tiers"]


def test_iter_export_chunks_paginates():
    # Two full pages plus a short tail to verify pagination loop exit.
    page1 = {
        "ids": [f"id{i}" for i in range(EXPORT_PAGE_SIZE)],
        "documents": [f"doc{i}" for i in range(EXPORT_PAGE_SIZE)],
        "metadatas": [{"agent_id": "a"} for _ in range(EXPORT_PAGE_SIZE)],
        "embeddings": [[0.0] for _ in range(EXPORT_PAGE_SIZE)],
    }
    page2 = {
        "ids": ["tail"],
        "documents": ["tail-doc"],
        "metadatas": [{"agent_id": "a"}],
        "embeddings": [[0.0]],
    }
    store = _vector_store_with_pages({COLLECTION_ROUTING_CACHE: [page1, page2]})
    cm = _make_cache_manager(store)

    list(iter_export_chunks(cm, ["routing"], app_version="0.20.0"))
    # Two get() calls for two pages.
    get_calls = [c for c in store.get.call_args_list if c.args[0] == COLLECTION_ROUTING_CACHE]
    # One detect-dim call + two pagination calls.
    pagination_calls = [c for c in get_calls if c.kwargs.get("limit") == EXPORT_PAGE_SIZE]
    assert len(pagination_calls) == 2
    offsets = [c.kwargs["offset"] for c in pagination_calls]
    assert offsets == [0, EXPORT_PAGE_SIZE]


def test_parse_envelope_rejects_wrong_format():
    raw = json.dumps({"export_format": "other", "format_version": 1, "tiers": {"routing": {"schema_version": 1, "entries": []}}}).encode()
    with pytest.raises(ImportValidationError):
        parse_envelope(raw)


def test_parse_envelope_rejects_future_format_version():
    raw = json.dumps(
        {
            "export_format": EXPORT_FORMAT_TAG,
            "format_version": SUPPORTED_FORMAT_VERSION + 1,
            "tiers": {"routing": {"schema_version": 1, "entries": []}},
        }
    ).encode()
    with pytest.raises(ImportValidationError):
        parse_envelope(raw)


def test_parse_envelope_rejects_future_schema_version():
    raw = json.dumps(
        {
            "export_format": EXPORT_FORMAT_TAG,
            "format_version": 1,
            "tiers": {"routing": {"schema_version": 99, "entries": []}},
        }
    ).encode()
    with pytest.raises(ImportValidationError):
        parse_envelope(raw)


def test_parse_envelope_accepts_minimal_valid():
    raw = json.dumps(
        {
            "export_format": EXPORT_FORMAT_TAG,
            "format_version": 1,
            "tiers": {"routing": {"schema_version": 1, "entries": []}},
        }
    ).encode()
    envelope = parse_envelope(raw)
    assert envelope["export_format"] == EXPORT_FORMAT_TAG


def test_parse_envelope_rejects_non_json():
    with pytest.raises(ImportValidationError):
        parse_envelope(b"not-json")


def test_parse_envelope_rejects_oversized():
    from app.cache import export_import as ei

    big = b"a" * (ei.MAX_IMPORT_BYTES + 1)
    with pytest.raises(ImportValidationError):
        parse_envelope(big)


@pytest.mark.asyncio
async def test_import_envelope_merge_calls_prepare_for_flush_then_upsert():
    store = MagicMock(spec=VectorStore)
    store.get.return_value = {"ids": [], "embeddings": []}
    cm = _make_cache_manager(store)

    envelope = _make_envelope(routing_entries=[_make_routing_entry()])
    summary = await import_envelope(
        cm, envelope, mode="merge", tiers=["routing"], re_embed=False
    )

    cm._routing_cache.prepare_for_flush.assert_called_once()
    cm.flush.assert_not_called()
    store.upsert.assert_called()
    cm._routing_cache._enforce_lru.assert_called_once()
    assert summary.tiers["routing"].imported == 1


@pytest.mark.asyncio
async def test_import_envelope_replace_calls_flush_first():
    store = MagicMock(spec=VectorStore)
    store.get.return_value = {"ids": [], "embeddings": []}
    cm = _make_cache_manager(store)

    envelope = _make_envelope(routing_entries=[_make_routing_entry()])
    await import_envelope(
        cm, envelope, mode="replace", tiers=["routing"], re_embed=False
    )

    cm.flush.assert_called_once_with("routing")
    store.upsert.assert_called()


@pytest.mark.asyncio
async def test_import_envelope_re_embed_drops_embeddings():
    store = MagicMock(spec=VectorStore)
    store.get.return_value = {"ids": [], "embeddings": []}
    cm = _make_cache_manager(store)

    envelope = _make_envelope(routing_entries=[_make_routing_entry()])
    summary = await import_envelope(
        cm, envelope, mode="merge", tiers=["routing"], re_embed=True
    )

    upsert_calls = store.upsert.call_args_list
    assert any(c.kwargs.get("embeddings") is None for c in upsert_calls)
    assert summary.tiers["routing"].re_embedded == 1


@pytest.mark.asyncio
async def test_import_envelope_dim_mismatch_forces_re_embed_with_warning():
    store = MagicMock(spec=VectorStore)
    # detect_embedding_dim sees a 384-dim entry already in collection.
    store.get.return_value = {
        "ids": ["existing"],
        "embeddings": [[0.0] * 384],
    }
    cm = _make_cache_manager(store)

    entry = _make_routing_entry()
    entry["embedding"] = [0.1, 0.2, 0.3]  # only 3 dims
    envelope = _make_envelope(routing_entries=[entry])
    summary = await import_envelope(
        cm, envelope, mode="merge", tiers=["routing"], re_embed=False
    )

    assert summary.tiers["routing"].re_embedded == 1
    assert any("dim mismatch" in w for w in summary.tiers["routing"].warnings)


@pytest.mark.asyncio
async def test_import_envelope_skips_missing_agent_id():
    store = MagicMock(spec=VectorStore)
    store.get.return_value = {"ids": [], "embeddings": []}
    cm = _make_cache_manager(store)

    bad = _make_routing_entry("bad")
    bad["metadata"] = dict(bad["metadata"])
    bad["metadata"]["agent_id"] = ""
    good1 = _make_routing_entry("good1")
    good2 = _make_routing_entry("good2")
    envelope = _make_envelope(routing_entries=[bad, good1, good2])
    summary = await import_envelope(
        cm, envelope, mode="merge", tiers=["routing"], re_embed=False
    )

    assert summary.tiers["routing"].imported == 2
    assert summary.tiers["routing"].skipped == 1
    assert any("agent_id" in w for w in summary.tiers["routing"].warnings)


@pytest.mark.asyncio
async def test_import_envelope_defaults_missing_language_to_en():
    store = MagicMock(spec=VectorStore)
    store.get.return_value = {"ids": [], "embeddings": []}
    cm = _make_cache_manager(store)

    entry = _make_routing_entry()
    entry["metadata"] = dict(entry["metadata"])
    entry["metadata"].pop("language", None)
    envelope = _make_envelope(routing_entries=[entry])
    summary = await import_envelope(
        cm, envelope, mode="merge", tiers=["routing"], re_embed=False
    )

    assert summary.tiers["routing"].imported == 1
    assert any("language" in w for w in summary.tiers["routing"].warnings)
    metadatas = store.upsert.call_args_list[0].kwargs["metadatas"]
    assert metadatas[0]["language"] == "en"


@pytest.mark.asyncio
async def test_import_envelope_drops_invalid_cached_action():
    store = MagicMock(spec=VectorStore)
    store.get.return_value = {"ids": [], "embeddings": []}
    cm = _make_cache_manager(store)

    bad = _make_response_entry("bad-resp")
    bad["metadata"] = dict(bad["metadata"])
    bad["metadata"]["cached_action"] = "{not-json"
    good = _make_response_entry("good-resp")
    envelope = _make_envelope(response_entries=[bad, good])
    summary = await import_envelope(
        cm, envelope, mode="merge", tiers=["response"], re_embed=False
    )

    assert summary.tiers["action"].imported == 1
    assert summary.tiers["action"].skipped == 1
    assert any("cached_action" in w for w in summary.tiers["action"].warnings)


@pytest.mark.asyncio
async def test_import_envelope_runs_enforce_lru_once_per_tier():
    store = MagicMock(spec=VectorStore)
    store.get.return_value = {"ids": [], "embeddings": []}
    cm = _make_cache_manager(store)

    envelope = _make_envelope(
        routing_entries=[_make_routing_entry("r1")],
        response_entries=[_make_response_entry("s1")],
    )
    await import_envelope(
        cm, envelope, mode="merge", tiers=["routing", "response"], re_embed=False
    )

    cm._routing_cache._enforce_lru.assert_called_once()
    cm._response_cache._enforce_lru.assert_called_once()


# ---------------------------------------------------------------------------
# 7.2 API tests
# ---------------------------------------------------------------------------


def _build_cache_api_app(cache_manager):
    """Build a minimal FastAPI app mounting only the cache router.

    Auth is overridden to a no-op admin session.
    """

    from fastapi import FastAPI

    from app.api.routes.cache_api import router as cache_router
    from app.security.auth import require_admin_session

    app = FastAPI()
    app.dependency_overrides[require_admin_session] = lambda: {"username": "admin"}
    app.state.cache_manager = cache_manager
    # ensure_setup_runtime_initialized expects this attribute.
    app.state.setup_runtime_initialized = True
    app.include_router(cache_router)
    return app


def _make_export_cache_manager() -> MagicMock:
    routing_pages = [
        {
            "ids": ["r1"],
            "documents": ["doc1"],
            "metadatas": [{"agent_id": "a1", "language": "en"}],
            "embeddings": [[0.1, 0.2]],
        }
    ]
    store = _vector_store_with_pages({COLLECTION_ROUTING_CACHE: routing_pages})
    return _make_cache_manager(store)


def _make_export_cache_manager_with_action() -> MagicMock:
    routing_pages = [
        {
            "ids": ["r1"],
            "documents": ["doc1"],
            "metadatas": [{"agent_id": "a1", "language": "en"}],
            "embeddings": [[0.1, 0.2]],
        }
    ]
    action_pages = [
        {
            "ids": ["s1"],
            "documents": ["doc2"],
            "metadatas": [{"agent_id": "b1", "language": "en"}],
            "embeddings": [[0.3, 0.4]],
        }
    ]
    store = _vector_store_with_pages(
        {
            COLLECTION_ROUTING_CACHE: routing_pages,
            COLLECTION_RESPONSE_CACHE: action_pages,
        }
    )
    return _make_cache_manager(store)


def test_export_endpoint_returns_attachment_headers():
    from fastapi.testclient import TestClient

    cm = _make_export_cache_manager()
    app = _build_cache_api_app(cm)
    client = TestClient(app)

    resp = client.get("/api/admin/cache/export?tier=routing")
    assert resp.status_code == 200
    cd = resp.headers.get("content-disposition", "")
    assert cd.startswith('attachment; filename="agent-assist-cache-')
    payload = json.loads(resp.content.decode("utf-8"))
    assert payload["export_format"] == EXPORT_FORMAT_TAG
    assert "routing" in payload["tiers"]


def test_export_rejects_invalid_tier():
    from fastapi.testclient import TestClient

    cm = _make_export_cache_manager()
    app = _build_cache_api_app(cm)
    client = TestClient(app)

    resp = client.get("/api/admin/cache/export?tier=foo")
    assert resp.status_code == 422


def test_export_503_when_cache_manager_missing():
    from fastapi.testclient import TestClient

    app = _build_cache_api_app(None)
    client = TestClient(app)

    resp = client.get("/api/admin/cache/export")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "error"


def test_import_endpoint_happy_path():
    from fastapi.testclient import TestClient

    store = MagicMock(spec=VectorStore)
    store.get.return_value = {"ids": [], "embeddings": []}
    cm = _make_cache_manager(store)
    app = _build_cache_api_app(cm)
    client = TestClient(app)

    envelope = _make_envelope(routing_entries=[_make_routing_entry()])
    raw = json.dumps(envelope).encode("utf-8")
    resp = client.post(
        "/api/admin/cache/import",
        files={"file": ("envelope.json", io.BytesIO(raw), "application/json")},
        data={"mode": "merge", "tiers": "routing", "re_embed": "false"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert body["mode"] == "merge"
    assert body["tiers"]["routing"]["imported"] == 1


def test_import_rejects_invalid_mode():
    from fastapi.testclient import TestClient

    store = MagicMock(spec=VectorStore)
    cm = _make_cache_manager(store)
    app = _build_cache_api_app(cm)
    client = TestClient(app)

    envelope = _make_envelope(routing_entries=[_make_routing_entry()])
    raw = json.dumps(envelope).encode("utf-8")
    resp = client.post(
        "/api/admin/cache/import",
        files={"file": ("envelope.json", io.BytesIO(raw), "application/json")},
        data={"mode": "bogus", "tiers": "routing"},
    )
    assert resp.status_code == 400


def test_import_rejects_oversized_payload(monkeypatch):
    from fastapi.testclient import TestClient

    from app.cache import export_import as ei

    store = MagicMock(spec=VectorStore)
    cm = _make_cache_manager(store)
    app = _build_cache_api_app(cm)
    client = TestClient(app)

    from app.api.routes import cache_api as cache_api_mod

    monkeypatch.setattr(ei, "MAX_IMPORT_BYTES", 8)
    monkeypatch.setattr(cache_api_mod, "MAX_IMPORT_BYTES", 8)
    raw = b"x" * 32
    resp = client.post(
        "/api/admin/cache/import",
        files={"file": ("envelope.json", io.BytesIO(raw), "application/json")},
        data={"mode": "merge", "tiers": "routing"},
    )
    assert resp.status_code == 413


def test_import_rejects_bad_envelope():
    from fastapi.testclient import TestClient

    store = MagicMock(spec=VectorStore)
    cm = _make_cache_manager(store)
    app = _build_cache_api_app(cm)
    client = TestClient(app)

    raw = json.dumps({"foo": "bar"}).encode("utf-8")
    resp = client.post(
        "/api/admin/cache/import",
        files={"file": ("envelope.json", io.BytesIO(raw), "application/json")},
        data={"mode": "merge", "tiers": "routing"},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["status"] == "error"
    assert "export_format" in body["detail"]


def test_import_replace_calls_flush():
    from fastapi.testclient import TestClient

    store = MagicMock(spec=VectorStore)
    store.get.return_value = {"ids": [], "embeddings": []}
    cm = _make_cache_manager(store)
    app = _build_cache_api_app(cm)
    client = TestClient(app)

    envelope = _make_envelope(routing_entries=[_make_routing_entry()])
    raw = json.dumps(envelope).encode("utf-8")
    resp = client.post(
        "/api/admin/cache/import",
        files={"file": ("envelope.json", io.BytesIO(raw), "application/json")},
        data={"mode": "replace", "tiers": "routing"},
    )
    assert resp.status_code == 200, resp.text
    cm.flush.assert_called_once_with("routing")


def test_import_passes_re_embed_flag():
    from fastapi.testclient import TestClient

    store = MagicMock(spec=VectorStore)
    cm = _make_cache_manager(store)
    app = _build_cache_api_app(cm)
    client = TestClient(app)

    envelope = _make_envelope(routing_entries=[_make_routing_entry()])
    raw = json.dumps(envelope).encode("utf-8")

    async def _fake_import_envelope(*args, **kwargs):
        return ImportSummary(
            mode=kwargs["mode"],
            format_version=1,
            tiers={"routing": TierImportResult(imported=1)},
        )

    with patch(
        "app.api.routes.cache_api.import_envelope",
        side_effect=_fake_import_envelope,
    ) as patched:
        resp = client.post(
            "/api/admin/cache/import",
            files={"file": ("envelope.json", io.BytesIO(raw), "application/json")},
            data={"mode": "merge", "tiers": "routing", "re_embed": "true"},
        )
    assert resp.status_code == 200, resp.text
    patched.assert_called_once()
    assert patched.call_args.kwargs["re_embed"] is True


# ---------------------------------------------------------------------------
# 0.21.0 rename alias contract (action / response)
# ---------------------------------------------------------------------------


def test_export_envelope_emits_format_version_2_and_action_tier():
    cm = _make_export_cache_manager_with_action()
    payload = b''.join(iter_export_chunks(cm, ["routing", "response"], app_version="0.21.0"))
    envelope = json.loads(payload)
    assert envelope["format_version"] == 2
    assert SUPPORTED_FORMAT_VERSION == 2
    assert "action" in envelope["tiers"]
    assert "response" not in envelope["tiers"]


@pytest.mark.asyncio
async def test_parse_envelope_v1_response_alias_round_trips_to_action():
    store = MagicMock(spec=VectorStore)
    store.get.return_value = {"ids": [], "embeddings": []}
    cm = _make_cache_manager(store)
    envelope = _make_envelope(response_entries=[_make_response_entry("alias-1")])
    assert envelope["format_version"] == 1
    assert "response" in envelope["tiers"]
    raw = json.dumps(envelope).encode("utf-8")
    parsed = parse_envelope(raw)
    assert "action" in parsed["tiers"]
    assert "response" not in parsed["tiers"]
    summary = await import_envelope(
        cm, parsed, mode="merge", tiers=["action"], re_embed=False
    )
    assert summary.tiers["action"].imported == 1
    cm._response_cache.prepare_for_flush.assert_called()


def test_api_export_accepts_legacy_tier_response():
    from fastapi.testclient import TestClient

    cm = _make_export_cache_manager_with_action()
    app = _build_cache_api_app(cm)
    client = TestClient(app)
    resp_legacy = client.get("/api/admin/cache/export?tier=response")
    assert resp_legacy.status_code == 200
    resp_canonical = client.get("/api/admin/cache/export?tier=action")
    assert resp_canonical.status_code == 200


def test_api_import_accepts_legacy_tiers_field_response():
    from fastapi.testclient import TestClient

    store = MagicMock(spec=VectorStore)
    store.get.return_value = {"ids": [], "embeddings": []}
    cm = _make_cache_manager(store)
    app = _build_cache_api_app(cm)
    client = TestClient(app)

    envelope = _make_envelope(response_entries=[_make_response_entry()])
    raw = json.dumps(envelope).encode("utf-8")
    resp = client.post(
        "/api/admin/cache/import",
        files={"file": ("envelope.json", io.BytesIO(raw), "application/json")},
        data={"mode": "merge", "tiers": "routing,response", "re_embed": "false"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "action" in body["tiers"]
    assert body["tiers"]["action"]["imported"] == 1


def test_api_flush_accepts_legacy_tier_response():
    from fastapi.testclient import TestClient

    store = MagicMock(spec=VectorStore)
    cm = _make_cache_manager(store)
    cm.flush = MagicMock()
    app = _build_cache_api_app(cm)
    client = TestClient(app)
    resp = client.post("/api/admin/cache/flush", json={"tier": "response"})
    assert resp.status_code == 200, resp.text
    cm.flush.assert_called_with("action")
