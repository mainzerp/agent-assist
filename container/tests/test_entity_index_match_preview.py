"""Tests for /api/admin/entity-index/match-preview.

Exercises the match preview endpoint with mocked ``entity_index`` /
``entity_matcher`` so we validate the response shape and the three
blocks (deterministic resolver, hybrid candidates, visibility summary)
without spinning up the full app lifespan.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import httpx
from fastapi import FastAPI

from app.api.routes import entity_index_api
from app.security.auth import require_admin_session
from app.entity.matcher import MatchResult


def _make_entry(entity_id: str, friendly_name: str, area: str | None = None):
    return SimpleNamespace(
        entity_id=entity_id,
        friendly_name=friendly_name,
        area=area,
        domain=entity_id.split(".", 1)[0],
    )


@pytest_asyncio.fixture()
async def preview_client():
    """Minimal FastAPI app wired with the entity_index router and mocks."""
    app = FastAPI()
    app.dependency_overrides[require_admin_session] = lambda: {"user": "test"}
    app.include_router(entity_index_api.router)

    entries = [
        _make_entry("light.keller", "Keller", area="keller"),
        _make_entry("light.bedroom", "Bedroom", area="bedroom"),
    ]

    entity_index = MagicMock()
    entity_index.list_entries = MagicMock(return_value=entries)
    entity_index.get_by_id = MagicMock(
        side_effect=lambda eid: next((e for e in entries if e.entity_id == eid), None)
    )

    entity_matcher = MagicMock()
    entity_matcher.match = AsyncMock(return_value=[
        MatchResult(
            entity_id="light.keller",
            friendly_name="Keller",
            score=0.92,
            signal_scores={"alias": 1.0, "embedding": 0.81, "levenshtein": 0.9},
        ),
        MatchResult(
            entity_id="light.bedroom",
            friendly_name="Bedroom",
            score=0.72,
            signal_scores={"embedding": 0.72},
        ),
    ])
    entity_matcher.filter_visible_results = AsyncMock(side_effect=lambda a, r: r)

    app.state.entity_index = entity_index
    app.state.entity_matcher = entity_matcher

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, entity_index, entity_matcher


@pytest.mark.asyncio
async def test_match_preview_returns_all_blocks(preview_client):
    client, _ei, _em = preview_client

    resolver_output = {
        "entity_id": "light.keller",
        "friendly_name": "Keller",
        "speech": None,
        "metadata": {
            "query": "keller",
            "resolution_path": "exact_friendly_name",
            "match_count": 1,
            "top_entity_id": "light.keller",
            "top_score": 0.92,
        },
    }

    with patch(
        "app.agents.action_executor._resolve_light_entity",
        new=AsyncMock(return_value=resolver_output),
    ), patch(
        "app.db.repository.EntityVisibilityRepository.get_rules",
        new=AsyncMock(return_value=[]),
    ):
        resp = await client.get(
            "/api/admin/entity-index/match-preview",
            params={"q": "keller", "agent_id": "light-agent"},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["query"] == "keller"
    assert data["agent_id"] == "light-agent"

    det = data["deterministic"]
    assert det["entity_id"] == "light.keller"
    assert det["friendly_name"] == "Keller"
    assert det["metadata"]["resolution_path"] == "exact_friendly_name"
    assert det["domain_allowed"] is True
    assert det["error"] is None

    assert data["hybrid_error"] is None
    assert len(data["hybrid"]) == 2
    top = data["hybrid"][0]
    assert top["entity_id"] == "light.keller"
    assert top["domain"] == "light"
    assert top["area"] == "keller"
    assert top["score"] == pytest.approx(0.92, abs=1e-3)
    assert "alias" in top["signal_scores"]

    vis = data["visibility"]
    assert vis["agent_id"] == "light-agent"
    assert vis["rules"] == []
    assert vis["total_entity_count"] == 2
    assert vis["visible_entity_count"] == 2


@pytest.mark.asyncio
async def test_match_preview_rejects_empty_query(preview_client):
    client, *_ = preview_client
    resp = await client.get(
        "/api/admin/entity-index/match-preview",
        params={"q": "   "},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_match_preview_surfaces_domain_gate_reject(preview_client):
    client, *_ = preview_client
    resolver_output = {
        "entity_id": "climate.thermostat",
        "friendly_name": "Thermostat",
        "speech": None,
        "metadata": {
            "query": "thermostat",
            "resolution_path": "exact_friendly_name",
            "match_count": 1,
            "top_entity_id": "climate.thermostat",
        },
    }
    with patch(
        "app.agents.action_executor._resolve_light_entity",
        new=AsyncMock(return_value=resolver_output),
    ), patch(
        "app.db.repository.EntityVisibilityRepository.get_rules",
        new=AsyncMock(return_value=[]),
    ):
        resp = await client.get(
            "/api/admin/entity-index/match-preview",
            params={"q": "thermostat"},
        )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["deterministic"]["entity_id"] == "climate.thermostat"
    assert data["deterministic"]["domain_allowed"] is False


@pytest.mark.asyncio
async def test_match_preview_503_when_index_missing():
    app = FastAPI()
    app.dependency_overrides[require_admin_session] = lambda: {"user": "test"}
    app.include_router(entity_index_api.router)
    app.state.entity_index = None
    app.state.entity_matcher = None

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/api/admin/entity-index/match-preview",
            params={"q": "keller"},
        )
    assert resp.status_code == 503
