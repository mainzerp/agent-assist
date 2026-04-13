"""Integration tests for the setup wizard flow.

Tests setup routes including step progression, HA/LLM connection testing,
and completion behavior.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import sys

import httpx
import pytest
import pytest_asyncio

from app.security.auth import (
    require_admin_session,
    require_admin_session_redirect,
    require_api_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_setup_app(*, setup_complete: bool = False):
    """Build a FastAPI app for setup wizard tests.

    By default setup is *not* complete so the middleware allows /setup/ routes
    but redirects everything else.
    """
    from app.main import create_app

    app = create_app()

    @asynccontextmanager
    async def _noop_lifespan(a):
        yield

    app.router.lifespan_context = _noop_lifespan

    # Set state directly
    app.state.startup_time = 0
    app.state.registry = MagicMock()
    app.state.dispatcher = MagicMock()
    app.state.ha_client = MagicMock()
    app.state.entity_index = None
    app.state.cache_manager = None
    app.state.entity_matcher = None
    app.state.alias_resolver = None
    app.state.custom_loader = None
    app.state.mcp_registry = MagicMock()
    app.state.mcp_registry.list_servers.return_value = []
    app.state.mcp_tool_manager = MagicMock()
    app.state.ws_client = None
    app.state.presence_detector = None
    app.state.plugin_loader = MagicMock()
    app.state.plugin_loader.loaded_plugins = {}
    return app


@pytest_asyncio.fixture()
async def setup_client(db_repository):
    """Client where setup is NOT complete (default seed state)."""
    app = _build_setup_app(setup_complete=False)
    with patch(
        "app.db.repository.SetupStateRepository.is_complete",
        new_callable=AsyncMock,
        return_value=False,
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", follow_redirects=False,
        ) as client:
            yield client


@pytest_asyncio.fixture()
async def setup_complete_client(db_repository):
    """Client where setup IS complete."""
    app = _build_setup_app(setup_complete=True)
    # Override admin session so dashboard routes are accessible
    app.dependency_overrides[require_admin_session] = lambda: {"username": "admin"}
    app.dependency_overrides[require_admin_session_redirect] = lambda: {"username": "admin"}
    with patch(
        "app.db.repository.SetupStateRepository.is_complete",
        new_callable=AsyncMock,
        return_value=True,
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", follow_redirects=False,
        ) as client:
            yield client


# ===================================================================
# Setup-incomplete state
# ===================================================================


@pytest.mark.integration
class TestSetupIncompleteRedirect:

    async def test_non_setup_routes_redirect_to_setup(self, setup_client: httpx.AsyncClient):
        resp = await setup_client.get("/dashboard/")
        assert resp.status_code == 302
        assert "/setup/" in resp.headers.get("location", "")

    async def test_api_routes_redirect_when_setup_incomplete(
        self, setup_client: httpx.AsyncClient
    ):
        resp = await setup_client.post(
            "/api/conversation", json={"text": "hello"}
        )
        assert resp.status_code == 302

    async def test_health_accessible_during_setup(self, setup_client: httpx.AsyncClient):
        resp = await setup_client.get("/api/health")
        assert resp.status_code == 200

    async def test_setup_routes_accessible_during_setup(self, setup_client: httpx.AsyncClient):
        resp = await setup_client.get("/setup/")
        # Should redirect to first incomplete step, not 302 to /setup/ again
        assert resp.status_code in (200, 302)
        if resp.status_code == 302:
            assert "/setup/step/" in resp.headers.get("location", "")


# ===================================================================
# Setup already complete
# ===================================================================


@pytest.mark.integration
class TestSetupAlreadyComplete:

    async def test_setup_index_redirects_to_dashboard_when_complete(
        self, setup_complete_client: httpx.AsyncClient
    ):
        with patch(
            "app.setup.routes.SetupStateRepository.get_all_steps",
            new_callable=AsyncMock,
            return_value=[
                {"step": "admin_password", "completed": True},
                {"step": "ha_connection", "completed": True},
                {"step": "container_api_key", "completed": True},
                {"step": "llm_providers", "completed": True},
                {"step": "review_complete", "completed": True},
            ],
        ):
            resp = await setup_complete_client.get("/setup/")
            assert resp.status_code == 302
            assert "/dashboard/" in resp.headers.get("location", "")


# ===================================================================
# Step submissions
# ===================================================================


@pytest.mark.integration
class TestSetupStepSubmissions:

    async def test_step1_admin_password(self, setup_client: httpx.AsyncClient):
        with patch(
            "app.setup.routes.AdminAccountRepository.create",
            new_callable=AsyncMock,
        ) as mock_create, patch(
            "app.setup.routes.SetupStateRepository.set_step_completed",
            new_callable=AsyncMock,
        ):
            resp = await setup_client.post(
                "/setup/step/1",
                data={"username": "admin", "password": "test-password-123"},
            )
            assert resp.status_code == 303
            assert "/setup/step/2" in resp.headers.get("location", "")
            mock_create.assert_awaited_once()

    async def test_step2_ha_connection(self, setup_client: httpx.AsyncClient):
        with patch(
            "app.setup.routes.SettingsRepository.set",
            new_callable=AsyncMock,
        ), patch(
            "app.ha_client.auth.set_ha_token",
            new_callable=AsyncMock,
        ), patch(
            "app.setup.routes.SetupStateRepository.set_step_completed",
            new_callable=AsyncMock,
        ):
            resp = await setup_client.post(
                "/setup/step/2",
                data={"ha_url": "http://homeassistant.local:8123", "ha_token": "test-token"},
            )
            assert resp.status_code == 303
            assert "/setup/step/3" in resp.headers.get("location", "")

    async def test_step3_api_key_generation(self, setup_client: httpx.AsyncClient):
        with patch(
            "app.setup.routes.store_secret",
            new_callable=AsyncMock,
        ), patch(
            "app.setup.routes.SetupStateRepository.set_step_completed",
            new_callable=AsyncMock,
        ), patch(
            "app.setup.routes.SetupStateRepository.get_all_steps",
            new_callable=AsyncMock,
            return_value=[
                {"step": "admin_password", "completed": True},
                {"step": "ha_connection", "completed": True},
                {"step": "container_api_key", "completed": True},
                {"step": "llm_providers", "completed": False},
                {"step": "review_complete", "completed": False},
            ],
        ):
            resp = await setup_client.post("/setup/step/3")
            assert resp.status_code == 200
            assert "text/html" in resp.headers.get("content-type", "")

    async def test_step4_llm_keys(self, setup_client: httpx.AsyncClient):
        with patch(
            "app.setup.routes.store_secret",
            new_callable=AsyncMock,
        ), patch(
            "app.setup.routes.SetupStateRepository.set_step_completed",
            new_callable=AsyncMock,
        ):
            resp = await setup_client.post(
                "/setup/step/4",
                data={"openrouter_key": "sk-test-123", "groq_key": "", "ollama_url": ""},
            )
            assert resp.status_code == 303
            assert "/setup/step/5" in resp.headers.get("location", "")

    async def test_step5_complete(self, setup_client: httpx.AsyncClient):
        with patch(
            "app.setup.routes.SetupStateRepository.set_step_completed",
            new_callable=AsyncMock,
        ):
            resp = await setup_client.post("/setup/step/5")
            assert resp.status_code == 303
            assert "/dashboard/" in resp.headers.get("location", "")

    async def test_step5_review_excludes_review_complete(self, setup_client: httpx.AsyncClient):
        """Step 5 review page should not show the 'review_complete' meta-step."""
        with patch(
            "app.setup.routes.SetupStateRepository.get_all_steps",
            new_callable=AsyncMock,
            return_value=[
                {"step": "admin_password", "completed": True},
                {"step": "ha_connection", "completed": True},
                {"step": "container_api_key", "completed": True},
                {"step": "llm_providers", "completed": True},
                {"step": "review_complete", "completed": False},
            ],
        ):
            resp = await setup_client.get("/setup/step/5")
            assert resp.status_code == 200
            body = resp.text
            assert "Admin Password" in body
            assert "Ha Connection" in body
            assert "Container Api Key" in body
            assert "Llm Providers" in body
            assert "Review Complete" not in body


# ===================================================================
# HA connection test endpoint
# ===================================================================


@pytest.mark.integration
class TestHAConnectionTest:

    async def test_ha_connection_test_success(self, setup_client: httpx.AsyncClient):
        with patch(
            "app.setup.routes.test_ha_connection",
            new_callable=AsyncMock,
            return_value=True,
        ):
            resp = await setup_client.post(
                "/setup/test/ha",
                data={"ha_url": "http://ha.local:8123", "ha_token": "valid-token"},
            )
            assert resp.status_code == 200
            assert "test-success" in resp.text
            assert "Connected to Home Assistant" in resp.text

    async def test_ha_connection_test_failure(self, setup_client: httpx.AsyncClient):
        with patch(
            "app.setup.routes.test_ha_connection",
            new_callable=AsyncMock,
            return_value=False,
        ):
            resp = await setup_client.post(
                "/setup/test/ha",
                data={"ha_url": "http://bad-url:8123", "ha_token": "bad-token"},
            )
            assert resp.status_code == 200
            assert "test-error" in resp.text
            assert "Failed to connect" in resp.text


# ===================================================================
# LLM test endpoint
# ===================================================================


@pytest.mark.integration
class TestLLMTest:

    async def test_llm_test_success(self, setup_client: httpx.AsyncClient):
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello!"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        mock_litellm_mod = MagicMock()
        mock_litellm_mod.acompletion = AsyncMock(return_value=mock_resp)
        with patch.dict(sys.modules, {"litellm": mock_litellm_mod}):
            resp = await setup_client.post(
                "/setup/test/llm",
                data={"provider": "openrouter", "api_key": "sk-test"},
            )
            assert resp.status_code == 200
            assert "test-success" in resp.text
            assert "Connected to openrouter" in resp.text

    async def test_llm_test_failure(self, setup_client: httpx.AsyncClient):
        mock_litellm_mod = MagicMock()
        mock_litellm_mod.acompletion = AsyncMock(side_effect=Exception("API error"))
        with patch.dict(sys.modules, {"litellm": mock_litellm_mod}):
            resp = await setup_client.post(
                "/setup/test/llm",
                data={"provider": "openrouter", "api_key": "bad-key"},
            )
            assert resp.status_code == 200
            assert "test-error" in resp.text
            assert "Error:" in resp.text

    async def test_llm_test_unknown_provider(self, setup_client: httpx.AsyncClient):
        mock_litellm_mod = MagicMock()
        with patch.dict(sys.modules, {"litellm": mock_litellm_mod}):
            resp = await setup_client.post(
                "/setup/test/llm",
                data={"provider": "unknown-provider", "api_key": "key"},
            )
            assert resp.status_code == 200
            assert "test-error" in resp.text
            assert "Unknown provider" in resp.text


# ===================================================================
# Phase 4.1: Additional setup wizard tests
# ===================================================================


@pytest.mark.integration
class TestSetupDuplicateAdmin:
    """Test that submitting step 1 twice (duplicate admin) does not crash (fix 1.4)."""

    async def test_duplicate_admin_submission_succeeds(self, setup_client: httpx.AsyncClient):
        """Submitting admin credentials a second time should use INSERT OR REPLACE."""
        with patch(
            "app.setup.routes.AdminAccountRepository.create",
            new_callable=AsyncMock,
        ) as mock_create, patch(
            "app.setup.routes.SetupStateRepository.set_step_completed",
            new_callable=AsyncMock,
        ):
            # First submission
            resp1 = await setup_client.post(
                "/setup/step/1",
                data={"username": "admin", "password": "first-password"},
            )
            assert resp1.status_code == 303
            # Second submission (same username, different password)
            resp2 = await setup_client.post(
                "/setup/step/1",
                data={"username": "admin", "password": "second-password"},
            )
            assert resp2.status_code == 303
            assert mock_create.await_count == 2


@pytest.mark.integration
class TestSetupXSSPrevention:
    """Test XSS prevention in the LLM test endpoint (fix 1.3)."""

    async def test_llm_test_provider_xss_escaped(self, setup_client: httpx.AsyncClient):
        """Provider name containing script tags should be HTML-escaped in the response."""
        mock_litellm_mod = MagicMock()
        mock_litellm_mod.acompletion = AsyncMock(side_effect=Exception("fail"))
        with patch.dict(sys.modules, {"litellm": mock_litellm_mod}):
            resp = await setup_client.post(
                "/setup/test/llm",
                data={
                    "provider": '<script>alert("xss")</script>',
                    "api_key": "test-key",
                },
            )
            assert resp.status_code == 200
            body = resp.text
            # Raw script tag must NOT appear in the response
            assert "<script>" not in body
            # Escaped version should be present
            assert "&lt;script&gt;" in body or "Unknown provider" in body
