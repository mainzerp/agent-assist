"""Tests for cancel-interaction routing (orchestrator classification + canned speech)."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock litellm before importing app modules; force-load llm client so
# ``@patch("app.llm.client.complete")`` resolves (matches test_agents.py).
_litellm_mock = MagicMock()
_litellm_mock.exceptions.AuthenticationError = type("AuthenticationError", (Exception,), {})
sys.modules.setdefault("litellm", _litellm_mock)

import app.llm.client  # noqa: E402,F401
from app.agents.cancel_speech import cancel_interaction_ack  # noqa: E402
from app.agents.orchestrator import OrchestratorAgent  # noqa: E402
from app.models.agent import AgentCard, AgentTask, TaskContext  # noqa: E402
from tests.helpers import make_agent_task  # noqa: E402

CANCEL_AGENT = "cancel-interaction"


def _make_orch():
    dispatcher = AsyncMock()
    response_mock = MagicMock()
    response_mock.error = None
    response_mock.result = {"speech": "unexpected"}
    dispatcher.dispatch = AsyncMock(return_value=response_mock)
    dispatcher.dispatch_stream = AsyncMock()

    registry = AsyncMock()
    registry.list_agents = AsyncMock(
        return_value=[
            AgentCard(agent_id="light-agent", name="Light Agent", description="", skills=["light"]),
            AgentCard(agent_id="general-agent", name="General Agent", description="", skills=["general"]),
        ]
    )

    cache_manager = MagicMock()
    cache_manager.process = AsyncMock(return_value=MagicMock(hit_type="miss", agent_id=None, similarity=0.5))
    cache_manager.apply_rewrite = AsyncMock()

    async def _store_routing_async(*args, **kwargs):
        return cache_manager.store_routing(*args, **kwargs)

    async def _store_response_async(entry):
        return cache_manager.store_response(entry)

    cache_manager.store_routing_async = _store_routing_async
    cache_manager.store_response_async = _store_response_async

    return OrchestratorAgent(dispatcher=dispatcher, registry=registry, cache_manager=cache_manager), dispatcher


class TestCancelInteractionAck:
    def test_english(self):
        assert cancel_interaction_ack("en") == "Okay."
        assert cancel_interaction_ack(None) == "Okay."

    def test_german(self):
        assert cancel_interaction_ack("de") == "Alles klar."
        assert cancel_interaction_ack("de-DE") == "Alles klar."


class TestOrchestratorCancelInteraction:
    @pytest.fixture(autouse=True)
    def _mock_conversation_repo(self):
        with patch("app.agents.orchestrator.ConversationRepository") as mock_repo:
            mock_repo.insert = AsyncMock(return_value=1)
            yield mock_repo

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_handle_task_cancel_does_not_dispatch(self, mock_complete, mock_track, mock_settings):
        mock_settings.get_value = AsyncMock(side_effect=lambda k, d=None: "auto" if k == "language" else d)
        orch, dispatcher = _make_orch()
        mock_complete.return_value = f"{CANCEL_AGENT} (98%): dismiss interaction"

        task = make_agent_task(description="nevermind", user_text="nevermind", context=TaskContext(language="en"))
        task.conversation_id = "c1"
        result = await orch.handle_task(task)

        assert result["speech"] == "Okay."
        assert result["routed_to"] == CANCEL_AGENT
        dispatcher.dispatch.assert_not_awaited()
        dispatcher.dispatch_stream.assert_not_awaited()

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_handle_task_cancel_german_ack(self, mock_complete, mock_track, mock_settings):
        mock_settings.get_value = AsyncMock(side_effect=lambda k, d=None: "de" if k == "language" else d)
        orch, dispatcher = _make_orch()
        mock_complete.return_value = f"{CANCEL_AGENT} (95%): dismiss"

        task = make_agent_task(description="Abbrechen", user_text="Abbrechen", context=TaskContext(language="de"))
        task.conversation_id = "c2"
        result = await orch.handle_task(task)

        assert result["speech"] == "Alles klar."
        dispatcher.dispatch.assert_not_awaited()

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_handle_task_stream_early_return_cancel(self, mock_complete, mock_track, mock_settings):
        mock_settings.get_value = AsyncMock(side_effect=lambda k, d=None: "auto" if k == "language" else d)
        orch, dispatcher = _make_orch()
        mock_complete.return_value = f"{CANCEL_AGENT} (95%): dismiss interaction"

        task = AgentTask(
            description="stop",
            user_text="stop",
            conversation_id="c3",
            context=TaskContext(language="en"),
        )

        chunks = []
        async for ch in orch.handle_task_stream(task):
            chunks.append(ch)

        assert len(chunks) == 1
        assert chunks[0]["done"] is True
        assert chunks[0]["mediated_speech"] == "Okay."
        dispatcher.dispatch.assert_not_awaited()
        dispatcher.dispatch_stream.assert_not_awaited()
