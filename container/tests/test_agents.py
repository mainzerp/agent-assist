"""Tests for app.agents -- all specialized agents, orchestrator, rewrite, and custom loader."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock litellm before importing any app modules that depend on it
_litellm_mock = MagicMock()


class _AuthenticationError(Exception):
    pass


_litellm_mock.exceptions.AuthenticationError = _AuthenticationError
sys.modules.setdefault("litellm", _litellm_mock)

from app.agents.base import BaseAgent  # noqa: E402
from app.agents.light import LightAgent  # noqa: E402
from app.agents.music import MusicAgent  # noqa: E402
from app.agents.climate import ClimateAgent  # noqa: E402
from app.agents.media import MediaAgent  # noqa: E402
from app.agents.timer import TimerAgent  # noqa: E402
from app.agents.scene import SceneAgent  # noqa: E402
from app.agents.automation import AutomationAgent  # noqa: E402
from app.agents.security import SecurityAgent  # noqa: E402
from app.agents.general import GeneralAgent  # noqa: E402
from app.agents.rewrite import RewriteAgent  # noqa: E402
from app.agents.orchestrator import OrchestratorAgent  # noqa: E402
from app.agents.custom_loader import DynamicAgent, CustomAgentLoader  # noqa: E402
from app.models.agent import AgentCard, AgentTask, TaskContext  # noqa: E402
import app.llm.client  # noqa: E402,F401 -- force module load for patch targets

from tests.helpers import make_agent_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(description: str = "turn on kitchen light", user_text: str | None = None, context: TaskContext | None = None) -> AgentTask:
    return make_agent_task(
        description=description,
        user_text=user_text or description,
        context=context,
    )


# ---------------------------------------------------------------------------
# BaseAgent abstract contract
# ---------------------------------------------------------------------------

class TestBaseAgent:
    """Tests for the BaseAgent abstract base class."""

    def test_base_agent_is_abstract(self):
        with pytest.raises(TypeError):
            BaseAgent()  # type: ignore[abstract]

    def test_base_agent_stores_ha_client_and_entity_index(self):
        ha = MagicMock()
        ei = MagicMock()
        agent = LightAgent(ha_client=ha, entity_index=ei)
        assert agent._ha_client is ha
        assert agent._entity_index is ei

    def test_base_agent_defaults_to_none_dependencies(self):
        agent = LightAgent()
        assert agent._ha_client is None
        assert agent._entity_index is None


# ---------------------------------------------------------------------------
# Agent card validation (all agents)
# ---------------------------------------------------------------------------

class TestAgentCards:
    """Each agent must expose a valid AgentCard."""

    @pytest.mark.parametrize("agent_cls,expected_id", [
        (LightAgent, "light-agent"),
        (MusicAgent, "music-agent"),
        (ClimateAgent, "climate-agent"),
        (MediaAgent, "media-agent"),
        (TimerAgent, "timer-agent"),
        (SceneAgent, "scene-agent"),
        (AutomationAgent, "automation-agent"),
        (SecurityAgent, "security-agent"),
        (GeneralAgent, "general-agent"),
        (RewriteAgent, "rewrite-agent"),
    ])
    def test_agent_card_has_correct_id(self, agent_cls, expected_id):
        agent = agent_cls()
        card = agent.agent_card
        assert isinstance(card, AgentCard)
        assert card.agent_id == expected_id

    @pytest.mark.parametrize("agent_cls", [
        LightAgent, MusicAgent, ClimateAgent, MediaAgent,
        TimerAgent, SceneAgent, AutomationAgent, SecurityAgent,
        GeneralAgent, RewriteAgent,
    ])
    def test_agent_card_has_skills(self, agent_cls):
        agent = agent_cls()
        card = agent.agent_card
        assert len(card.skills) > 0

    @pytest.mark.parametrize("agent_cls", [
        LightAgent, MusicAgent, ClimateAgent, MediaAgent,
        TimerAgent, SceneAgent, AutomationAgent, SecurityAgent,
        GeneralAgent, RewriteAgent,
    ])
    def test_agent_card_has_endpoint(self, agent_cls):
        agent = agent_cls()
        card = agent.agent_card
        assert card.endpoint.startswith("local://")


# ---------------------------------------------------------------------------
# Specialized agents: handle_task via mocked LLM
# ---------------------------------------------------------------------------

class TestLightAgent:

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Turned on the kitchen light.")
    async def test_handle_task_returns_speech(self, mock_complete):
        agent = LightAgent(ha_client=MagicMock(), entity_index=MagicMock())
        result = await agent.handle_task(_make_task("Turn on the kitchen light"))
        assert result["speech"] == "Turned on the kitchen light."
        mock_complete.assert_awaited_once()

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Brightness set to 50%.")
    async def test_handle_task_with_conversation_context(self, mock_complete):
        ctx = TaskContext(conversation_turns=[{"role": "user", "content": "hi"}])
        task = _make_task("Set bedroom light brightness to 50%", context=ctx)
        agent = LightAgent()
        result = await agent.handle_task(task)
        assert "Brightness" in result["speech"] or "50" in result["speech"]
        # Should have system + conversation turn + user message
        call_messages = mock_complete.call_args[0][1]
        assert len(call_messages) >= 3

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Color changed.")
    async def test_handle_task_includes_system_prompt(self, mock_complete):
        agent = LightAgent()
        await agent.handle_task(_make_task("make it blue"))
        call_messages = mock_complete.call_args[0][1]
        assert call_messages[0]["role"] == "system"

    @patch("app.llm.client.complete", new_callable=AsyncMock,
           return_value='```json\n{"action": "turn_on", "entity": "kitchen light", "parameters": {}}\n```\nTurning on the kitchen light.')
    async def test_handle_task_no_ha_client_returns_friendly_error(self, mock_complete):
        """When ha_client is None but LLM returns a valid action, return a friendly error."""
        agent = LightAgent(ha_client=None, entity_index=MagicMock())
        result = await agent.handle_task(_make_task("Turn on the kitchen light"))
        assert "unavailable" in result["speech"].lower()
        assert result["action_executed"] is None
        assert "json" not in result["speech"].lower()

    @patch("app.agents.light.execute_action", new_callable=AsyncMock,
           side_effect=Exception("HA connection lost"))
    @patch("app.llm.client.complete", new_callable=AsyncMock,
           return_value='```json\n{"action": "turn_on", "entity": "bedroom lamp", "parameters": {}}\n```\nTurning on the bedroom lamp.')
    async def test_handle_task_execute_action_exception(self, mock_complete, mock_exec):
        """When execute_action raises, return a friendly error instead of propagating."""
        agent = LightAgent(ha_client=MagicMock(), entity_index=MagicMock())
        result = await agent.handle_task(_make_task("Turn on the bedroom lamp"))
        assert "sorry" in result["speech"].lower()
        assert result["action_executed"] is None

    @patch("app.llm.client.complete", new_callable=AsyncMock,
           return_value='Here is some info about lights. {"action": "turn_on", "entity": "x", "parameters": {}} All done.')
    async def test_handle_task_strips_json_from_fallback(self, mock_complete):
        """When no action is parsed (parse_action returns None), JSON should be stripped from speech."""
        with patch("app.agents.light.parse_action", return_value=None):
            agent = LightAgent()
            result = await agent.handle_task(_make_task("tell me about lights"))
            assert "{" not in result["speech"]
            assert "action" not in result["speech"]

    @patch("app.agents.light.execute_action", new_callable=AsyncMock,
           return_value={"success": True, "entity_id": "light.kitchen", "new_state": "on", "speech": "Done."})
    @patch("app.llm.client.complete", new_callable=AsyncMock,
           return_value='```json\n{"action": "turn_on", "entity": "kitchen light", "parameters": {}}\n```\nDone.')
    async def test_handle_task_passes_agent_id_to_execute_action(self, mock_complete, mock_exec):
        agent = LightAgent(ha_client=MagicMock(), entity_index=MagicMock(), entity_matcher=MagicMock())
        await agent.handle_task(_make_task("Turn on kitchen light"))
        mock_exec.assert_awaited_once()
        _, kwargs = mock_exec.call_args
        assert kwargs.get("agent_id") == "light-agent"


class TestMusicAgent:

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Playing jazz.")
    async def test_handle_task_play_command(self, mock_complete):
        agent = MusicAgent()
        result = await agent.handle_task(_make_task("play some jazz music"))
        assert result["speech"] == "Playing jazz."

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Volume set to 30.")
    async def test_handle_task_volume_command(self, mock_complete):
        agent = MusicAgent()
        result = await agent.handle_task(_make_task("set volume to 30"))
        assert "Volume" in result["speech"] or "30" in result["speech"]

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="")
    async def test_handle_task_empty_llm_response(self, mock_complete):
        agent = MusicAgent()
        result = await agent.handle_task(_make_task("play jazz"))
        assert "did not return a response" in result["speech"]
        assert result["action_executed"] is None

    @patch("app.agents.music.execute_music_action", new_callable=AsyncMock,
           return_value={"success": True, "entity_id": "media_player.ma_kitchen", "new_state": "playing", "speech": "Done, Kitchen Speaker is now playing."})
    @patch("app.llm.client.complete", new_callable=AsyncMock,
           return_value='```json\n{"action": "play_media", "entity": "kitchen speaker", "parameters": {"media_id": "jazz"}}\n```\nPlaying jazz on the kitchen speaker.')
    async def test_handle_task_action_parsed_executes(self, mock_complete, mock_exec):
        agent = MusicAgent(ha_client=MagicMock(), entity_index=MagicMock(), entity_matcher=MagicMock())
        result = await agent.handle_task(_make_task("play jazz on kitchen speaker"))
        assert result["action_executed"]["success"] is True
        assert result["action_executed"]["entity_id"] == "media_player.ma_kitchen"

    @patch("app.llm.client.complete", new_callable=AsyncMock,
           return_value='```json\n{"action": "play_media", "entity": "kitchen speaker", "parameters": {"media_id": "jazz"}}\n```\nPlaying jazz.')
    async def test_handle_task_no_ha_client_returns_friendly_error(self, mock_complete):
        agent = MusicAgent(ha_client=None, entity_index=MagicMock())
        result = await agent.handle_task(_make_task("play jazz on kitchen speaker"))
        assert "unavailable" in result["speech"].lower()
        assert result["action_executed"] is None

    @patch("app.agents.music.execute_music_action", new_callable=AsyncMock,
           side_effect=Exception("HA connection lost"))
    @patch("app.llm.client.complete", new_callable=AsyncMock,
           return_value='```json\n{"action": "media_pause", "entity": "living room", "parameters": {}}\n```\nPausing.')
    async def test_handle_task_execute_action_exception(self, mock_complete, mock_exec):
        agent = MusicAgent(ha_client=MagicMock(), entity_index=MagicMock())
        result = await agent.handle_task(_make_task("pause the living room"))
        assert "sorry" in result["speech"].lower()
        assert result["action_executed"] is None

    @patch("app.llm.client.complete", new_callable=AsyncMock,
           return_value='Currently playing "Bohemian Rhapsody" on the kitchen speaker. {"action": "media_play", "entity": "x", "parameters": {}} Enjoy!')
    async def test_handle_task_strips_json_from_informational(self, mock_complete):
        with patch("app.agents.music.parse_action", return_value=None):
            agent = MusicAgent()
            result = await agent.handle_task(_make_task("what's playing?"))
            assert "{" not in result["speech"]
            assert "action" not in result["speech"]

    @patch("app.agents.music.execute_music_action", new_callable=AsyncMock,
           return_value={"success": True, "entity_id": "media_player.ma_kitchen", "new_state": "playing", "speech": "Done."})
    @patch("app.llm.client.complete", new_callable=AsyncMock,
           return_value='```json\n{"action": "play_media", "entity": "kitchen speaker", "parameters": {"media_id": "jazz"}}\n```\nPlaying.')
    async def test_handle_task_with_entity_matcher(self, mock_complete, mock_exec):
        matcher = MagicMock()
        agent = MusicAgent(ha_client=MagicMock(), entity_index=MagicMock(), entity_matcher=matcher)
        await agent.handle_task(_make_task("play jazz on kitchen speaker"))
        # entity_matcher should be passed through to execute_music_action
        call_args = mock_exec.call_args
        assert call_args[0][3] is matcher

    @patch("app.agents.music.execute_music_action", new_callable=AsyncMock,
           return_value={"success": True, "entity_id": "media_player.ma_kitchen", "new_state": "playing", "speech": "Done."})
    @patch("app.llm.client.complete", new_callable=AsyncMock,
           return_value='```json\n{"action": "play_media", "entity": "kitchen speaker", "parameters": {"media_id": "jazz"}}\n```\nPlaying.')
    async def test_handle_task_passes_agent_id_to_execute_music_action(self, mock_complete, mock_exec):
        agent = MusicAgent(ha_client=MagicMock(), entity_index=MagicMock(), entity_matcher=MagicMock())
        await agent.handle_task(_make_task("play jazz on kitchen speaker"))
        mock_exec.assert_awaited_once()
        _, kwargs = mock_exec.call_args
        assert kwargs.get("agent_id") == "music-agent"


class TestClimateAgent:

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Temperature set to 72F.")
    async def test_handle_task_set_temperature(self, mock_complete):
        agent = ClimateAgent()
        result = await agent.handle_task(_make_task("set thermostat to 72"))
        assert "72" in result["speech"]

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Switched to cool mode.")
    async def test_handle_task_set_mode(self, mock_complete):
        agent = ClimateAgent()
        result = await agent.handle_task(_make_task("set mode to cool"))
        assert "cool" in result["speech"]


class TestMediaAgent:

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Playing on TV.")
    async def test_handle_task_playback(self, mock_complete):
        agent = MediaAgent()
        result = await agent.handle_task(_make_task("play on the TV"))
        assert result["speech"] == "Playing on TV."

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Casting started.")
    async def test_handle_task_casting(self, mock_complete):
        agent = MediaAgent()
        result = await agent.handle_task(_make_task("cast to living room speaker"))
        assert result["speech"] == "Casting started."


class TestTimerAgent:

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Timer set for 5 minutes.")
    async def test_handle_task_set_timer(self, mock_complete):
        agent = TimerAgent()
        result = await agent.handle_task(_make_task("set a timer for 5 minutes"))
        assert "5 minutes" in result["speech"]

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Timer cancelled.")
    async def test_handle_task_cancel_timer(self, mock_complete):
        agent = TimerAgent()
        result = await agent.handle_task(_make_task("cancel the timer"))
        assert "cancelled" in result["speech"]


class TestSceneAgent:

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Movie scene activated.")
    async def test_handle_task_activate_scene(self, mock_complete):
        agent = SceneAgent()
        result = await agent.handle_task(_make_task("activate movie scene"))
        assert "activated" in result["speech"]

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Scene turned off.")
    async def test_handle_task_deactivate_scene(self, mock_complete):
        agent = SceneAgent()
        result = await agent.handle_task(_make_task("turn off the scene"))
        assert result["speech"] == "Scene turned off."


class TestAutomationAgent:

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Automation enabled.")
    async def test_handle_task_enable_automation(self, mock_complete):
        agent = AutomationAgent()
        result = await agent.handle_task(_make_task("enable morning routine automation"))
        assert "enabled" in result["speech"].lower() or "Automation" in result["speech"]

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Automation triggered.")
    async def test_handle_task_trigger_automation(self, mock_complete):
        agent = AutomationAgent()
        result = await agent.handle_task(_make_task("trigger the nighttime automation"))
        assert "triggered" in result["speech"].lower()


class TestSecurityAgentHandler:

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Front door locked.")
    async def test_handle_task_lock(self, mock_complete):
        agent = SecurityAgent()
        result = await agent.handle_task(_make_task("lock the front door"))
        assert "locked" in result["speech"].lower()

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Alarm armed in away mode.")
    async def test_handle_task_alarm_arm(self, mock_complete):
        agent = SecurityAgent()
        result = await agent.handle_task(_make_task("arm the alarm in away mode"))
        assert "armed" in result["speech"].lower()


class TestGeneralAgent:

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="The weather today is sunny.")
    async def test_handle_task_freeform_qa(self, mock_complete):
        agent = GeneralAgent()
        result = await agent.handle_task(_make_task("what is the weather like?"))
        assert "weather" in result["speech"].lower()

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="That is an interesting question.")
    async def test_handle_task_no_action_executed(self, mock_complete):
        agent = GeneralAgent()
        result = await agent.handle_task(_make_task("tell me a joke"))
        assert "action_executed" not in result or result.get("action_executed") is None

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Done.")
    async def test_handle_task_uses_system_prompt(self, mock_complete):
        agent = GeneralAgent()
        await agent.handle_task(_make_task("hello"))
        call_messages = mock_complete.call_args[0][1]
        assert call_messages[0]["role"] == "system"


# ---------------------------------------------------------------------------
# handle_task_stream default behavior
# ---------------------------------------------------------------------------

class TestBaseAgentStream:

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="All done.")
    async def test_handle_task_stream_default_wraps_handle_task(self, mock_complete):
        agent = LightAgent()
        task = _make_task("turn on kitchen light")
        chunks = []
        async for chunk in agent.handle_task_stream(task):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0]["token"] == "All done."
        assert chunks[0]["done"] is True


# ---------------------------------------------------------------------------
# RewriteAgent
# ---------------------------------------------------------------------------

class TestRewriteAgent:

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="I've turned on the light for you.")
    async def test_rewrite_returns_rephrased_text(self, mock_complete):
        agent = RewriteAgent()
        result = await agent.rewrite("Done, kitchen light is on.")
        assert result == "I've turned on the light for you."

    @patch("app.llm.client.complete", new_callable=AsyncMock, side_effect=Exception("LLM failure"))
    async def test_rewrite_fallback_on_failure(self, mock_complete):
        agent = RewriteAgent()
        result = await agent.rewrite("Done, kitchen light is on.")
        assert result == "Done, kitchen light is on."

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Rephrased text.")
    async def test_handle_task_a2a_interface(self, mock_complete):
        agent = RewriteAgent()
        result = await agent.handle_task(_make_task("Original cached text"))
        assert result["speech"] == "Rephrased text."

    def test_rewrite_agent_card(self):
        agent = RewriteAgent()
        assert agent.agent_card.agent_id == "rewrite-agent"
        assert "rewrite" in agent.agent_card.skills


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------

class TestOrchestratorAgent:

    def _make_orchestrator(self, dispatch_result=None):
        dispatcher = AsyncMock()
        registry = AsyncMock()
        cache_manager = MagicMock()
        cache_manager.process = AsyncMock(return_value=MagicMock(hit_type="miss", agent_id=None))

        # Mock dispatch response
        response_mock = MagicMock()
        response_mock.error = None
        response_mock.result = dispatch_result or {"speech": "Done!"}
        dispatcher.dispatch = AsyncMock(return_value=response_mock)

        registry.list_agents = AsyncMock(return_value=[
            AgentCard(agent_id="light-agent", name="Light Agent", description="", skills=["light"]),
            AgentCard(agent_id="music-agent", name="Music Agent", description="", skills=["music"]),
            AgentCard(agent_id="general-agent", name="General Agent", description="", skills=["general"]),
        ])

        orchestrator = OrchestratorAgent(
            dispatcher=dispatcher,
            registry=registry,
            cache_manager=cache_manager,
        )
        return orchestrator, dispatcher, registry, cache_manager

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_handle_task_routes_to_correct_agent(self, mock_complete, mock_track, mock_settings):
        orch, dispatcher, *_ = self._make_orchestrator()
        mock_complete.return_value = "light-agent: Turn on kitchen light"
        task = _make_task("turn on kitchen light", user_text="turn on kitchen light")
        task.conversation_id = "conv-1"
        result = await orch.handle_task(task)
        assert result["speech"] == "Done!"
        dispatcher.dispatch.assert_awaited_once()

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_handle_task_returns_routed_to_field(self, mock_complete, mock_track, mock_settings):
        orch, *_ = self._make_orchestrator()
        mock_complete.return_value = "light-agent: Turn on kitchen light"
        task = _make_task("turn on kitchen light")
        result = await orch.handle_task(task)
        assert result["routed_to"] == "light-agent"

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_handle_task_fallback_on_unknown_agent(self, mock_complete, mock_track, mock_settings):
        orch, *_ = self._make_orchestrator()
        mock_complete.return_value = "unknown-agent: do something"
        task = _make_task("something random")
        result = await orch.handle_task(task)
        # Should fall back to general-agent
        assert result["routed_to"] == "general-agent"

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_agent_timeout", new_callable=AsyncMock)
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_handle_task_fallback_on_timeout(self, mock_complete, mock_track, mock_timeout, mock_settings):
        orch, dispatcher, *_ = self._make_orchestrator()
        mock_complete.return_value = "light-agent: Turn on light"
        orch._default_timeout = 0.001  # very short timeout

        # First dispatch times out, fallback succeeds
        fallback_response = MagicMock()
        fallback_response.error = None
        fallback_response.result = {"speech": "Fallback response."}
        dispatcher.dispatch = AsyncMock(side_effect=[asyncio.TimeoutError(), fallback_response])

        task = _make_task("turn on kitchen light")
        result = await orch.handle_task(task)
        assert result["speech"] == "Fallback response."

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_handle_task_fallback_on_dispatch_error(self, mock_complete, mock_track, mock_settings):
        orch, dispatcher, *_ = self._make_orchestrator()
        mock_complete.return_value = "light-agent: Turn on light"

        # First dispatch returns error, fallback succeeds
        error_response = MagicMock()
        error_response.error = MagicMock(message="Agent error")
        error_response.result = None

        ok_response = MagicMock()
        ok_response.error = None
        ok_response.result = {"speech": "General answered."}

        dispatcher.dispatch = AsyncMock(side_effect=[error_response, ok_response])
        task = _make_task("turn on kitchen light")
        result = await orch.handle_task(task)
        assert result["speech"] == "General answered."

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_classification_falls_back_on_llm_failure(self, mock_complete, mock_track, mock_settings):
        orch, *_ = self._make_orchestrator()
        mock_complete.side_effect = Exception("LLM error")
        task = _make_task("turn on kitchen light")
        result = await orch.handle_task(task)
        assert result["routed_to"] == "general-agent"

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_conversation_turns_stored(self, mock_complete, mock_track, mock_settings):
        orch, *_ = self._make_orchestrator()
        mock_complete.return_value = "light-agent: Turn on light"
        task = _make_task("turn on kitchen light")
        task.conversation_id = "conv-test"
        await orch.handle_task(task)
        turns = orch._conversations.get("conv-test", [])
        assert len(turns) == 2  # user + assistant

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_conversation_turns_limited(self, mock_complete, mock_track, mock_settings):
        orch, *_ = self._make_orchestrator()
        mock_complete.return_value = "general-agent: answer"
        for i in range(10):
            task = _make_task(f"Question {i}")
            task.conversation_id = "conv-limit"
            await orch.handle_task(task)
        turns = orch._conversations.get("conv-limit", [])
        # _MAX_TURNS = 3, so max 6 messages (3 pairs)
        assert len(turns) <= 6

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    async def test_classify_returns_routing_cached_on_cache_hit(self, mock_track, mock_settings):
        orch, *_ = self._make_orchestrator()
        # Configure cache to return a routing hit
        orch._cache_manager.process = AsyncMock(
            return_value=MagicMock(hit_type="routing_hit", agent_id="light-agent")
        )
        classifications, routing_cached = await orch._classify("turn on kitchen light")
        assert classifications[0][0] == "light-agent"
        assert classifications[0][2] == 1.0
        assert routing_cached is True

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_classify_returns_not_cached_on_llm_path(self, mock_complete, mock_track, mock_settings):
        orch, *_ = self._make_orchestrator()
        mock_complete.return_value = "light-agent: Turn on kitchen light"
        classifications, routing_cached = await orch._classify("turn on kitchen light")
        assert classifications[0][0] == "light-agent"
        assert routing_cached is False

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_presence_room_injected_into_context(self, mock_complete, mock_track, mock_settings):
        orch, dispatcher, *_ = self._make_orchestrator()
        mock_complete.return_value = "light-agent: Turn on light"

        detector = MagicMock()
        detector.get_most_likely_room.return_value = "kitchen"
        orch._presence_detector = detector

        task = _make_task("turn on the light")
        await orch.handle_task(task)

        # Check the dispatched task has presence_room in context
        call_args = dispatcher.dispatch.call_args[0][0]
        dispatched_task = call_args.params["task"]
        assert dispatched_task["context"]["presence_room"] == "kitchen"

    def test_orchestrator_agent_card(self):
        orch = OrchestratorAgent(dispatcher=AsyncMock())
        card = orch.agent_card
        assert card.agent_id == "orchestrator"
        assert "intent_classification" in card.skills

    @patch("app.agents.orchestrator.SettingsRepository")
    async def test_initialize_loads_reliability_config(self, mock_settings):
        mock_settings.get_value = AsyncMock(side_effect=lambda key, default=None: {
            "a2a.default_timeout": "10",
            "a2a.max_iterations": "5",
        }.get(key, default))
        orch = OrchestratorAgent(dispatcher=AsyncMock())
        await orch.initialize()
        assert orch._default_timeout == 10
        assert orch._max_iterations == 5

    async def test_parse_classification_valid(self):
        orch = OrchestratorAgent(dispatcher=AsyncMock())
        orch._registry = AsyncMock()
        orch._registry.list_agents = AsyncMock(return_value=[
            AgentCard(agent_id="light-agent", name="", description="", skills=[]),
        ])
        results = await orch._parse_classification("light-agent: Turn on kitchen light", "turn on kitchen light")
        assert len(results) == 1
        assert results[0][0] == "light-agent"
        assert results[0][1] == "Turn on kitchen light"
        assert results[0][2] == 0.8  # default when no confidence in format

    async def test_parse_classification_no_colon_falls_back(self):
        orch = OrchestratorAgent(dispatcher=AsyncMock())
        results = await orch._parse_classification("gibberish", "original text")
        assert len(results) == 1
        assert results[0][0] == "general-agent"
        assert results[0][1] == "original text"
        assert results[0][2] == 0.0

    async def test_parse_classification_multi_line(self):
        orch = OrchestratorAgent(dispatcher=AsyncMock())
        orch._registry = AsyncMock()
        orch._registry.list_agents = AsyncMock(return_value=[
            AgentCard(agent_id="light-agent", name="", description="", skills=[]),
            AgentCard(agent_id="music-agent", name="", description="", skills=[]),
        ])
        response = "light-agent (95%): turn on the shelf\nmusic-agent (90%): play jazz"
        results = await orch._parse_classification(response, "original")
        assert len(results) == 2
        assert results[0][0] == "light-agent"
        assert results[0][2] == 0.95
        assert results[1][0] == "music-agent"
        assert results[1][2] == 0.90

    async def test_parse_classification_unknown_agent_skipped(self):
        orch = OrchestratorAgent(dispatcher=AsyncMock())
        orch._registry = AsyncMock()
        orch._registry.list_agents = AsyncMock(return_value=[
            AgentCard(agent_id="light-agent", name="", description="", skills=[]),
        ])
        response = "light-agent (95%): turn on light\nfake-agent (80%): do stuff"
        results = await orch._parse_classification(response, "original")
        assert len(results) == 1
        assert results[0][0] == "light-agent"

    async def test_parse_classification_cap_at_3(self):
        orch = OrchestratorAgent(dispatcher=AsyncMock())
        orch._registry = AsyncMock()
        orch._registry.list_agents = AsyncMock(return_value=[
            AgentCard(agent_id="light-agent", name="", description="", skills=[]),
            AgentCard(agent_id="music-agent", name="", description="", skills=[]),
            AgentCard(agent_id="climate-agent", name="", description="", skills=[]),
            AgentCard(agent_id="timer-agent", name="", description="", skills=[]),
        ])
        response = (
            "light-agent (95%): a\nmusic-agent (90%): b\n"
            "climate-agent (85%): c\ntimer-agent (80%): d"
        )
        results = await orch._parse_classification(response, "original")
        assert len(results) == 3

    async def test_parse_classification_dedup_same_agent(self):
        orch = OrchestratorAgent(dispatcher=AsyncMock())
        orch._registry = AsyncMock()
        orch._registry.list_agents = AsyncMock(return_value=[
            AgentCard(agent_id="light-agent", name="", description="", skills=[]),
        ])
        response = "light-agent (90%): task one\nlight-agent (80%): task two"
        results = await orch._parse_classification(response, "original")
        assert len(results) == 1
        assert results[0][0] == "light-agent"
        assert results[0][2] == 0.9
        assert "task one" in results[0][1]
        assert "task two" in results[0][1]

    @patch("app.agents.orchestrator.SettingsRepository")
    async def test_mediate_response_disabled_by_default(self, mock_settings):
        """When personality.prompt is empty, mediation returns speech unchanged."""
        orch, *_ = self._make_orchestrator()
        mock_settings.get_value = AsyncMock(return_value="")
        result = await orch._mediate_response("Done, light is on.", "turn on light", "light-agent")
        assert result == "Done, light is on."

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_mediate_response_with_personality(self, mock_complete, mock_settings):
        """When personality.prompt is set, calls LLM with personality."""
        orch, *_ = self._make_orchestrator()
        mock_settings.get_value = AsyncMock(side_effect=lambda k, d=None: {
            "personality.prompt": "You are Lucia, a friendly assistant.",
            "rewrite.model": "groq/llama-3.1-8b-instant",
            "rewrite.temperature": "0.3",
        }.get(k, d))
        mock_complete.return_value = "Hey there! The light is now on."
        result = await orch._mediate_response(
            "Done, light is on.", "turn on light", "light-agent"
        )
        assert result == "Hey there! The light is now on."
        mock_complete.assert_awaited_once()

    @patch("app.agents.orchestrator.SettingsRepository")
    async def test_mediate_response_empty_speech(self, mock_settings):
        """When agent speech is empty, returns it unchanged even with personality."""
        orch, *_ = self._make_orchestrator()
        mock_settings.get_value = AsyncMock(return_value="You are a friendly assistant.")
        result = await orch._mediate_response("", "turn on light", "light-agent")
        assert result == ""

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_handle_task_creates_return_span(self, mock_complete, mock_track, mock_settings):
        """handle_task should create a 'return' span when a span_collector is present."""
        from app.analytics.tracer import SpanCollector
        orch, *_ = self._make_orchestrator()
        mock_complete.return_value = "light-agent: Turn on light"
        mock_settings.get_value = AsyncMock(return_value="false")
        collector = SpanCollector("trace-return-test")
        task = _make_task("turn on light")
        task._span_collector = collector
        task.conversation_id = "conv-ret"
        with patch("app.analytics.tracer.create_trace_summary", new_callable=AsyncMock):
            await orch.handle_task(task)
        span_names = [s["span_name"] for s in collector._spans]
        assert "return" in span_names
        ret_span = [s for s in collector._spans if s["span_name"] == "return"][0]
        assert ret_span["agent_id"] == "orchestrator"
        assert ret_span["metadata"]["from_agent"] == "light-agent"
        assert "final_response" in ret_span["metadata"]
        assert "mediated" in ret_span["metadata"]

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_handle_task_multi_agent_parallel_dispatch(self, mock_complete, mock_track, mock_settings):
        """Multi-agent classification dispatches to multiple agents and merges via LLM."""
        orch, dispatcher, *_ = self._make_orchestrator()
        merged_text = "The shelf light is now on, and jazz is playing."
        # First call: classification. Second call: LLM merge.
        mock_complete.side_effect = [
            "light-agent (95%): turn on shelf\nmusic-agent (90%): play jazz",
            merged_text,
        ]
        mock_settings.get_value = AsyncMock(side_effect=lambda k, d=None: {
            "personality.prompt": "",
            "rewrite.model": "groq/llama-3.1-8b-instant",
            "rewrite.temperature": "0.3",
        }.get(k, d))

        # Dispatcher returns different responses per agent
        response_light = MagicMock()
        response_light.error = None
        response_light.result = {"speech": "Shelf is on."}
        response_music = MagicMock()
        response_music.error = None
        response_music.result = {"speech": "Playing jazz."}
        dispatcher.dispatch = AsyncMock(side_effect=[response_light, response_music])

        task = _make_task("turn on shelf and play jazz", user_text="turn on shelf and play jazz")
        task.conversation_id = "conv-multi"
        result = await orch.handle_task(task)
        assert result["speech"] == merged_text
        assert "light-agent" in result["routed_to"]
        assert "music-agent" in result["routed_to"]
        # LLM called twice: once for classify, once for merge
        assert mock_complete.await_count == 2

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.agents.orchestrator.track_agent_timeout", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_handle_task_multi_agent_partial_timeout(self, mock_complete, mock_track, mock_timeout, mock_settings):
        """When one agent times out in multi-dispatch, partial results are merged."""
        orch, dispatcher, *_ = self._make_orchestrator()
        # Classification call, then merge call
        mock_complete.side_effect = [
            "light-agent (95%): turn on shelf\nmusic-agent (90%): play jazz",
            "Here is the merged result.",
        ]
        mock_settings.get_value = AsyncMock(side_effect=lambda k, d=None: {
            "personality.prompt": "",
            "rewrite.model": "groq/llama-3.1-8b-instant",
            "rewrite.temperature": "0.3",
        }.get(k, d))
        orch._default_timeout = 0.001

        # First dispatch times out then fallback, second succeeds
        fallback_resp = MagicMock()
        fallback_resp.error = None
        fallback_resp.result = {"speech": "Fallback."}
        dispatcher.dispatch = AsyncMock(side_effect=[
            asyncio.TimeoutError(), fallback_resp,  # light-agent -> timeout -> fallback
            asyncio.TimeoutError(), MagicMock(error=None, result={"speech": "Jazz."}),  # music-agent -> timeout -> fallback
        ])

        task = _make_task("turn on shelf and play jazz")
        task.conversation_id = "conv-multi-timeout"
        result = await orch.handle_task(task)
        assert result["speech"]  # should have some content

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_handle_task_stream_multi_agent_yields_single_chunk(self, mock_complete, mock_track, mock_settings):
        """Streaming with multi-agent falls back to handle_task and yields one chunk."""
        orch, dispatcher, *_ = self._make_orchestrator()
        merged_text = "Shelf is on and jazz is playing."
        # Stream classify + merge (no duplicate classify thanks to _pre_classified)
        mock_complete.side_effect = [
            "light-agent (95%): turn on shelf\nmusic-agent (90%): play jazz",
            merged_text,
        ]
        mock_settings.get_value = AsyncMock(side_effect=lambda k, d=None: {
            "personality.prompt": "",
            "rewrite.model": "groq/llama-3.1-8b-instant",
            "rewrite.temperature": "0.3",
        }.get(k, d))

        response_light = MagicMock()
        response_light.error = None
        response_light.result = {"speech": "Shelf is on."}
        response_music = MagicMock()
        response_music.error = None
        response_music.result = {"speech": "Playing jazz."}
        dispatcher.dispatch = AsyncMock(side_effect=[response_light, response_music])

        task = _make_task("turn on shelf and play jazz")
        task.conversation_id = "conv-stream-multi"
        chunks = []
        async for chunk in orch.handle_task_stream(task):
            chunks.append(chunk)
        # Multi-agent streaming yields a single chunk with done=True
        assert any(c["done"] for c in chunks)
        full = "".join(c.get("token", "") for c in chunks)
        assert full == merged_text

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_classify_multi_agent_skips_cache_store(self, mock_complete, mock_track, mock_settings):
        """Multi-agent results should NOT be cached."""
        orch, *_ = self._make_orchestrator()
        mock_complete.return_value = "light-agent (95%): a\nmusic-agent (90%): b"
        classifications, routing_cached = await orch._classify("do two things")
        assert len(classifications) == 2
        assert routing_cached is False
        orch._cache_manager.store_routing.assert_not_called()

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_classify_single_agent_stores_cache(self, mock_complete, mock_track, mock_settings):
        """Single-agent results should still be cached."""
        orch, *_ = self._make_orchestrator()
        mock_complete.return_value = "light-agent (95%): turn on light"
        classifications, routing_cached = await orch._classify("turn on light")
        assert len(classifications) == 1
        orch._cache_manager.store_routing.assert_called_once()

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_handle_task_pre_classified_skips_classify(self, mock_complete, mock_track, mock_settings):
        """handle_task with _pre_classified skips the _classify() call entirely."""
        orch, dispatcher, *_ = self._make_orchestrator()
        mock_settings.get_value = AsyncMock(return_value="")
        pre = ([("light-agent", "turn on light", 0.95)], False)
        task = _make_task("turn on light")
        task.conversation_id = "conv-pre"
        result = await orch.handle_task(task, _pre_classified=pre)
        assert result["routed_to"] == "light-agent"
        # LLM should NOT be called for classification (only dispatch happens)
        mock_complete.assert_not_awaited()

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_handle_task_stream_multi_agent_classifies_once(self, mock_complete, mock_track, mock_settings):
        """Streaming multi-agent should call _classify only once (not twice)."""
        orch, dispatcher, *_ = self._make_orchestrator()
        merged = "Done both."
        mock_complete.side_effect = [
            "light-agent (95%): on\nmusic-agent (90%): play",
            merged,
        ]
        mock_settings.get_value = AsyncMock(side_effect=lambda k, d=None: {
            "personality.prompt": "",
            "rewrite.model": "groq/llama-3.1-8b-instant",
            "rewrite.temperature": "0.3",
        }.get(k, d))
        r1 = MagicMock(error=None, result={"speech": "On."})
        r2 = MagicMock(error=None, result={"speech": "Play."})
        dispatcher.dispatch = AsyncMock(side_effect=[r1, r2])
        task = _make_task("do both")
        task.conversation_id = "conv-once"
        chunks = [c async for c in orch.handle_task_stream(task)]
        assert any(c["done"] for c in chunks)
        # Only 2 LLM calls: classify + merge (no duplicate classify)
        assert mock_complete.await_count == 2

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.agents.orchestrator.track_request", new_callable=AsyncMock)
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_dispatch_span_includes_condensed_task(self, mock_complete, mock_track, mock_settings):
        """Dispatch span metadata should include condensed_task."""
        from app.analytics.tracer import SpanCollector
        orch, *_ = self._make_orchestrator()
        mock_complete.return_value = "light-agent: Turn on kitchen light"
        mock_settings.get_value = AsyncMock(return_value="")
        collector = SpanCollector("trace-cond-test")
        task = _make_task("turn on kitchen light")
        task._span_collector = collector
        task.conversation_id = "conv-cond"
        with patch("app.analytics.tracer.create_trace_summary", new_callable=AsyncMock):
            await orch.handle_task(task)
        dispatch_spans = [s for s in collector._spans if s["span_name"] == "dispatch"]
        assert len(dispatch_spans) == 1
        assert "condensed_task" in dispatch_spans[0]["metadata"]
        assert dispatch_spans[0]["metadata"]["condensed_task"]


# ---------------------------------------------------------------------------
# LightAgent empty response guard
# ---------------------------------------------------------------------------

class TestLightAgentEmptyResponse:

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="")
    async def test_empty_string_response(self, mock_complete):
        agent = LightAgent()
        result = await agent.handle_task(_make_task("turn on light"))
        assert "did not return a response" in result["speech"]
        assert result["action_executed"] is None

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value=None)
    async def test_none_response(self, mock_complete):
        agent = LightAgent()
        result = await agent.handle_task(_make_task("turn on light"))
        assert "did not return a response" in result["speech"]
        assert result["action_executed"] is None


# ---------------------------------------------------------------------------
# DynamicAgent
# ---------------------------------------------------------------------------

class TestDynamicAgent:

    def test_agent_card_has_custom_prefix(self):
        agent = DynamicAgent(
            name="my-tool",
            description="A custom tool",
            system_prompt="You are a tool helper.",
            skills=["tool_use"],
        )
        card = agent.agent_card
        assert card.agent_id == "custom-my-tool"
        assert card.name == "my-tool"

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="Custom response.")
    async def test_handle_task_uses_custom_system_prompt(self, mock_complete):
        agent = DynamicAgent(
            name="helper",
            description="desc",
            system_prompt="You are a custom helper.",
            skills=["help"],
        )
        result = await agent.handle_task(_make_task("help me"))
        assert result["speech"] == "Custom response."
        call_messages = mock_complete.call_args[0][1]
        assert "custom helper" in call_messages[0]["content"]

    @patch("app.llm.client.complete", new_callable=AsyncMock, return_value="resp")
    async def test_handle_task_appends_name_preservation_instruction(self, mock_complete):
        agent = DynamicAgent(name="x", description="", system_prompt="base", skills=[])
        await agent.handle_task(_make_task("test"))
        system_msg = mock_complete.call_args[0][1][0]["content"]
        assert "NEVER translate or normalize entity/room names" in system_msg


# ---------------------------------------------------------------------------
# CustomAgentLoader
# ---------------------------------------------------------------------------

class TestCustomAgentLoader:

    @patch("app.agents.custom_loader.CustomAgentRepository")
    async def test_load_all_registers_agents(self, mock_repo):
        mock_repo.list_enabled = AsyncMock(return_value=[
            {"name": "toolbot", "description": "A tool bot", "system_prompt": "sys", "intent_patterns": ["tool_use"]},
        ])
        registry = AsyncMock()
        loader = CustomAgentLoader(registry=registry)
        count = await loader.load_all()
        assert count == 1
        registry.register.assert_awaited_once()

    @patch("app.agents.custom_loader.CustomAgentRepository")
    async def test_load_all_handles_single_bad_row(self, mock_repo):
        mock_repo.list_enabled = AsyncMock(return_value=[
            {"name": "bad"},  # missing system_prompt will cause _load_one to fail
        ])
        registry = AsyncMock()
        loader = CustomAgentLoader(registry=registry)
        # _load_one fails but load_all catches it and continues
        count = await loader.load_all()
        assert count == 0

    @patch("app.agents.custom_loader.CustomAgentRepository")
    async def test_reload_unregisters_then_reloads(self, mock_repo):
        mock_repo.list_enabled = AsyncMock(return_value=[
            {"name": "bot1", "description": "d", "system_prompt": "s", "intent_patterns": []},
        ])
        registry = AsyncMock()
        loader = CustomAgentLoader(registry=registry)
        await loader.load_all()
        assert len(loader._loaded) == 1

        # Reload
        mock_repo.list_enabled = AsyncMock(return_value=[
            {"name": "bot2", "description": "d2", "system_prompt": "s2", "intent_patterns": []},
        ])
        count = await loader.reload()
        assert count == 1
        registry.unregister.assert_awaited()

    @patch("app.agents.custom_loader.CustomAgentRepository")
    async def test_load_uses_intent_patterns_as_skills(self, mock_repo):
        mock_repo.list_enabled = AsyncMock(return_value=[
            {"name": "custom1", "description": "d", "system_prompt": "s", "intent_patterns": ["skill_a", "skill_b"]},
        ])
        registry = AsyncMock()
        loader = CustomAgentLoader(registry=registry)
        await loader.load_all()
        registered_agent = registry.register.call_args[0][0]
        assert registered_agent.agent_card.skills == ["skill_a", "skill_b"]

    @patch("app.agents.custom_loader.CustomAgentRepository")
    async def test_load_defaults_skills_to_name_when_no_patterns(self, mock_repo):
        mock_repo.list_enabled = AsyncMock(return_value=[
            {"name": "mybot", "description": "d", "system_prompt": "s", "intent_patterns": []},
        ])
        registry = AsyncMock()
        loader = CustomAgentLoader(registry=registry)
        await loader.load_all()
        registered_agent = registry.register.call_args[0][0]
        assert registered_agent.agent_card.skills == ["mybot"]


# ---------------------------------------------------------------------------
# Merge responses action status and fallback 3-tuple
# ---------------------------------------------------------------------------

class TestMergeResponsesActionStatus:

    @patch("app.agents.orchestrator.SettingsRepository")
    @patch("app.llm.client.complete", new_callable=AsyncMock)
    async def test_merge_responses_includes_action_status_in_prompt(self, mock_complete, mock_settings):
        """Action status markers should appear in the LLM prompt sent to the merge call."""
        orch = OrchestratorAgent(dispatcher=AsyncMock())
        mock_settings.get_value = AsyncMock(side_effect=lambda k, d=None: {
            "personality.prompt": "",
            "rewrite.model": "groq/llama-3.1-8b-instant",
            "rewrite.temperature": "0.3",
        }.get(k, d))
        mock_complete.return_value = "Merged result."

        agent_responses = [
            ("light-agent", "Light is on.", True),
            ("music-agent", "Could not play music.", False),
        ]
        result = await orch._merge_responses(agent_responses, "turn on light and play music")
        assert result == "Merged result."

        # Check the messages sent to the LLM contain action status markers
        call_messages = mock_complete.call_args[0][1]
        user_content = call_messages[1]["content"]
        assert "[action executed]" in user_content
        assert "[no action executed]" in user_content

    def test_merge_responses_fallback_handles_3_tuple(self):
        """_format_fallback should handle 3-tuple format without errors."""
        responses = [
            ("light-agent", "Light is on.", True),
            ("music-agent", "", False),
            ("general-agent", "Here is info.", True),
        ]
        result = OrchestratorAgent._format_fallback(responses)
        assert "[light-agent] Light is on." in result
        assert "[general-agent] Here is info." in result
        # Empty speech should be filtered out
        assert "music-agent" not in result


# ---------------------------------------------------------------------------
# AgentConfig default temperature
# ---------------------------------------------------------------------------

class TestAgentConfigDefaultTemperature:

    def test_default_temperature_is_0_2(self):
        from app.models.agent import AgentConfig
        config = AgentConfig(agent_id="test-agent")
        assert config.temperature == 0.2
