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
        agent_id, condensed = await orch._parse_classification("light-agent: Turn on kitchen light", "turn on kitchen light")
        assert agent_id == "light-agent"
        assert condensed == "Turn on kitchen light"

    async def test_parse_classification_no_colon_falls_back(self):
        orch = OrchestratorAgent(dispatcher=AsyncMock())
        agent_id, condensed = await orch._parse_classification("gibberish", "original text")
        assert agent_id == "general-agent"
        assert condensed == "original text"


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
