"""Light control agent with direct HA REST API execution."""

from __future__ import annotations

import logging

from app.agents.base import BaseAgent
from app.models.agent import AgentCard, AgentTask

logger = logging.getLogger(__name__)


class LightAgent(BaseAgent):
    """Controls lighting devices via HA REST API."""

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="light-agent",
            name="Light Agent",
            description="Controls lighting devices: on/off, brightness, color, color temperature.",
            skills=["light_control", "brightness", "color"],
            endpoint="local://light-agent",
        )

    async def handle_task(self, task: AgentTask) -> dict:
        system_prompt = self._load_prompt("light")
        messages = [{"role": "system", "content": system_prompt}]

        if task.context and task.context.conversation_turns:
            for turn in task.context.conversation_turns:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })

        # task.description = condensed task from orchestrator (primary input)
        # task.user_text = original unmodified user text (fallback only)
        messages.append({"role": "user", "content": task.description})

        response = await self._call_llm(messages)

        # Phase 1.9: parse response for structured action, resolve entity,
        # execute via ha_client, verify result. For now, return LLM response.
        return {"speech": response, "action_executed": None}
