"""Scene activation agent."""

from __future__ import annotations

import logging

from app.agents.base import BaseAgent
from app.models.agent import AgentCard, AgentTask

logger = logging.getLogger(__name__)


class SceneAgent(BaseAgent):
    """Activates and manages scenes via HA REST API."""

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="scene-agent",
            name="Scene Agent",
            description="Activates and manages Home Assistant scenes.",
            skills=["scene_activate", "scene_list"],
            endpoint="local://scene-agent",
        )

    async def handle_task(self, task: AgentTask) -> dict:
        system_prompt = self._load_prompt("scene")
        messages = [{"role": "system", "content": system_prompt}]

        if task.context and task.context.conversation_turns:
            for turn in task.context.conversation_turns:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })

        messages.append({"role": "user", "content": task.description})
        response = await self._call_llm(messages)
        return {"speech": response, "action_executed": None}
