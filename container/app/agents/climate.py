"""Climate and HVAC control agent."""

from __future__ import annotations

import logging

from app.agents.base import BaseAgent
from app.models.agent import AgentCard, AgentTask

logger = logging.getLogger(__name__)


class ClimateAgent(BaseAgent):
    """Controls climate and HVAC devices via HA REST API."""

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="climate-agent",
            name="Climate Agent",
            description="Controls climate and HVAC: temperature, mode, fan speed, humidity.",
            skills=["temperature", "hvac_mode", "fan_speed", "humidity"],
            endpoint="local://climate-agent",
        )

    async def handle_task(self, task: AgentTask) -> dict:
        system_prompt = self._load_prompt("climate")
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
