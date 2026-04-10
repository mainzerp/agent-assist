"""Security system agent."""

from __future__ import annotations

import logging

from app.agents.base import BaseAgent
from app.models.agent import AgentCard, AgentTask

logger = logging.getLogger(__name__)


class SecurityAgent(BaseAgent):
    """Controls security devices via HA REST API."""

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="security-agent",
            name="Security Agent",
            description="Controls locks, alarm panels, and camera status.",
            skills=["lock_control", "alarm_control", "camera_status"],
            endpoint="local://security-agent",
        )

    async def handle_task(self, task: AgentTask) -> dict:
        system_prompt = self._load_prompt("security")
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
