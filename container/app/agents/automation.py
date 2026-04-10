"""Automation management agent."""

from __future__ import annotations

import logging

from app.agents.base import BaseAgent
from app.models.agent import AgentCard, AgentTask

logger = logging.getLogger(__name__)


class AutomationAgent(BaseAgent):
    """Manages Home Assistant automations via HA REST API."""

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="automation-agent",
            name="Automation Agent",
            description="Enables, disables, and triggers automations.",
            skills=["automation_enable", "automation_disable", "automation_trigger"],
            endpoint="local://automation-agent",
        )

    async def handle_task(self, task: AgentTask) -> dict:
        system_prompt = self._load_prompt("automation")
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
