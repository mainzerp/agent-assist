"""Timer and alarm agent."""

from __future__ import annotations

import logging

from app.agents.base import BaseAgent
from app.models.agent import AgentCard, AgentTask

logger = logging.getLogger(__name__)


class TimerAgent(BaseAgent):
    """Controls timers and reminders via HA REST API."""

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="timer-agent",
            name="Timer Agent",
            description="Manages timers, alarms, and reminders.",
            skills=["timer_set", "timer_cancel", "reminder"],
            endpoint="local://timer-agent",
        )

    async def handle_task(self, task: AgentTask) -> dict:
        system_prompt = self._load_prompt("timer")
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
