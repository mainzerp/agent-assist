"""General fallback agent for unroutable requests."""

from __future__ import annotations

import logging

from app.agents.base import BaseAgent
from app.models.agent import AgentCard, AgentTask, TaskResult

logger = logging.getLogger(__name__)


class GeneralAgent(BaseAgent):
    """Handles general Q&A and unroutable requests. No HA service calls."""

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="general-agent",
            name="General Agent",
            description="Handles general questions and requests that do not fall into a specific device control domain.",
            skills=["general_qa", "fallback"],
            endpoint="local://general-agent",
        )

    async def handle_task(self, task: AgentTask) -> TaskResult:
        system_prompt = self._load_prompt("general")
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
        return TaskResult(speech=response)
