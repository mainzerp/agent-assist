"""Light control agent with direct HA REST API execution."""

from __future__ import annotations

import logging

from app.agents.base import BaseAgent
from app.agents.action_executor import parse_action, execute_action
from app.models.agent import AgentCard, AgentTask

logger = logging.getLogger(__name__)


class LightAgent(BaseAgent):
    """Controls lighting devices via HA REST API."""

    def __init__(self, ha_client=None, entity_index=None, entity_matcher=None) -> None:
        super().__init__(ha_client=ha_client, entity_index=entity_index)
        self._entity_matcher = entity_matcher

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

        # Parse structured action from LLM response
        action = parse_action(response)
        if action and self._ha_client:
            result = await execute_action(
                action,
                self._ha_client,
                self._entity_index,
                self._entity_matcher,
            )
            return {
                "speech": result["speech"],
                "action_executed": {
                    "action": action.get("action"),
                    "entity_id": result.get("entity_id"),
                    "success": result.get("success"),
                    "new_state": result.get("new_state"),
                },
            }

        # No action parsed (informational query) -- return LLM text as-is
        return {"speech": response, "action_executed": None}
