"""Dynamic loader for runtime-created custom agents."""

from __future__ import annotations

import logging
from typing import Any

from app.agents.base import BaseAgent
from app.models.agent import AgentCard, AgentTask
from app.a2a.registry import AgentRegistry
from app.db.repository import CustomAgentRepository

logger = logging.getLogger(__name__)


class DynamicAgent(BaseAgent):
    """A dynamically-created agent from the custom_agents DB table."""

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        skills: list[str],
        ha_client=None,
        entity_index=None,
    ) -> None:
        super().__init__(ha_client=ha_client, entity_index=entity_index)
        self._name = name
        self._description = description
        self._system_prompt = system_prompt
        self._skills = skills

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id=f"custom-{self._name}",
            name=self._name,
            description=self._description,
            skills=self._skills,
            endpoint=f"local://custom-{self._name}",
        )

    async def handle_task(self, task: AgentTask) -> dict:
        prompt = self._system_prompt + "\nNEVER translate or normalize entity/room names."
        messages = [{"role": "system", "content": prompt}]

        if task.context and task.context.conversation_turns:
            for turn in task.context.conversation_turns:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })

        messages.append({"role": "user", "content": task.description})
        response = await self._call_llm(messages)
        return {"speech": response, "action_executed": None}


class CustomAgentLoader:
    """Loads custom agent definitions from DB and registers with A2A."""

    def __init__(self, registry: AgentRegistry, ha_client=None, entity_index=None) -> None:
        self._registry = registry
        self._ha_client = ha_client
        self._entity_index = entity_index
        self._loaded: dict[str, DynamicAgent] = {}

    async def load_all(self) -> int:
        """Load all enabled custom agents from DB and register."""
        agents = await CustomAgentRepository.list_enabled()
        count = 0
        for row in agents:
            try:
                await self._load_one(row)
                count += 1
            except Exception:
                logger.error("Failed to load custom agent '%s'", row.get("name"), exc_info=True)
        logger.info("Loaded %d custom agents", count)
        return count

    async def reload(self) -> int:
        """Hot reload: unregister all custom agents, re-load from DB."""
        for agent_id in list(self._loaded.keys()):
            await self._registry.unregister(agent_id)
        self._loaded.clear()
        return await self.load_all()

    async def _load_one(self, row: dict[str, Any]) -> None:
        name = row["name"]
        intent_patterns = row.get("intent_patterns") or []
        skills = intent_patterns if intent_patterns else [name]
        agent = DynamicAgent(
            name=name,
            description=row.get("description", ""),
            system_prompt=row["system_prompt"],
            skills=skills,
            ha_client=self._ha_client,
            entity_index=self._entity_index,
        )
        await self._registry.register(agent)
        self._loaded[agent.agent_card.agent_id] = agent
