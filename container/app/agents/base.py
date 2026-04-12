"""Base agent class with HA client and entity index access."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncGenerator

from app.models.agent import AgentCard, AgentTask

logger = logging.getLogger(__name__)

# Prompts directory (container/app/prompts/)
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class BaseAgent(ABC):
    """Abstract base class for all specialized agents.

    Subclasses must implement handle_task(). Optionally override
    handle_task_stream() for token-level streaming support.
    """

    def __init__(
        self,
        ha_client=None,
        entity_index=None,
    ) -> None:
        self._ha_client = ha_client
        self._entity_index = entity_index

    @property
    @abstractmethod
    def agent_card(self) -> AgentCard:
        """Return the AgentCard describing this agent's capabilities."""
        ...

    @abstractmethod
    async def handle_task(self, task: AgentTask) -> dict:
        """Process a task and return the full result.

        Returns:
            dict with at least {"speech": str}. May include
            {"action_executed": {...}} for HA action results.
        """
        ...

    async def handle_task_stream(self, task: AgentTask) -> AsyncGenerator[dict, None]:
        """Process a task and yield streaming token dicts.

        Default implementation wraps handle_task() in a single yield.
        Override in subclasses that support true token-level streaming.

        Yields:
            dict with {"token": str, "done": bool} for each chunk.
            The last chunk must have done=True and may include
            conversation_id.
        """
        result = await self.handle_task(task)
        chunk = {
            "token": result.get("speech", ""),
            "done": True,
            "conversation_id": task.conversation_id,
        }
        if result.get("action_executed"):
            chunk["action_executed"] = result["action_executed"]
        yield chunk

    def _load_prompt(self, name: str) -> str:
        """Load a prompt file from the prompts/ directory.

        Args:
            name: Filename without extension (e.g. "light" loads "light.txt").

        Returns:
            Prompt text content.
        """
        path = _PROMPTS_DIR / f"{name}.txt"
        return path.read_text(encoding="utf-8").strip()

    async def _call_llm(self, messages: list[dict], **overrides) -> str:
        """Call the LLM using this agent's config.

        Uses the agent_card.agent_id to look up per-agent LLM config
        from the SQLite agent_configs table via llm.complete().
        """
        from app.llm.client import complete

        return await complete(self.agent_card.agent_id, messages, **overrides)
