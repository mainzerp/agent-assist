"""Base agent class with HA client and entity index access."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from pathlib import Path

from app.models.agent import AgentCard, AgentError, AgentErrorCode, AgentTask, TaskContext, TaskResult

logger = logging.getLogger(__name__)

# Prompts directory (container/app/prompts/)
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Module-level cache for loaded prompt files (Q-4). Prompts only change on
# container restart, so there is no invalidation path.
_prompt_cache: dict[str, str] = {}


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
    async def handle_task(self, task: AgentTask) -> dict | TaskResult:
        """Process a task and return the full result.

        Returns:
            TaskResult (preferred) or dict with at least {"speech": str}.
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
        # Support both TaskResult and raw dict
        result_dict = result.model_dump() if hasattr(result, "model_dump") else result
        chunk = {
            "token": result_dict.get("speech", ""),
            "done": True,
            "conversation_id": task.conversation_id,
        }
        action = result_dict.get("action_executed")
        if action:
            chunk["action_executed"] = action
        if result_dict.get("voice_followup"):
            chunk["voice_followup"] = True
        yield chunk

    def _load_prompt(self, name: str) -> str:
        """Load a prompt file from the prompts/ directory.

        Results are cached in ``_prompt_cache`` keyed by the resolved file
        path (Q-4). Prompts only change on container restart.

        Args:
            name: Filename without extension (e.g. "light" loads "light.txt").

        Returns:
            Prompt text content.
        """
        path = _PROMPTS_DIR / f"{name}.txt"
        cache_key = str(path)
        cached = _prompt_cache.get(cache_key)
        if cached is not None:
            return cached
        content = path.read_text(encoding="utf-8").strip()
        # Resolve {personality_base} include if present
        if "{personality_base}" in content:
            base_path = _PROMPTS_DIR / "personality_base.txt"
            if base_path.exists():
                base_content = base_path.read_text(encoding="utf-8").strip()
                content = content.replace("{personality_base}", base_content)
        _prompt_cache[cache_key] = content
        return content

    def _error_result(
        self,
        code: AgentErrorCode,
        speech: str,
        *,
        recoverable: bool = True,
    ) -> TaskResult:
        """Build a TaskResult with a structured error."""
        return TaskResult(
            speech=speech,
            error=AgentError(code=code, message=speech, recoverable=recoverable),
        )

    @staticmethod
    def _build_time_location_context(context: TaskContext | None) -> str:
        """Build a short context block for local time and location."""
        if not context or not context.local_time:
            return ""
        parts = [f"Current local time: {context.local_time}"]
        if context.timezone and context.timezone != "UTC":
            parts.append(f"Timezone: {context.timezone}")
        if context.location_name:
            parts.append(f"Home location: {context.location_name}")
        return "\n".join(parts)

    async def _call_llm(self, messages: list[dict], **overrides) -> str:
        """Call the LLM using this agent's config.

        Uses the agent_card.agent_id to look up per-agent LLM config
        from the SQLite agent_configs table via llm.complete().
        """
        from app.llm.client import complete

        return await complete(self.agent_card.agent_id, messages, **overrides)
