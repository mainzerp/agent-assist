"""Base class for agents that parse LLM output into HA actions."""

from __future__ import annotations

import logging
import re

from app.agents.base import BaseAgent
from app.agents.action_executor import parse_action
from app.models.agent import AgentTask, TaskResult, ActionExecuted, AgentErrorCode

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```json\s*\n?.*?\n?\s*```", re.DOTALL)
_RAW_JSON_OBJ_RE = re.compile(r'\{[^{}]*"action"\s*:.*?\}', re.DOTALL)


def strip_json_blocks(text: str) -> str:
    """Remove JSON code fences and raw JSON action objects from text."""
    text = _JSON_FENCE_RE.sub("", text)
    text = _RAW_JSON_OBJ_RE.sub("", text)
    return text.strip() or "Sorry, I could not process that request."


class ActionableAgent(BaseAgent):
    """Base for domain agents that parse actions from LLM output and execute via HA.

    Subclasses must define:
        - agent_card (property)
        - _prompt_name (str): name of the prompt file (e.g., "light")
        - _do_execute(): async method that delegates to the domain-specific executor
    """

    _prompt_name: str = ""

    def __init__(self, ha_client=None, entity_index=None, entity_matcher=None) -> None:
        super().__init__(ha_client=ha_client, entity_index=entity_index)
        self._entity_matcher = entity_matcher

    async def _do_execute(self, action, ha_client, entity_index, entity_matcher, *, agent_id):
        """Execute the parsed action. Subclasses must override."""
        raise NotImplementedError

    async def handle_task(self, task: AgentTask) -> TaskResult:
        agent_id = self.agent_card.agent_id
        span_collector = task.span_collector
        system_prompt = self._load_prompt(self._prompt_name)
        messages = [{"role": "system", "content": system_prompt}]

        if task.context and task.context.conversation_turns:
            for turn in task.context.conversation_turns:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })

        messages.append({"role": "user", "content": task.description})

        if span_collector:
            async with span_collector.start_span("llm_call", agent_id=agent_id) as span:
                response = await self._call_llm(messages)
                span["metadata"]["model"] = agent_id
                span["metadata"]["llm_response"] = response[:500] if response else ""
        else:
            response = await self._call_llm(messages)

        if not response:
            logger.warning("LLM returned empty response for %s task: %s", agent_id, task.description[:100])
            return self._error_result(
                AgentErrorCode.LLM_EMPTY_RESPONSE,
                "The language model did not return a response. Please try again.",
            )

        action = parse_action(response)

        # Path A: Action + HA client -> execute
        if action and self._ha_client:
            try:
                if span_collector:
                    async with span_collector.start_span("ha_action", agent_id=agent_id) as span:
                        result = await self._do_execute(
                            action, self._ha_client, self._entity_index,
                            self._entity_matcher, agent_id=agent_id,
                        )
                        span["metadata"]["action"] = action.get("action")
                        span["metadata"]["entity"] = action.get("entity")
                        span["metadata"]["success"] = result.get("success")
                        span["metadata"]["action_params"] = {
                            k: v for k, v in action.items()
                            if k not in ("action", "entity")
                        }
                        span["metadata"]["result_speech"] = (result.get("speech") or "")[:500]
                else:
                    result = await self._do_execute(
                        action, self._ha_client, self._entity_index,
                        self._entity_matcher, agent_id=agent_id,
                    )
                return TaskResult(
                    speech=result["speech"],
                    action_executed=ActionExecuted(
                        action=action.get("action", ""),
                        entity_id=result.get("entity_id", ""),
                        success=result.get("success", False),
                        new_state=result.get("new_state"),
                    ),
                )
            except Exception:
                logger.exception("Action execution failed for %s action=%s", agent_id, action)
                entity = action.get("entity", "the device")
                return self._error_result(
                    AgentErrorCode.ACTION_FAILED,
                    f"Sorry, I could not execute the action on {entity}.",
                )

        # Path B: Action but no HA client
        if action and not self._ha_client:
            logger.warning("Action parsed but ha_client is None for %s: %s", agent_id, action)
            entity = action.get("entity", "the device")
            return self._error_result(
                AgentErrorCode.HA_UNAVAILABLE,
                f"I understood the request for {entity}, but the smart home connection is currently unavailable.",
                recoverable=False,
            )

        # Path C: No action (informational)
        return TaskResult(speech=strip_json_blocks(response))
