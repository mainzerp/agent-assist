"""Music agent targeting Music Assistant integration with direct HA execution."""

from __future__ import annotations

import logging
import re

from app.agents.base import BaseAgent
from app.agents.action_executor import parse_action
from app.agents.music_executor import execute_music_action
from app.models.agent import AgentCard, AgentTask

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```json\s*\n?.*?\n?\s*```", re.DOTALL)
_RAW_JSON_OBJ_RE = re.compile(r'\{[^{}]*"action"\s*:.*?\}', re.DOTALL)


def _strip_json_blocks(text: str) -> str:
    """Remove JSON code fences and raw JSON action objects from text."""
    text = _JSON_FENCE_RE.sub("", text)
    text = _RAW_JSON_OBJ_RE.sub("", text)
    return text.strip() or "Sorry, I could not process that request."


class MusicAgent(BaseAgent):
    """Controls music playback via Music Assistant (HA integration).

    Targets Music Assistant media_player entities and MA-specific services
    (mass.play_media, mass.search) for library search, queue management,
    and multi-room audio. Falls back to standard media_player services
    for basic transport controls (play/pause/skip/volume).
    """

    def __init__(self, ha_client=None, entity_index=None, entity_matcher=None) -> None:
        super().__init__(ha_client=ha_client, entity_index=entity_index)
        self._entity_matcher = entity_matcher

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="music-agent",
            name="Music Agent",
            description=(
                "Controls music playback via Music Assistant: play, pause, skip, volume, "
                "library search, queue management, playlist selection, multi-room audio."
            ),
            skills=[
                "music_playback",
                "volume_control",
                "playlist_selection",
                "library_search",
                "queue_management",
            ],
            endpoint="local://music-agent",
        )

    async def handle_task(self, task: AgentTask) -> dict:
        span_collector = getattr(task, "_span_collector", None)
        system_prompt = self._load_prompt("music")
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

        if span_collector:
            async with span_collector.start_span("llm_call", agent_id="music-agent") as span:
                response = await self._call_llm(messages)
                span["metadata"]["model"] = "music-agent"
                span["metadata"]["llm_response"] = response[:500] if response else ""
        else:
            response = await self._call_llm(messages)

        # Guard: empty/None LLM response
        if not response:
            logger.warning("LLM returned empty response for music-agent task: %s", task.description[:100])
            return {
                "speech": "The language model did not return a response. Please try again.",
                "action_executed": None,
            }

        # Parse structured action from LLM response
        action = parse_action(response)

        # Path A: Action parsed AND ha_client available -> execute
        if action and self._ha_client:
            try:
                if span_collector:
                    async with span_collector.start_span("ha_action", agent_id="music-agent") as span:
                        result = await execute_music_action(
                            action,
                            self._ha_client,
                            self._entity_index,
                            self._entity_matcher,
                            agent_id="music-agent",
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
                    result = await execute_music_action(
                        action,
                        self._ha_client,
                        self._entity_index,
                        self._entity_matcher,
                        agent_id="music-agent",
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
            except Exception:
                logger.exception("execute_music_action failed for action=%s", action)
                entity = action.get("entity", "the device")
                return {
                    "speech": f"Sorry, I could not execute the action on {entity}.",
                    "action_executed": None,
                }

        # Path B: Action parsed BUT ha_client is None -> error
        if action and not self._ha_client:
            logger.warning(
                "Action parsed but ha_client is None -- cannot execute: %s",
                action,
            )
            entity = action.get("entity", "the device")
            return {
                "speech": f"I understood the request for {entity}, but the smart home connection is currently unavailable.",
                "action_executed": None,
            }

        # Path C: No action parsed (informational query) -> strip any accidental JSON
        return {"speech": _strip_json_blocks(response), "action_executed": None}
