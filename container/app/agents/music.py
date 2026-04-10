"""Music agent targeting Music Assistant integration."""

from __future__ import annotations

import logging

from app.agents.base import BaseAgent
from app.models.agent import AgentCard, AgentTask

logger = logging.getLogger(__name__)


class MusicAgent(BaseAgent):
    """Controls music playback via Music Assistant (HA integration).

    Targets Music Assistant media_player entities and MA-specific services
    (mass.play_media, mass.search) for library search, queue management,
    and multi-room audio. Falls back to standard media_player services
    for basic transport controls (play/pause/skip/volume).
    """

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
        # task.description = condensed task from orchestrator (primary input)
        # task.user_text = original unmodified user text (fallback only)
        system_prompt = self._load_prompt("music")
        messages = [{"role": "system", "content": system_prompt}]

        if task.context and task.context.conversation_turns:
            for turn in task.context.conversation_turns:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })

        messages.append({"role": "user", "content": task.description})

        response = await self._call_llm(messages)

        # Phase 1.9: parse response for structured action, resolve entity,
        # execute via ha_client (mass.play_media, mass.search, etc.),
        # verify result. For now, return LLM response.
        return {"speech": response, "action_executed": None}
