"""Music agent targeting Music Assistant integration with direct HA execution."""

from app.agents.actionable import ActionableAgent
from app.agents.music_executor import execute_music_action
from app.models.agent import AgentCard


class MusicAgent(ActionableAgent):
    """Controls music playback via Music Assistant (HA integration).

    Targets Music Assistant media_player entities and MA-specific services
    (mass.play_media, mass.search) for library search, queue management,
    and multi-room audio. Falls back to standard media_player services
    for basic transport controls (play/pause/skip/volume).
    """

    _prompt_name = "music"

    async def _do_execute(self, action, ha_client, entity_index, entity_matcher, *, agent_id):
        return await execute_music_action(action, ha_client, entity_index, entity_matcher, agent_id=agent_id)

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
