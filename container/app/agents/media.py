"""Media player control agent with direct HA REST API execution."""

from app.agents.actionable import ActionableAgent
from app.agents.media_executor import execute_media_action
from app.models.agent import AgentCard


class MediaAgent(ActionableAgent):
    """Controls generic media player devices via HA REST API."""

    _prompt_name = "media"

    async def _do_execute(self, action, ha_client, entity_index, entity_matcher, *, agent_id):
        return await execute_media_action(action, ha_client, entity_index, entity_matcher, agent_id=agent_id)

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="media-agent",
            name="Media Agent",
            description="Controls generic media players: TV, speakers, casting, playback.",
            skills=["tv_control", "speaker_control", "casting", "playback", "volume_control", "source_selection"],
            endpoint="local://media-agent",
        )
