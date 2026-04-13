"""Scene activation agent with direct HA REST API execution."""

from app.agents.actionable import ActionableAgent
from app.agents.scene_executor import execute_scene_action
from app.models.agent import AgentCard


class SceneAgent(ActionableAgent):
    """Activates and manages scenes via HA REST API."""

    _prompt_name = "scene"

    async def _do_execute(self, action, ha_client, entity_index, entity_matcher, *, agent_id):
        return await execute_scene_action(action, ha_client, entity_index, entity_matcher, agent_id=agent_id)

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="scene-agent",
            name="Scene Agent",
            description="Activates and manages Home Assistant scenes.",
            skills=["scene_activate", "scene_list"],
            endpoint="local://scene-agent",
        )
