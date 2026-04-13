"""Light control agent with direct HA REST API execution."""

from app.agents.actionable import ActionableAgent
from app.agents.action_executor import execute_action
from app.models.agent import AgentCard


class LightAgent(ActionableAgent):
    """Controls lighting devices via HA REST API."""

    _prompt_name = "light"

    async def _do_execute(self, action, ha_client, entity_index, entity_matcher, *, agent_id):
        return await execute_action(action, ha_client, entity_index, entity_matcher, agent_id=agent_id)

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="light-agent",
            name="Light Agent",
            description="Controls lighting devices: on/off, brightness, color, color temperature.",
            skills=["light_control", "brightness", "color"],
            endpoint="local://light-agent",
        )
