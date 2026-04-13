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
            description="Controls and queries lights, switches, and illuminance sensors: on/off, toggle, brightness, color, color temperature. Reports light/switch status and light-level readings. Lists all lights and switches.",
            skills=["light_control", "switch_control", "brightness", "color", "toggle", "illuminance_sensor", "light_status", "light_query", "switch_status", "switch_query"],
            endpoint="local://light-agent",
        )
