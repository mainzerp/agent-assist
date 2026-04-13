"""Security system agent with direct HA REST API execution."""

from app.agents.actionable import ActionableAgent
from app.agents.security_executor import execute_security_action
from app.models.agent import AgentCard


class SecurityAgent(ActionableAgent):
    """Controls security devices via HA REST API."""

    _prompt_name = "security"

    async def _do_execute(self, action, ha_client, entity_index, entity_matcher, *, agent_id):
        return await execute_security_action(action, ha_client, entity_index, entity_matcher, agent_id=agent_id)

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="security-agent",
            name="Security Agent",
            description="Controls locks, alarm panels, and camera status. Reads security sensors (motion, door, window, smoke, gas, CO).",
            skills=["lock_control", "alarm_control", "camera_status", "sensor_reading"],
            endpoint="local://security-agent",
        )
