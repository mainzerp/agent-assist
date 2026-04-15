"""Security system agent with direct HA REST API execution."""

from app.agents.actionable import ActionableAgent
from app.agents.security_executor import execute_security_action
from app.models.agent import AgentCard


class SecurityAgent(ActionableAgent):
    """Controls security devices via HA REST API."""

    _prompt_name = "security"

    async def _do_execute(self, action, ha_client, entity_index, entity_matcher, *, agent_id, span_collector=None):
        return await execute_security_action(action, ha_client, entity_index, entity_matcher, agent_id=agent_id, span_collector=span_collector)

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="security-agent",
            name="Security Agent",
            description="Controls and queries locks, alarm panels, cameras, and security sensors (motion, door, window, doorbell, smoke, gas). Lock/unlock, arm/disarm, camera on/off. Reports status and lists all security devices.",
            skills=["lock_control", "alarm_control", "camera_control", "door_sensor", "window_sensor", "motion_sensor", "doorbell", "smoke_sensor", "security_status", "security_query"],
            endpoint="local://security-agent",
        )
