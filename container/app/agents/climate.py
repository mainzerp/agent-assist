"""Climate and HVAC control agent with direct HA REST API execution."""

from app.agents.actionable import ActionableAgent
from app.agents.climate_executor import execute_climate_action
from app.models.agent import AgentCard


class ClimateAgent(ActionableAgent):
    """Controls climate and HVAC devices via HA REST API."""

    _prompt_name = "climate"

    async def _do_execute(self, action, ha_client, entity_index, entity_matcher, *, agent_id):
        return await execute_climate_action(action, ha_client, entity_index, entity_matcher, agent_id=agent_id)

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="climate-agent",
            name="Climate Agent",
            description="Controls climate and HVAC: temperature, mode, fan speed, humidity. Reads climate sensor data (temperature, humidity, pressure, dew point).",
            skills=["temperature", "hvac_mode", "fan_speed", "humidity", "sensor_reading"],
            endpoint="local://climate-agent",
        )
