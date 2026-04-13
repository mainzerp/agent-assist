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
            description="Controls and queries climate/HVAC devices and environmental sensors. Set temperature, HVAC mode, fan speed, humidity, turn on/off. Reads sensors: temperature, humidity, pressure, dew point, wind, precipitation.",
            skills=["temperature", "hvac_mode", "fan_speed", "humidity", "climate_on_off", "sensor_reading", "climate_status", "sensor_query", "weather_sensor"],
            endpoint="local://climate-agent",
        )
