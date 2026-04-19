"""Climate and HVAC control agent with direct HA REST API execution."""

from app.agents.actionable import ActionableAgent
from app.agents.climate_executor import execute_climate_action
from app.models.agent import AgentCard


class ClimateAgent(ActionableAgent):
    """Controls climate and HVAC devices via HA REST API."""

    _prompt_name = "climate"

    async def _do_execute(self, action, ha_client, entity_index, entity_matcher, *, agent_id, span_collector=None):
        # FLOW-CTX-1 (0.18.6): use the originating satellite's area
        # as a tie-breaker for ambiguous thermostat/sensor queries.
        ctx = getattr(self, "_current_task_context", None)
        area_id = ctx.area_id if ctx else None
        return await execute_climate_action(
            action,
            ha_client,
            entity_index,
            entity_matcher,
            agent_id=agent_id,
            span_collector=span_collector,
            preferred_area_id=area_id,
        )

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="climate-agent",
            name="Climate Agent",
            description="Controls and queries climate/HVAC devices, environmental sensors, and local weather conditions/forecasts. Set temperature, HVAC mode, fan speed, humidity, turn on/off. Reads sensors: temperature, humidity, pressure, dew point, wind, precipitation. Queries weather entities for current conditions and forecasts.",
            skills=[
                "temperature",
                "hvac_mode",
                "fan_speed",
                "humidity",
                "climate_on_off",
                "sensor_reading",
                "climate_status",
                "sensor_query",
                "weather_sensor",
                "current_weather",
                "weather_forecast",
            ],
            endpoint="local://climate-agent",
        )
