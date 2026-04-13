"""Timer and alarm agent with direct HA REST API execution."""

from app.agents.actionable import ActionableAgent
from app.agents.timer_executor import execute_timer_action
from app.models.agent import AgentCard


class TimerAgent(ActionableAgent):
    """Controls timers and reminders via HA REST API."""

    _prompt_name = "timer"

    async def _do_execute(self, action, ha_client, entity_index, entity_matcher, *, agent_id):
        return await execute_timer_action(action, ha_client, entity_index, entity_matcher, agent_id=agent_id)

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="timer-agent",
            name="Timer Agent",
            description="Manages timers, alarms, reminders, sleep timers, delayed actions, and calendar events.",
            skills=["timer_set", "timer_cancel", "timer_query", "timer_snooze",
                    "reminder", "delayed_action", "sleep_timer", "calendar"],
            endpoint="local://timer-agent",
        )
