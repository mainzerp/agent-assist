"""Timer and alarm agent with direct HA REST API execution."""

from app.agents.actionable import ActionableAgent
from app.agents.timer_executor import execute_timer_action
from app.models.agent import AgentCard


class TimerAgent(ActionableAgent):
    """Controls timers and reminders via HA REST API."""

    _prompt_name = "timer"

    async def _do_execute(self, action, ha_client, entity_index, entity_matcher, *, agent_id, span_collector=None):
        # FLOW-CTX-1 (0.18.6): ``_current_task_context`` is now set
        # by ``ActionableAgent.handle_task`` for every subclass, so
        # we no longer need an override just to capture it here.
        ctx = getattr(self, "_current_task_context", None)
        device_id = ctx.device_id if ctx else None
        area_id = ctx.area_id if ctx else None
        return await execute_timer_action(
            action,
            ha_client,
            entity_index,
            entity_matcher,
            agent_id=agent_id,
            device_id=device_id,
            area_id=area_id,
            span_collector=span_collector,
        )

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="timer-agent",
            name="Timer Agent",
            description="Manages timers, alarms, reminders, and scheduled actions. Start, cancel, pause, resume, snooze timers. Sets alarms, schedules delayed actions and sleep timers, creates calendar reminders. Reports timer status and remaining time.",
            skills=[
                "timer_set",
                "timer_cancel",
                "timer_pause",
                "timer_resume",
                "timer_snooze",
                "timer_query",
                "alarm",
                "reminder",
                "delayed_action",
                "sleep_timer",
                "calendar",
            ],
            endpoint="local://timer-agent",
        )
