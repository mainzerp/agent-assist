"""Timer and alarm agent with direct HA REST API execution."""

from app.agents.actionable import ActionableAgent
from app.agents.timer_executor import execute_timer_action
from app.models.agent import AgentCard, AgentTask, TaskResult


class TimerAgent(ActionableAgent):
    """Controls timers and reminders via HA REST API."""

    _prompt_name = "timer"

    async def _do_execute(self, action, ha_client, entity_index, entity_matcher, *, agent_id, span_collector=None):
        device_id = None
        area_id = None
        if hasattr(self, "_current_task_context") and self._current_task_context:
            device_id = self._current_task_context.device_id
            area_id = self._current_task_context.area_id
        return await execute_timer_action(
            action, ha_client, entity_index, entity_matcher,
            agent_id=agent_id, device_id=device_id, area_id=area_id,
            span_collector=span_collector,
        )

    async def handle_task(self, task: AgentTask) -> TaskResult:
        self._current_task_context = task.context
        try:
            return await super().handle_task(task)
        finally:
            self._current_task_context = None

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="timer-agent",
            name="Timer Agent",
            description="Manages timers, alarms, reminders, and scheduled actions. Start, cancel, pause, resume, snooze timers. Sets alarms, schedules delayed actions and sleep timers, creates calendar reminders. Reports timer status and remaining time.",
            skills=["timer_set", "timer_cancel", "timer_pause", "timer_resume", "timer_snooze",
                    "timer_query", "alarm", "reminder", "delayed_action", "sleep_timer", "calendar"],
            endpoint="local://timer-agent",
        )
