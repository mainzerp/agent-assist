"""Automation management agent with direct HA REST API execution."""

from app.agents.actionable import ActionableAgent
from app.agents.automation_executor import execute_automation_action
from app.models.agent import AgentCard


class AutomationAgent(ActionableAgent):
    """Manages Home Assistant automations via HA REST API."""

    _prompt_name = "automation"

    async def _do_execute(self, action, ha_client, entity_index, entity_matcher, *, agent_id):
        return await execute_automation_action(action, ha_client, entity_index, entity_matcher, agent_id=agent_id)

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="automation-agent",
            name="Automation Agent",
            description="Enables, disables, and triggers automations.",
            skills=["automation_enable", "automation_disable", "automation_trigger"],
            endpoint="local://automation-agent",
        )
