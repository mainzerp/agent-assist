"""General fallback agent for unroutable requests."""

from __future__ import annotations

import logging

from app.agents.base import BaseAgent
from app.models.agent import AgentCard, AgentTask, TaskResult

logger = logging.getLogger(__name__)


class GeneralAgent(BaseAgent):
    """Handles general Q&A and unroutable requests. No HA service calls."""

    def __init__(self, ha_client=None, entity_index=None, mcp_tool_manager=None):
        super().__init__(ha_client=ha_client, entity_index=entity_index)
        self._mcp_tool_manager = mcp_tool_manager

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="general-agent",
            name="General Agent",
            description="Handles general knowledge, conversation, web search, current events, and requests outside device control. Can search the web for real-time information. Fallback for unroutable requests.",
            skills=["general_qa", "web_search", "current_events", "conversation", "fallback"],
            endpoint="local://general-agent",
        )

    async def handle_task(self, task: AgentTask) -> TaskResult:
        system_prompt = self._load_prompt("general")
        messages = [{"role": "system", "content": system_prompt}]

        if task.context and task.context.conversation_turns:
            for turn in task.context.conversation_turns:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })

        # task.description = condensed task from orchestrator (primary input)
        # task.user_text = original unmodified user text (fallback only)
        messages.append({"role": "user", "content": task.description})

        # Check for available MCP tools
        tools = await self._get_mcp_tools()
        if tools:
            tool_schemas = self._mcp_tools_to_openai_format(tools)
            response = await self._call_llm_with_tools(messages, tool_schemas, tools)
        else:
            response = await self._call_llm(messages)

        return TaskResult(speech=response)

    async def _get_mcp_tools(self) -> list[dict]:
        """Get MCP tools assigned to this agent."""
        if not self._mcp_tool_manager:
            return []
        try:
            return await self._mcp_tool_manager.get_tools_for_agent(self.agent_card.agent_id)
        except Exception:
            logger.warning("Failed to get MCP tools for general-agent", exc_info=True)
            return []

    @staticmethod
    def _mcp_tools_to_openai_format(mcp_tools: list[dict]) -> list[dict]:
        """Convert MCP tool descriptors to OpenAI function-calling format."""
        openai_tools = []
        for tool in mcp_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        return openai_tools

    async def _call_llm_with_tools(self, messages, tool_schemas, mcp_tools):
        """Call LLM with tool calling support."""
        from app.llm.client import complete_with_tools

        # Build a tool name -> tool info mapping for execution
        tool_map = {}
        for tool in mcp_tools:
            tool_map[tool["name"]] = tool

        async def execute_tool(name: str, arguments: dict) -> str:
            tool_info = tool_map.get(name)
            if not tool_info:
                return f"Error: unknown tool '{name}'"
            server_name = tool_info.get("_server_name", "")
            try:
                result = await self._mcp_tool_manager.call_tool(server_name, name, arguments)
                # MCP call_tool returns a CallToolResult; extract text content
                if hasattr(result, "content"):
                    texts = [c.text for c in result.content if hasattr(c, "text")]
                    return "\n".join(texts) if texts else str(result)
                return str(result)
            except Exception as e:
                logger.warning("MCP tool '%s' failed: %s", name, e)
                return f"Tool error: {e}"

        return await complete_with_tools(
            self.agent_card.agent_id,
            messages,
            tools=tool_schemas,
            tool_executor=execute_tool,
        )
