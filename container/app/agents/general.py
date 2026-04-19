"""General fallback agent for unroutable requests."""

from __future__ import annotations

import logging

from app.agents.base import BaseAgent
from app.analytics.tracer import _optional_span
from app.models.agent import AgentErrorCode
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
            expected_latency="high",
        )

    async def handle_task(self, task: AgentTask) -> TaskResult:
        span_collector = task.span_collector
        system_prompt = self._load_prompt("general")

        language = task.context.language if task.context else None

        # Inject time/location context
        time_location = self._build_time_location_context(task.context)
        if time_location:
            system_prompt += f"\n\n{time_location}"

        # Inject language directive for non-English users
        if language and language.lower() not in ("en", "english", ""):
            system_prompt += f"\n\nIMPORTANT: Respond in {language}. The user's language is {language}. Keep entity names, device names, and room names exactly as the user wrote them -- do NOT translate those."

        if task.context and task.context.sequential_send:
            system_prompt += (
                "\n\nThis response will be delivered as text to a device (not spoken aloud). "
                "You MAY include URLs and links if relevant. "
                "Format for readability -- you can use line breaks."
            )

        messages = [{"role": "system", "content": system_prompt}]

        if task.context and task.context.conversation_turns:
            for turn in task.context.conversation_turns:
                messages.append(
                    {
                        "role": turn.get("role", "user"),
                        "content": turn.get("content", ""),
                    }
                )

        # task.description = condensed task from orchestrator (primary input)
        # task.user_text = original unmodified user text (fallback only)
        user_content = task.description
        if task.user_text and task.user_text != task.description:
            user_content = f'{task.description}\n\n(Original user message: "{task.user_text}")'
        messages.append({"role": "user", "content": user_content})

        # Check for available MCP tools
        llm_kwargs = {}
        if task.context and task.context.sequential_send:
            llm_kwargs["max_tokens"] = 2048
        tools = await self._get_mcp_tools()
        if tools:
            tool_schemas = self._mcp_tools_to_openai_format(tools)
            async with _optional_span(span_collector, "llm_call", agent_id="general-agent") as span:
                response = await self._call_llm_with_tools(
                    messages, tool_schemas, tools, span_collector=span_collector, **llm_kwargs
                )
                span["metadata"]["model"] = "general-agent"
                span["metadata"]["llm_response"] = response[:500] if response else ""
                span["metadata"]["tools_available"] = len(tool_schemas)
        else:
            async with _optional_span(span_collector, "llm_call", agent_id="general-agent") as span:
                response = await self._call_llm(messages, span_collector=span_collector, **llm_kwargs)
                span["metadata"]["model"] = "general-agent"
                span["metadata"]["llm_response"] = response[:500] if response else ""

        if not response or not response.strip():
            logger.warning("LLM returned empty response for general-agent task: %s", task.description[:100])
            return self._error_result(
                AgentErrorCode.LLM_EMPTY_RESPONSE,
                "The language model did not return a response. Please try again.",
            )

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
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                }
            )
        return openai_tools

    async def _call_llm_with_tools(self, messages, tool_schemas, mcp_tools, span_collector=None, **overrides):
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

        _inner_executor = execute_tool

        async def _traced_executor(name: str, arguments: dict) -> str:
            async with _optional_span(span_collector, "mcp_tool_call", agent_id="general-agent") as tool_span:
                tool_span["metadata"]["tool_name"] = name
                tool_span["metadata"]["arguments"] = str(arguments)[:300]
                result = await _inner_executor(name, arguments)
                tool_span["metadata"]["result"] = result[:500] if result else ""
                return result

        return await complete_with_tools(
            self.agent_card.agent_id,
            messages,
            tools=tool_schemas,
            tool_executor=_traced_executor,
            span_collector=span_collector,
            **overrides,
        )
