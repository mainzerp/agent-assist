"""Agent configuration models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentCard(BaseModel):
    """A2A agent card describing agent capabilities."""

    agent_id: str
    name: str
    description: str
    skills: list[str] = Field(default_factory=list)
    input_types: list[str] = Field(default_factory=lambda: ["text/plain"])
    output_types: list[str] = Field(default_factory=lambda: ["text/plain", "application/json"])
    endpoint: str = Field("", description="Agent endpoint URL (local:// for in-process)")
    expected_latency: str = Field("low", description="Expected response latency: low, medium, high")


class AgentConfig(BaseModel):
    """Agent runtime configuration loaded from SQLite."""

    agent_id: str
    enabled: bool = True
    model: str | None = None
    timeout: int = 5
    max_iterations: int = 3
    temperature: float = 0.2
    max_tokens: int = 1024
    description: str | None = None
    reasoning_effort: str | None = None


class AgentTask(BaseModel):
    """Task dispatched from orchestrator to a specialized agent via A2A."""

    model_config = {"arbitrary_types_allowed": True}

    description: str = Field(..., description="Condensed task with preserved entity names")
    user_text: str = Field(..., description="Original unmodified user input")
    conversation_id: str | None = None
    context: TaskContext | None = None

    # Runtime-only: not serialized, not included in model_dump()
    span_collector: Any = Field(default=None, exclude=True)


class TaskContext(BaseModel):
    """Context propagated with an agent task."""

    conversation_turns: list[dict] = Field(default_factory=list)
    presence_room: str | None = None
    entity_states: dict | None = None
    mcp_tools: list[str] = Field(default_factory=list)
    device_id: str | None = None
    area_id: str | None = None
    language: str = "en"
    sequential_send: bool = False


class ActionExecuted(BaseModel):
    """Result of a Home Assistant action execution."""

    action: str = Field(..., description="HA action name (e.g. turn_on, turn_off)")
    entity_id: str = Field(..., description="Target entity ID (e.g. light.kitchen)")
    success: bool = Field(True, description="Whether the action succeeded")
    new_state: str | None = Field(None, description="Entity state after action")
    cacheable: bool = Field(True, description="Whether response may be stored in the response cache")


class AgentErrorCode(str, Enum):
    """Structured error codes for agent failures."""

    ENTITY_NOT_FOUND = "entity_not_found"
    ACTION_FAILED = "action_failed"
    HA_UNAVAILABLE = "ha_unavailable"
    LLM_ERROR = "llm_error"
    LLM_EMPTY_RESPONSE = "llm_empty_response"
    TIMEOUT = "timeout"
    PARSE_ERROR = "parse_error"
    AGENT_NOT_FOUND = "agent_not_found"
    INTERNAL = "internal"


class AgentError(BaseModel):
    """Structured error returned by an agent."""

    code: AgentErrorCode
    message: str
    recoverable: bool = True


class TaskResult(BaseModel):
    """Standardized result returned by all agents from handle_task().

    Backward compatible: .model_dump() produces the same dict shape
    that agents previously returned manually.
    """

    speech: str = Field(..., description="Natural language response text")
    action_executed: ActionExecuted | None = Field(None, description="HA action result if an action was performed")
    metadata: dict = Field(default_factory=dict, description="Agent-specific metadata")
    error: AgentError | None = Field(None, description="Structured error if the agent encountered a problem")
