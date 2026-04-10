"""Agent configuration models."""

from __future__ import annotations

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


class AgentConfig(BaseModel):
    """Agent runtime configuration loaded from SQLite."""

    agent_id: str
    enabled: bool = True
    model: str | None = None
    timeout: int = 5
    max_iterations: int = 3
    temperature: float = 0.7
    max_tokens: int = 256
    description: str | None = None


class AgentTask(BaseModel):
    """Task dispatched from orchestrator to a specialized agent via A2A."""

    description: str = Field(..., description="Condensed task with preserved entity names")
    user_text: str = Field(..., description="Original unmodified user input")
    conversation_id: str | None = None
    context: TaskContext | None = None


class TaskContext(BaseModel):
    """Context propagated with an agent task."""

    conversation_turns: list[dict] = Field(default_factory=list)
    presence_room: str | None = None
    entity_states: dict | None = None
    mcp_tools: list[str] = Field(default_factory=list)
