"""Dataclasses describing a real-scenario test definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExpectedCall:
    domain: str
    service: str
    target_entity: str | None = None
    service_data_keys: list[str] = field(default_factory=list)
    service_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpectedError:
    code: str


@dataclass
class Expected:
    routed_agent: str | None = None
    service_calls: list[ExpectedCall] = field(default_factory=list)
    speech_contains: list[str] = field(default_factory=list)
    speech_excludes: list[str] = field(default_factory=list)
    action_executed: dict[str, Any] | None = None
    error: ExpectedError | None = None
    allow_extra_calls: bool = False


@dataclass
class EntityOverride:
    entity_id: str
    state: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Preconditions:
    entity_overrides: list[EntityOverride] = field(default_factory=list)
    settings: dict[str, str] = field(default_factory=dict)
    send_device_mappings: list[dict[str, Any]] = field(default_factory=list)
    frozen_time: str | None = None


@dataclass
class LlmReplies:
    """Replies to feed the deterministic LLM stub.

    classify: orchestrator classification reply (single string).
    agents: per agent_id list of replies; consumed FIFO per agent_id.
    """

    classify: str | None = None
    agents: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ScenarioContext:
    source: str = "ha"
    area_id: str | None = None
    area_name: str | None = None
    device_id: str | None = None
    device_name: str | None = None
    conversation_id: str | None = None
    user_id: str | None = None


@dataclass
class FollowUpTurn:
    text: str
    llm: LlmReplies = field(default_factory=LlmReplies)
    expected: Expected = field(default_factory=Expected)


@dataclass
class Scenario:
    """Top-level scenario record loaded from YAML."""

    id: str
    agent: str
    description: str
    snapshot: str
    language: str
    request_text: str
    context: ScenarioContext
    preconditions: Preconditions
    llm: LlmReplies
    expected: Expected
    follow_up: list[FollowUpTurn] = field(default_factory=list)
    path: str = ""
    xfail: str | None = None
