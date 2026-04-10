"""Test factory functions and utilities for agent-assist tests."""

from __future__ import annotations

import random
import uuid
from typing import Any
from unittest.mock import MagicMock

from app.models.conversation import ActionResult, ConversationRequest, ConversationResponse, StreamToken
from app.models.agent import AgentCard, AgentConfig, AgentTask, TaskContext
from app.models.cache import CachedAction, ResponseCacheEntry, RoutingCacheEntry
from app.models.entity_index import EntityIndexEntry


# ---------------------------------------------------------------------------
# Conversation factories
# ---------------------------------------------------------------------------

def make_conversation_request(
    text: str = "turn on the kitchen light",
    conversation_id: str | None = None,
    language: str = "en",
) -> ConversationRequest:
    """Build a ConversationRequest with sensible defaults."""
    return ConversationRequest(
        text=text,
        conversation_id=conversation_id,
        language=language,
    )


def make_conversation_response(
    speech: str = "Done, kitchen light is on.",
    conversation_id: str | None = None,
    action_executed: ActionResult | None = None,
) -> ConversationResponse:
    """Build a ConversationResponse with sensible defaults."""
    return ConversationResponse(
        speech=speech,
        conversation_id=conversation_id,
        action_executed=action_executed,
    )


def make_action_result(
    service: str = "light/turn_on",
    entity_id: str = "light.kitchen_ceiling",
    result: str = "success",
    service_data: dict | None = None,
) -> ActionResult:
    """Build an ActionResult with sensible defaults."""
    return ActionResult(
        service=service,
        entity_id=entity_id,
        result=result,
        service_data=service_data,
    )


def make_stream_token(
    token: str = "Hello",
    done: bool = False,
    conversation_id: str | None = None,
) -> StreamToken:
    """Build a StreamToken."""
    return StreamToken(token=token, done=done, conversation_id=conversation_id)


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

def make_agent_card(
    agent_id: str = "light-agent",
    name: str = "Light Agent",
    description: str = "Controls lighting devices",
    skills: list[str] | None = None,
    endpoint: str = "local://light-agent",
) -> AgentCard:
    """Build an AgentCard."""
    return AgentCard(
        agent_id=agent_id,
        name=name,
        description=description,
        skills=skills or ["light_control"],
        endpoint=endpoint,
    )


def make_agent_config(
    agent_id: str = "light-agent",
    enabled: bool = True,
    model: str | None = "openrouter/openai/gpt-4o-mini",
    timeout: int = 5,
    max_iterations: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 256,
    description: str | None = "Lighting control",
) -> AgentConfig:
    """Build an AgentConfig."""
    return AgentConfig(
        agent_id=agent_id,
        enabled=enabled,
        model=model,
        timeout=timeout,
        max_iterations=max_iterations,
        temperature=temperature,
        max_tokens=max_tokens,
        description=description,
    )


def make_agent_task(
    description: str = "Turn on the kitchen light",
    user_text: str = "turn on the kitchen light",
    conversation_id: str | None = None,
    context: TaskContext | None = None,
) -> AgentTask:
    """Build an AgentTask."""
    return AgentTask(
        description=description,
        user_text=user_text,
        conversation_id=conversation_id,
        context=context,
    )


# ---------------------------------------------------------------------------
# Entity factories
# ---------------------------------------------------------------------------

def make_entity_state(
    entity_id: str = "light.kitchen_ceiling",
    friendly_name: str = "Kitchen Ceiling",
    domain: str | None = None,
    state: str = "off",
    area: str | None = "kitchen",
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an HA entity state dict as returned by GET /api/states."""
    if domain is None:
        domain = entity_id.split(".")[0] if "." in entity_id else ""
    attrs: dict[str, Any] = {
        "friendly_name": friendly_name,
    }
    if area is not None:
        attrs["area_id"] = area
    if attributes:
        attrs.update(attributes)
    return {
        "entity_id": entity_id,
        "state": state,
        "attributes": attrs,
    }


def make_entity_index_entry(
    entity_id: str = "light.kitchen_ceiling",
    friendly_name: str = "Kitchen Ceiling",
    domain: str | None = None,
    area: str | None = "kitchen",
    device_class: str | None = None,
    aliases: list[str] | None = None,
) -> EntityIndexEntry:
    """Build an EntityIndexEntry."""
    if domain is None:
        domain = entity_id.split(".")[0] if "." in entity_id else ""
    return EntityIndexEntry(
        entity_id=entity_id,
        friendly_name=friendly_name,
        domain=domain,
        area=area,
        device_class=device_class,
        aliases=aliases or [],
    )


# ---------------------------------------------------------------------------
# Cache factories
# ---------------------------------------------------------------------------

def make_routing_cache_entry(
    query_text: str = "turn on kitchen lights",
    agent_id: str = "light-agent",
    confidence: float = 0.95,
    hit_count: int = 1,
) -> RoutingCacheEntry:
    """Build a RoutingCacheEntry."""
    return RoutingCacheEntry(
        query_text=query_text,
        agent_id=agent_id,
        confidence=confidence,
        hit_count=hit_count,
    )


def make_response_cache_entry(
    query_text: str = "turn on kitchen lights",
    response_text: str = "Done, kitchen light is on.",
    agent_id: str = "light-agent",
    confidence: float = 0.97,
    cached_action: CachedAction | None = None,
    entity_ids: list[str] | None = None,
) -> ResponseCacheEntry:
    """Build a ResponseCacheEntry."""
    return ResponseCacheEntry(
        query_text=query_text,
        response_text=response_text,
        agent_id=agent_id,
        confidence=confidence,
        cached_action=cached_action,
        entity_ids=entity_ids or ["light.kitchen_ceiling"],
    )


def make_cached_action(
    service: str = "light/turn_on",
    entity_id: str = "light.kitchen_ceiling",
    service_data: dict | None = None,
) -> CachedAction:
    """Build a CachedAction."""
    return CachedAction(
        service=service,
        entity_id=entity_id,
        service_data=service_data or {},
    )


# ---------------------------------------------------------------------------
# A2A protocol factories
# ---------------------------------------------------------------------------

def make_a2a_request(
    method: str = "message/send",
    params: dict[str, Any] | None = None,
    id: str | None = None,
) -> dict[str, Any]:
    """Build an A2A JSON-RPC request dict."""
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
        "id": id or str(uuid.uuid4()),
    }


def make_a2a_response(
    result: Any = None,
    id: str | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an A2A JSON-RPC response dict."""
    resp: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": id or str(uuid.uuid4()),
    }
    if error is not None:
        resp["error"] = error
    else:
        resp["result"] = result or {"status": "ok"}
    return resp


# ---------------------------------------------------------------------------
# LLM mock factories
# ---------------------------------------------------------------------------

def make_mock_llm_response(
    content: str = "I turned on the kitchen light for you.",
    role: str = "assistant",
) -> MagicMock:
    """Build a mock litellm ChatCompletion response object.

    Mimics the structure returned by litellm.acompletion().
    """
    choice = MagicMock()
    choice.message.content = content
    choice.message.role = role
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    response.model = "openrouter/openai/gpt-4o-mini"
    response.usage.prompt_tokens = 50
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 70
    return response


def make_mock_embedding(dim: int = 384) -> list[float]:
    """Return a random embedding vector of the given dimension."""
    return [random.uniform(-1.0, 1.0) for _ in range(dim)]
