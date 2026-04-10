"""Message dispatcher routing A2A messages to agents."""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from app.a2a.protocol import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcStreamChunk,
    MessageSendParams,
    MessageStreamParams,
    AgentDiscoverParams,
    error_response,
    success_response,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)
from app.a2a.registry import AgentRegistry
from app.a2a.transport import Transport
from app.models.agent import AgentTask

logger = logging.getLogger(__name__)


class Dispatcher:
    """Routes incoming JSON-RPC 2.0 requests to the correct handler."""

    def __init__(self, registry: AgentRegistry, transport: Transport) -> None:
        self._registry = registry
        self._transport = transport

    async def dispatch(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Dispatch a non-streaming JSON-RPC request."""
        method = request.method

        if method == "message/send":
            return await self._handle_message_send(request)
        elif method == "agent/discover":
            return await self._handle_agent_discover(request)
        elif method == "agent/list":
            return await self._handle_agent_list(request)
        else:
            return error_response(request.id, METHOD_NOT_FOUND, f"Method not found: {method}")

    async def dispatch_stream(self, request: JsonRpcRequest) -> AsyncGenerator[JsonRpcStreamChunk, None]:
        """Dispatch a streaming JSON-RPC request (message/stream)."""
        if request.method != "message/stream":
            yield JsonRpcStreamChunk(
                id=request.id,
                result={"token": "", "done": True, "error": f"Method not found: {request.method}"},
                done=True,
            )
            return

        params = request.params or {}
        agent_id = params.get("agent_id")
        task_dict = params.get("task")
        span_collector = params.pop("_span_collector", None)

        if not agent_id or task_dict is None:
            yield JsonRpcStreamChunk(
                id=request.id,
                result={"token": "", "done": True, "error": "Missing agent_id or task"},
                done=True,
            )
            return

        task = AgentTask(**task_dict)
        if span_collector:
            task._span_collector = span_collector
        async for chunk in self._transport.stream(agent_id, task, request.id):
            yield chunk

    async def _handle_message_send(self, request: JsonRpcRequest) -> JsonRpcResponse:
        params = request.params or {}
        agent_id = params.get("agent_id")
        task_dict = params.get("task")
        span_collector = params.pop("_span_collector", None)

        if not agent_id or task_dict is None:
            return error_response(request.id, INVALID_PARAMS, "Missing agent_id or task")

        task = AgentTask(**task_dict)
        if span_collector:
            task._span_collector = span_collector
        return await self._transport.send(agent_id, task, request.id)

    async def _handle_agent_discover(self, request: JsonRpcRequest) -> JsonRpcResponse:
        params = request.params or {}
        agent_id = params.get("agent_id")
        if not agent_id:
            return error_response(request.id, INVALID_PARAMS, "Missing agent_id")

        card = await self._registry.discover(agent_id)
        if card is None:
            return error_response(request.id, INVALID_PARAMS, f"Agent not found: {agent_id}")
        return success_response(request.id, card.model_dump())

    async def _handle_agent_list(self, request: JsonRpcRequest) -> JsonRpcResponse:
        agents = await self._registry.list_agents()
        return success_response(request.id, {"agents": [a.model_dump() for a in agents]})
