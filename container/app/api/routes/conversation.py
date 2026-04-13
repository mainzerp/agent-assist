"""WebSocket, SSE, and REST conversation endpoints."""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from app.a2a.protocol import JsonRpcRequest
from app.analytics.tracer import SpanCollector
from app.models.conversation import ConversationRequest, ConversationResponse, StreamToken
from app.models.agent import AgentTask, TaskContext
from app.security.auth import require_api_key, require_api_key_ws

logger = logging.getLogger(__name__)

router = APIRouter(tags=["conversation"])

# Maximum allowed WebSocket message size in bytes (10 KB)
_MAX_WS_MESSAGE_SIZE = 10_000

# The dispatcher is set by main.py during startup
_dispatcher = None


def set_dispatcher(dispatcher) -> None:
    """Called by main.py to inject the A2A dispatcher."""
    global _dispatcher
    _dispatcher = dispatcher


def _build_a2a_request(conv_request: ConversationRequest, method: str, span_collector=None) -> tuple[JsonRpcRequest, AgentTask]:
    """Convert a ConversationRequest into an A2A JsonRpcRequest + AgentTask."""
    context = None
    if conv_request.device_id or conv_request.area_id:
        context = TaskContext(
            device_id=conv_request.device_id,
            area_id=conv_request.area_id,
        )
    task = AgentTask(
        description=conv_request.text,
        user_text=conv_request.text,
        conversation_id=conv_request.conversation_id,
        context=context,
    )
    request_id = str(uuid.uuid4())
    # Route all requests through the orchestrator for intent classification
    a2a_request = JsonRpcRequest(
        method=method,
        params={
            "agent_id": "orchestrator",
            "task": task.model_dump(),
            "_span_collector": span_collector,
        },
        id=request_id,
    )
    return a2a_request, task


@router.post("/api/conversation", response_model=ConversationResponse)
async def conversation_rest(
    request: Request,
    conv_request: ConversationRequest,
    _: str = Depends(require_api_key),
):
    """REST endpoint -- full response."""
    span_collector = getattr(request.state, "span_collector", None)
    if span_collector:
        span_collector.source = "ha"
    a2a_request, _ = _build_a2a_request(conv_request, "message/send", span_collector)
    response = await _dispatcher.dispatch(a2a_request)

    if response.error:
        return ConversationResponse(
            speech=f"Error: {response.error.message}",
            conversation_id=conv_request.conversation_id,
        )

    result = response.result or {}
    return ConversationResponse(
        speech=result.get("speech", ""),
        conversation_id=result.get("conversation_id") or conv_request.conversation_id,
    )


@router.post("/api/conversation/stream")
async def conversation_sse(
    request: Request,
    conv_request: ConversationRequest,
    _: str = Depends(require_api_key),
):
    """SSE streaming endpoint."""
    span_collector = getattr(request.state, "span_collector", None)
    if span_collector:
        span_collector.source = "ha"
    a2a_request, _ = _build_a2a_request(conv_request, "message/stream", span_collector)

    async def generate():
        root_span_id = getattr(request.state, "root_span_id", None)
        if span_collector and root_span_id:
            span_collector._span_stack.append(root_span_id)
        try:
            async for chunk in _dispatcher.dispatch_stream(a2a_request):
                token = StreamToken(
                    token=chunk.result.get("token", ""),
                    done=chunk.done,
                    conversation_id=chunk.result.get("conversation_id") if chunk.done else None,
                    mediated_speech=chunk.result.get("mediated_speech") if chunk.done else None,
                )
                yield f"data: {token.model_dump_json()}\n\n"
        finally:
            if span_collector and root_span_id and root_span_id in span_collector._span_stack:
                span_collector._span_stack.remove(root_span_id)
            if span_collector:
                await span_collector.flush()

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.websocket("/ws/conversation")
async def ws_conversation(
    websocket: WebSocket,
    _: str = Depends(require_api_key_ws),
):
    """WebSocket streaming endpoint."""
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            if len(raw) > _MAX_WS_MESSAGE_SIZE:
                await websocket.send_json({"error": "Message too large", "max_bytes": _MAX_WS_MESSAGE_SIZE})
                continue
            data = json.loads(raw)
            conv_request = ConversationRequest(**data)
            trace_id = uuid.uuid4().hex[:16]
            span_collector = SpanCollector(trace_id)
            span_collector.source = "ha"
            a2a_request, _ = _build_a2a_request(conv_request, "message/stream", span_collector)

            try:
                async for chunk in _dispatcher.dispatch_stream(a2a_request):
                    token = StreamToken(
                        token=chunk.result.get("token", ""),
                        done=chunk.done,
                        conversation_id=chunk.result.get("conversation_id") if chunk.done else None,
                        mediated_speech=chunk.result.get("mediated_speech") if chunk.done else None,
                    )
                    await websocket.send_json(token.model_dump())
            finally:
                await span_collector.flush()
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
