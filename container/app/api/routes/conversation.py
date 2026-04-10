"""WebSocket, SSE, and REST conversation endpoints."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from app.a2a.protocol import JsonRpcRequest
from app.models.conversation import ConversationRequest, ConversationResponse, StreamToken
from app.models.agent import AgentTask
from app.security.auth import require_api_key, require_api_key_ws

logger = logging.getLogger(__name__)

router = APIRouter(tags=["conversation"])

# The dispatcher is set by main.py during startup
_dispatcher = None


def set_dispatcher(dispatcher) -> None:
    """Called by main.py to inject the A2A dispatcher."""
    global _dispatcher
    _dispatcher = dispatcher


def _build_a2a_request(conv_request: ConversationRequest, method: str) -> tuple[JsonRpcRequest, AgentTask]:
    """Convert a ConversationRequest into an A2A JsonRpcRequest + AgentTask."""
    task = AgentTask(
        description=conv_request.text,
        user_text=conv_request.text,
        conversation_id=conv_request.conversation_id,
    )
    request_id = str(uuid.uuid4())
    # Route all requests through the orchestrator for intent classification
    a2a_request = JsonRpcRequest(
        method=method,
        params={"agent_id": "orchestrator", "task": task.model_dump()},
        id=request_id,
    )
    return a2a_request, task


@router.post("/api/conversation", response_model=ConversationResponse)
async def conversation_rest(
    request: ConversationRequest,
    _: str = Depends(require_api_key),
):
    """REST endpoint -- full response."""
    a2a_request, _ = _build_a2a_request(request, "message/send")
    response = await _dispatcher.dispatch(a2a_request)

    if response.error:
        return ConversationResponse(
            speech=f"Error: {response.error.message}",
            conversation_id=request.conversation_id,
        )

    result = response.result or {}
    return ConversationResponse(
        speech=result.get("speech", ""),
        conversation_id=request.conversation_id,
    )


@router.post("/api/conversation/stream")
async def conversation_sse(
    request: ConversationRequest,
    _: str = Depends(require_api_key),
):
    """SSE streaming endpoint."""
    a2a_request, _ = _build_a2a_request(request, "message/stream")

    async def generate():
        async for chunk in _dispatcher.dispatch_stream(a2a_request):
            token = StreamToken(
                token=chunk.result.get("token", ""),
                done=chunk.done,
                conversation_id=request.conversation_id if chunk.done else None,
            )
            yield f"data: {token.model_dump_json()}\n\n"

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
            data = await websocket.receive_json()
            conv_request = ConversationRequest(**data)
            a2a_request, _ = _build_a2a_request(conv_request, "message/stream")

            async for chunk in _dispatcher.dispatch_stream(a2a_request):
                token = StreamToken(
                    token=chunk.result.get("token", ""),
                    done=chunk.done,
                    conversation_id=conv_request.conversation_id if chunk.done else None,
                )
                await websocket.send_json(token.model_dump())
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
