"""Test that pure ASGI middleware does not buffer streaming responses.

Regression test for CRIT-6 (deep code review): the SetupRedirectMiddleware
and TracingMiddleware previously subclassed BaseHTTPMiddleware which buffers
the entire response body before sending it. SSE/WS endpoints could not flush
the first byte until the stream completed.

We drive the ASGI protocol directly (httpx.ASGITransport itself buffers,
which would defeat the test) and assert that the first http.response.body
message arrives well before the downstream generator finishes.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route

from app.middleware.auth import SetupRedirectMiddleware
from app.middleware.tracing import TracingMiddleware

SLEEP_S = 0.4
TIMING_BUDGET_S = 0.2


async def _slow_sse(request: Request) -> StreamingResponse:
    async def event_gen():
        yield b"data: first\n\n"
        await asyncio.sleep(SLEEP_S)
        yield b"data: done\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _make_streaming_app():
    return Starlette(routes=[Route("/sse", _slow_sse)])


def _scope():
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/sse",
        "raw_path": b"/sse",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }


async def _drive(app):
    """Run the ASGI app, capturing each send message with its arrival time."""
    received = [{"type": "http.request", "body": b"", "more_body": False}]

    async def receive():
        if received:
            return received.pop(0)
        return {"type": "http.disconnect"}

    captured: list[tuple[float, dict]] = []
    t0 = time.perf_counter()

    async def send(message):
        captured.append((time.perf_counter() - t0, message))

    await app(_scope(), receive, send)
    return captured


def _first_body_offset(captured):
    for delta, msg in captured:
        if msg["type"] == "http.response.body" and msg.get("body"):
            return delta, msg["body"]
    raise AssertionError("no body message captured")


@pytest.mark.asyncio
@patch("app.middleware.auth.SetupStateRepository")
async def test_setup_redirect_middleware_does_not_buffer(mock_repo):
    mock_repo.is_complete = AsyncMock(return_value=True)

    app = _make_streaming_app()
    app.add_middleware(SetupRedirectMiddleware)

    captured = await _drive(app)
    first_offset, first_body = _first_body_offset(captured)
    assert first_offset < TIMING_BUDGET_S, (
        f"first body chunk arrived after {first_offset:.3f}s -- middleware is buffering"
    )
    assert b"first" in first_body


@pytest.mark.asyncio
@patch("app.middleware.tracing.TraceSummaryRepository", create=True)
async def test_tracing_middleware_does_not_buffer(mock_summary):
    mock_summary.update_duration = AsyncMock()

    app = _make_streaming_app()
    app.add_middleware(TracingMiddleware)

    captured = await _drive(app)

    start_msg = next(m for _, m in captured if m["type"] == "http.response.start")
    header_names = {k.decode("ascii").lower() for k, _ in start_msg["headers"]}
    assert "x-trace-id" in header_names

    first_offset, first_body = _first_body_offset(captured)
    assert first_offset < TIMING_BUDGET_S, (
        f"first body chunk arrived after {first_offset:.3f}s -- middleware is buffering"
    )
    assert b"first" in first_body


@pytest.mark.asyncio
@patch("app.middleware.tracing.TraceSummaryRepository", create=True)
async def test_tracing_middleware_populates_websocket_span(mock_summary):
    """FLOW-WS-SPAN-1 (P1-6): TracingMiddleware must create a per-connection
    SpanCollector with source derived from the WS path, so the WS endpoint
    does not have to hardcode ``source="ha"``."""
    mock_summary.update_duration = AsyncMock()

    captured_state: dict = {}

    async def dummy_asgi(scope, receive, send):
        captured_state["state"] = scope.get("state", {}).copy()

    middleware = TracingMiddleware(dummy_asgi)

    async def _receive():
        return {"type": "websocket.disconnect"}

    async def _send(_):
        return None

    ws_scope = {
        "type": "websocket",
        "path": "/ws/conversation",
        "headers": [],
        "client": ("127.0.0.1", 12345),
    }

    await middleware(ws_scope, _receive, _send)

    sc = captured_state["state"]["span_collector"]
    assert sc.source == "ha"
    assert captured_state["state"]["source"] == "ha"
    assert captured_state["state"]["trace_id"]
    assert captured_state["state"]["root_span_id"]


@pytest.mark.asyncio
@patch("app.middleware.tracing.TraceSummaryRepository", create=True)
async def test_tracing_middleware_ws_source_defaults_to_api(mock_summary):
    """FLOW-WS-SPAN-1 (P1-6): WS routes that are not /ws/conversation fall
    back to ``source="api"`` instead of silently inheriting ``"ha"``."""
    mock_summary.update_duration = AsyncMock()

    captured_state: dict = {}

    async def dummy_asgi(scope, receive, send):
        captured_state["state"] = scope.get("state", {}).copy()

    middleware = TracingMiddleware(dummy_asgi)

    async def _receive():
        return {"type": "websocket.disconnect"}

    async def _send(_):
        return None

    ws_scope = {
        "type": "websocket",
        "path": "/ws/some-future-route",
        "headers": [],
        "client": ("127.0.0.1", 12345),
    }

    await middleware(ws_scope, _receive, _send)

    sc = captured_state["state"]["span_collector"]
    assert sc.source == "api"
