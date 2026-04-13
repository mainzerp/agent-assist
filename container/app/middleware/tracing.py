"""Request tracing middleware with trace ID propagation and span collection."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.analytics.tracer import SpanCollector

logger = logging.getLogger(__name__)


class TracingMiddleware(BaseHTTPMiddleware):
    """Generates a trace ID per request, attaches SpanCollector, and logs latency."""

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = uuid.uuid4().hex[:16]
        request.state.trace_id = trace_id

        # Attach span collector for downstream use
        span_collector = SpanCollector(trace_id)
        request.state.span_collector = span_collector

        method = request.method
        path = request.url.path

        logger.info("[%s] %s %s started", trace_id, method, path)
        t0 = time.perf_counter()
        start_time = datetime.now(timezone.utc).isoformat()
        status_code = 500

        root_span_id = uuid.uuid4().hex[:12]
        request.state.root_span_id = root_span_id
        span_collector._span_stack.append(root_span_id)

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Trace-Id"] = trace_id
        except Exception:
            status_code = 500
            raise
        finally:
            if span_collector._span_stack:
                span_collector._span_stack.pop()

            duration_ms = (time.perf_counter() - t0) * 1000

            # Record the top-level HTTP span
            span_collector._spans.append({
                "span_id": root_span_id,
                "trace_id": trace_id,
                "span_name": f"{method} {path}",
                "agent_id": None,
                "parent_span": None,
                "start_time": start_time,
                "duration_ms": round(duration_ms, 2),
                "status": "ok" if status_code < 400 else "error",
                "metadata": {"status_code": status_code},
            })

            # Flush all collected spans (fire-and-forget)
            try:
                await span_collector.flush()
            except Exception:
                logger.warning("Failed to flush spans for trace %s", trace_id, exc_info=True)

            # Update trace summary duration (fire-and-forget)
            try:
                from app.db.repository import TraceSummaryRepository
                await TraceSummaryRepository.update_duration(trace_id, round(duration_ms, 2))
            except Exception:
                pass

        logger.info(
            "[%s] %s %s -> %d (%.1fms)",
            trace_id, method, path, response.status_code, duration_ms,
        )
        return response
