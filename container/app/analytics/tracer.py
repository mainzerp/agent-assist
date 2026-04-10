"""Lightweight span collection for request tracing.

All operations are fire-and-forget: errors are logged, never raised.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from app.db.repository import TraceSpanRepository, TraceSummaryRepository

logger = logging.getLogger(__name__)


class SpanCollector:
    """Collects spans during a single request and flushes them in bulk."""

    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self.source = "api"  # default, overridden by caller ("ha", "chat")
        self._spans: list[dict[str, Any]] = []
        self._span_stack: list[str] = []

    @asynccontextmanager
    async def start_span(
        self,
        name: str,
        agent_id: str | None = None,
        parent_span: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Context manager that records a span's start time and duration."""
        span_id = uuid.uuid4().hex[:12]
        if parent_span is None and self._span_stack:
            parent_span = self._span_stack[-1]

        span: dict[str, Any] = {
            "span_id": span_id,
            "trace_id": self.trace_id,
            "span_name": name,
            "agent_id": agent_id,
            "parent_span": parent_span,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "status": "ok",
            "metadata": {},
        }
        self._span_stack.append(span_id)
        t0 = time.perf_counter()
        try:
            yield span
        except Exception:
            span["status"] = "error"
            raise
        finally:
            self._span_stack.pop()
            span["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            self._spans.append(span)

    async def flush(self) -> None:
        """Bulk insert all collected spans. Fire-and-forget."""
        if not self._spans:
            return
        try:
            await TraceSpanRepository.insert_batch(self._spans)
        except Exception:
            logger.warning("Failed to flush %d trace spans", len(self._spans), exc_info=True)
        finally:
            self._spans.clear()


async def record_span(
    trace_id: str,
    span_name: str,
    start_time: str,
    duration_ms: float,
    agent_id: str | None = None,
    parent_span: str | None = None,
    status: str = "ok",
    metadata: dict | None = None,
) -> None:
    """Record a single span directly. Fire-and-forget."""
    try:
        await TraceSpanRepository.insert(
            trace_id=trace_id,
            span_name=span_name,
            start_time=start_time,
            duration_ms=duration_ms,
            agent_id=agent_id,
            parent_span=parent_span,
            status=status,
            metadata=metadata,
        )
    except Exception:
        logger.warning("Failed to record span %s", span_name, exc_info=True)


async def create_trace_summary(
    trace_id: str,
    conversation_id: str | None,
    user_input: str,
    final_response: str,
    routing_agent: str,
    routing_confidence: float,
    routing_duration_ms: float | None,
    condensed_task: str,
    agents: list[str],
    source: str,
    agent_instructions: dict[str, str] | None = None,
) -> None:
    """Create a trace_summary record. Fire-and-forget."""
    try:
        await TraceSummaryRepository.create({
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "user_input": user_input,
            "final_response": final_response,
            "agents": agents,
            "total_duration_ms": None,
            "source": source,
            "routing_agent": routing_agent,
            "routing_confidence": routing_confidence,
            "routing_duration_ms": routing_duration_ms,
            "routing_reasoning": None,
            "agent_instructions": agent_instructions or {routing_agent: condensed_task},
        })
    except Exception:
        logger.warning("Failed to create trace summary for %s", trace_id, exc_info=True)
