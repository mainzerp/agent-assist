"""Lightweight span collection for request tracing.

All operations are fire-and-forget: errors are logged, never raised.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import re
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from app.db.repository import TraceSpanRepository, TraceSummaryRepository

logger = logging.getLogger(__name__)

# Patterns to redact from trace metadata values
_SENSITIVE_PATTERNS = [
    (re.compile(r"\b\d{4,8}\b"), "[REDACTED_CODE]"),
    (re.compile(r'"code"\s*:\s*"[^"]*"'), '"code": "[REDACTED]"'),
]

# Parent-span tracking per async context (Q-8). A ContextVar is the
# correct mechanism for nested spans under ``asyncio.gather`` so parallel
# branches don't see each other's parents via a shared list.
_current_parent: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_parent",
    default=None,
)


def _sanitize_metadata(metadata: dict) -> dict:
    """Redact sensitive patterns from metadata string values."""
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            for pattern, replacement in _SENSITIVE_PATTERNS:
                value = pattern.sub(replacement, value)
        sanitized[key] = value
    return sanitized


SpanSource = Literal["ha", "chat", "api"]


class SpanCollector:
    """Collects spans during a single request and flushes them in bulk.

    FLOW-MED-9: ``source`` must be supplied at construction so it
    cannot silently default to ``"api"`` when a future call site
    forgets to assign it post-hoc. Callers that truly do not know
    the source (middleware building a collector before the route
    handler runs) pass ``source="api"`` explicitly and the route
    handler may rebuild the collector with the correct source if
    needed.
    """

    def __init__(self, trace_id: str, source: SpanSource = "api") -> None:
        self.trace_id = trace_id
        self.source: SpanSource = source
        self._spans: list[dict[str, Any]] = []

    def push_parent(self, span_id: str) -> contextvars.Token:
        """Set ``span_id`` as the current parent for subsequent spans and
        return a token that must be passed to :meth:`pop_parent` when done.

        Used by entry points (middleware / stream handlers) that want every
        nested span to be re-parented under a shared root span.
        """
        return _current_parent.set(span_id)

    def pop_parent(self, token: contextvars.Token) -> None:
        """Restore the previous parent span set by :meth:`push_parent`."""
        try:
            _current_parent.reset(token)
        except ValueError:
            # Token from a different context (e.g. reset after the setting
            # task has finished). Safe to ignore -- parent tracking is
            # best-effort.
            logger.debug("Could not reset _current_parent token", exc_info=True)

    @asynccontextmanager
    async def start_span(
        self,
        name: str,
        agent_id: str | None = None,
        parent_span: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Context manager that records a span's start time and duration."""
        span_id = uuid.uuid4().hex[:12]
        if parent_span is None:
            parent_span = _current_parent.get()

        span: dict[str, Any] = {
            "span_id": span_id,
            "trace_id": self.trace_id,
            "span_name": name,
            "agent_id": agent_id,
            "parent_span": parent_span,
            "start_time": datetime.now(UTC).isoformat(),
            "status": "ok",
            "metadata": {},
        }
        token = _current_parent.set(span_id)
        t0 = time.perf_counter()
        try:
            yield span
        except Exception:
            span["status"] = "error"
            raise
        finally:
            _current_parent.reset(token)
            # Allow callers to override duration (e.g. filler spans with pre-recorded timestamps)
            if "_override_duration_ms" in span:
                span["duration_ms"] = span.pop("_override_duration_ms")
                # Compute end_time from start_time + overridden duration
                try:
                    st = datetime.fromisoformat(span["start_time"])
                    span["end_time"] = (st + timedelta(milliseconds=span["duration_ms"])).isoformat()
                except Exception:
                    span["end_time"] = datetime.now(UTC).isoformat()
            else:
                span["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                span["end_time"] = datetime.now(UTC).isoformat()
            self._spans.append(span)

    async def flush(self) -> None:
        """Bulk insert all collected spans. Fire-and-forget."""
        if not self._spans:
            return
        try:
            for span in self._spans:
                if span.get("metadata"):
                    span["metadata"] = _sanitize_metadata(span["metadata"])
            await TraceSpanRepository.insert_batch(self._spans)
            # Compute and store total duration from spans
            try:
                starts = [datetime.fromisoformat(s["start_time"]) for s in self._spans if s.get("start_time")]
                if starts:
                    min_start = min(starts)
                    max_end = max(
                        datetime.fromisoformat(s["end_time"])
                        if s.get("end_time")
                        else datetime.fromisoformat(s["start_time"]) + timedelta(milliseconds=s.get("duration_ms", 0))
                        for s in self._spans
                        if s.get("start_time")
                    )
                    total_ms = round((max_end - min_start).total_seconds() * 1000, 2)
                    await TraceSummaryRepository.update_duration(self.trace_id, total_ms)
            except Exception:
                logger.debug("Could not compute total duration for trace %s", self.trace_id, exc_info=True)
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
    end_time: str | None = None,
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
            end_time=end_time,
        )
    except Exception:
        logger.warning("Failed to record span %s", span_name, exc_info=True)


async def create_trace_summary(
    trace_id: str,
    conversation_id: str | None,
    user_input: str,
    final_response: str,
    routing_agent: str,
    routing_confidence: float | None,
    routing_duration_ms: float | None,
    condensed_task: str,
    agents: list[str],
    source: str,
    agent_instructions: dict[str, str] | None = None,
    conversation_turns: list[dict] | None = None,
    device_id: str | None = None,
    area_id: str | None = None,
    device_name: str | None = None,
    area_name: str | None = None,
) -> None:
    """Create a trace_summary record. Fire-and-forget.

    FLOW-CTX-1 (0.18.6): ``device_*``/``area_*`` identify which
    satellite originated the trace. They default to ``None`` so
    existing call sites (tests, unauthenticated REST) stay valid.
    """
    try:
        await TraceSummaryRepository.create(
            {
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
                "conversation_turns": conversation_turns,
                "device_id": device_id,
                "area_id": area_id,
                "device_name": device_name,
                "area_name": area_name,
            }
        )
    except Exception:
        logger.warning("Failed to create trace summary for %s", trace_id, exc_info=True)


class _NoOpSpan:
    """No-op span for when span_collector is None."""

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        return {}

    def get(self, key, default=None):
        return default


@contextlib.asynccontextmanager
async def _optional_span(span_collector, name, **kwargs):
    if span_collector:
        async with span_collector.start_span(name, **kwargs) as span:
            yield span
    else:
        yield _NoOpSpan()
