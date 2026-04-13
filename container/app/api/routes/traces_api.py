"""Trace viewer admin API endpoints."""

from __future__ import annotations

import csv
import io
import logging

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.responses import Response

from app.security.auth import require_admin_session
from app.db.repository import TraceSummaryRepository, TraceSpanRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin/traces",
    tags=["admin-traces"],
    dependencies=[Depends(require_admin_session)],
)


# --- Models ---

class LabelUpdate(BaseModel):
    label: str | None = None


# --- Static routes MUST come before /{trace_id} ---

@router.get("/export")
async def export_traces(
    search: str | None = Query(None),
    agent: str | None = Query(None),
    label: str | None = Query(None),
    date_from: str | None = Query(None, alias="from"),
    date_to: str | None = Query(None, alias="to"),
):
    """Export filtered traces as CSV."""
    rows = await TraceSummaryRepository.export_filtered(
        search=search, agent=agent, label=label,
        date_from=date_from, date_to=date_to,
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Timestamp", "Trace ID", "Conversation ID", "User Input",
        "Final Response", "Agent", "Confidence", "Duration (ms)",
        "Label", "Source",
    ])
    for row in rows:
        writer.writerow([
            row.get("created_at", ""),
            row.get("trace_id", ""),
            row.get("conversation_id", ""),
            row.get("user_input", ""),
            row.get("final_response", ""),
            row.get("routing_agent", ""),
            row.get("routing_confidence", ""),
            row.get("total_duration_ms", ""),
            row.get("label", ""),
            row.get("source", ""),
        ])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=traces_export.csv"},
    )


@router.get("/labels")
async def list_trace_labels():
    """List all distinct labels used on traces."""
    labels = await TraceSummaryRepository.list_labels()
    return {"labels": labels}


# --- Parameterized routes ---

@router.get("")
async def list_traces(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    search: str | None = Query(None),
    agent: str | None = Query(None),
    label: str | None = Query(None),
    date_from: str | None = Query(None, alias="from"),
    date_to: str | None = Query(None, alias="to"),
):
    """List recent traces with search, filters, and pagination."""
    traces = await TraceSummaryRepository.list_filtered(
        search=search, agent=agent, label=label,
        date_from=date_from, date_to=date_to,
        page=page, per_page=per_page,
    )
    total = await TraceSummaryRepository.count_filtered(
        search=search, agent=agent, label=label,
        date_from=date_from, date_to=date_to,
    )
    labels = await TraceSummaryRepository.list_labels()
    agents = await TraceSummaryRepository.list_agents()
    return {
        "traces": traces,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max(1, (total + per_page - 1) // per_page),
        "labels": labels,
        "agents": agents,
    }


@router.get("/{trace_id}")
async def get_trace_detail(trace_id: str):
    """Get detailed trace info including spans and agent executions."""
    summary = await TraceSummaryRepository.get(trace_id)
    if not summary:
        return JSONResponse(status_code=404, content={"detail": "Trace not found"})

    spans = await TraceSpanRepository.get_trace_spans(trace_id)
    spans.sort(key=lambda s: s.get("start_time", ""))

    # Build agent_executions from spans
    agent_executions = []
    for span in spans:
        if span.get("agent_id") and span["span_name"] in ("dispatch", "classify", "return", "rewrite", "ha_action"):
            response_key = "final_response" if span["span_name"] == "return" else ("rewritten_text" if span["span_name"] == "rewrite" else ("result_speech" if span["span_name"] == "ha_action" else "agent_response"))
            agent_executions.append({
                "agent_id": span["agent_id"],
                "span_name": span["span_name"],
                "duration_ms": span["duration_ms"],
                "status": span["status"],
                "response": (span.get("metadata") or {}).get(response_key, ""),
            })

    # Build inter-agent communication from spans
    agent_communication = []
    classify_span = None
    dispatch_spans = []
    return_span = None
    routing_cached = False

    for span in spans:
        if span.get("span_name") == "classify":
            classify_span = span
            routing_cached = (span.get("metadata") or {}).get("routing_cached", False)
        elif span.get("span_name") == "dispatch":
            dispatch_spans.append(span)
        elif span.get("span_name") == "return":
            return_span = span

    user_input = summary.get("user_input", "")
    final_response = summary.get("final_response", "")

    # Detect response_cache_hit (no classify span, return span has response_cache_hit)
    response_cache_hit = False
    if return_span and not classify_span:
        ret_meta = return_span.get("metadata") or {}
        response_cache_hit = ret_meta.get("response_cache_hit", False)

    if response_cache_hit:
        # Response cache hit short-circuit
        target = (return_span.get("metadata") or {}).get("from_agent", "")
        routing_cached = True
        agent_communication.append({
            "from_agent": "user",
            "to_agent": "orchestrator",
            "task": user_input,
            "response": "",
            "memory": summary.get("conversation_turns") or [],
        })
        # Find ha_action and rewrite spans for the cached path
        ha_action_span = None
        rewrite_span = None
        for span in spans:
            if span.get("span_name") == "ha_action":
                ha_action_span = span
            elif span.get("span_name") == "rewrite":
                rewrite_span = span
        if ha_action_span:
            ha_meta = ha_action_span.get("metadata") or {}
            agent_communication.append({
                "from_agent": "orchestrator (cached action)",
                "to_agent": ha_meta.get("entity", "Home Assistant"),
                "task": ha_meta.get("action", ""),
                "response": "success" if ha_meta.get("success") else "failed",
            })
        if rewrite_span:
            rw_meta = rewrite_span.get("metadata") or {}
            agent_communication.append({
                "from_agent": "response cache",
                "to_agent": "rewrite-agent",
                "task": (rw_meta.get("original_text") or "")[:200],
                "response": (rw_meta.get("rewritten_text") or "")[:200],
            })
        agent_communication.append({
            "from_agent": "orchestrator (response cache)",
            "to_agent": "user",
            "task": "",
            "response": final_response,
            "response_cache_hit": True,
        })
    elif classify_span:
        meta = classify_span.get("metadata") or {}
        target = meta.get("target_agent", "")

        # Step 1: User -> Orchestrator
        agent_communication.append({
            "from_agent": "user",
            "to_agent": "orchestrator",
            "task": user_input,
            "response": "",
            "memory": summary.get("conversation_turns") or [],
        })

        if len(dispatch_spans) <= 1:
            # Single-agent path (backward compatible)
            dispatch_span = dispatch_spans[0] if dispatch_spans else None
            agent_resp = ""
            if dispatch_span:
                agent_resp = (dispatch_span.get("metadata") or {}).get("agent_response", "")
            condensed = meta.get("condensed_task", "")

            step2 = {
                "from_agent": "orchestrator",
                "to_agent": target,
                "task": condensed,
                "response": agent_resp,
            }
            if condensed == user_input:
                step2["task_pass_through"] = True
            agent_communication.append(step2)

            mediated = False
            if return_span:
                ret_meta = return_span.get("metadata") or {}
                mediated = ret_meta.get("mediated", False)
            agent_communication.append({
                "from_agent": target,
                "to_agent": "orchestrator",
                "task": "",
                "response": final_response,
                "response_unchanged": (agent_resp == final_response and not mediated),
            })
        else:
            # Multi-agent fan-out path
            for ds in dispatch_spans:
                ds_meta = ds.get("metadata") or {}
                ds_agent = ds.get("agent_id", "")
                ds_resp = ds_meta.get("agent_response", "")
                ds_task = ds_meta.get("condensed_task", "")
                agent_communication.append({
                    "from_agent": "orchestrator",
                    "to_agent": ds_agent,
                    "task": ds_task,
                    "response": ds_resp,
                    "parallel": True,
                })
            # Combined return
            mediated = True  # Multi-agent always has LLM merge
            if return_span:
                ret_meta = return_span.get("metadata") or {}
                mediated = ret_meta.get("mediated", True)
            agent_communication.append({
                "from_agent": ", ".join(ds.get("agent_id", "") for ds in dispatch_spans),
                "to_agent": "orchestrator",
                "task": "",
                "response": final_response,
                "response_unchanged": False,
            })

    routing = {
        "selected_agent": summary.get("routing_agent"),
        "confidence": summary.get("routing_confidence"),
        "duration_ms": summary.get("routing_duration_ms"),
        "reasoning": summary.get("routing_reasoning"),
    }

    # Enrich with multi-agent routing details from classify span
    if classify_span:
        cls_meta = classify_span.get("metadata") or {}
        if cls_meta.get("multi_agent"):
            routing["multi_agent"] = True
            routing["all_agents"] = cls_meta.get("target_agent", "")
            routing["agent_instructions"] = summary.get("agent_instructions") or {}

    # Compute total duration from spans if not stored
    total_duration_ms = summary.get("total_duration_ms")
    if not total_duration_ms and spans:
        try:
            from datetime import datetime, timedelta
            starts = [datetime.fromisoformat(s["start_time"]) for s in spans if s.get("start_time")]
            if starts:
                min_start = min(starts)
                max_end = max(
                    datetime.fromisoformat(s["start_time"]) + timedelta(milliseconds=s.get("duration_ms", 0))
                    for s in spans if s.get("start_time")
                )
                total_duration_ms = round((max_end - min_start).total_seconds() * 1000, 2)
        except Exception:
            pass

    return {
        "trace_id": trace_id,
        "conversation_id": summary.get("conversation_id"),
        "timestamp": summary.get("created_at"),
        "duration_ms": total_duration_ms,
        "user_input": summary.get("user_input"),
        "final_response": summary.get("final_response"),
        "routing": routing,
        "agent_instructions": summary.get("agent_instructions"),
        "label": summary.get("label"),
        "source": summary.get("source"),
        "spans": spans,
        "agent_executions": agent_executions,
        "agent_communication": agent_communication,
        "routing_cached": routing_cached,
        "conversation_turns": summary.get("conversation_turns") or [],
    }


@router.put("/{trace_id}/label")
async def update_trace_label(trace_id: str, payload: LabelUpdate):
    """Update or clear the label on a trace."""
    summary = await TraceSummaryRepository.get(trace_id)
    if not summary:
        return JSONResponse(status_code=404, content={"detail": "Trace not found"})
    await TraceSummaryRepository.update_label(trace_id, payload.label)
    return {"status": "ok", "trace_id": trace_id, "label": payload.label}
