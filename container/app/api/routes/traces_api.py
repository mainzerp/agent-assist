"""Trace viewer admin API endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from app.security.auth import require_admin_session
from app.db.repository import TraceSpanRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin/traces",
    tags=["admin-traces"],
    dependencies=[Depends(require_admin_session)],
)


@router.get("")
async def list_traces(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
):
    """List recent traces with pagination."""
    traces = await TraceSpanRepository.list_traces(page=page, per_page=per_page)
    total = await TraceSpanRepository.count_traces()
    return {
        "traces": traces,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page if per_page else 0,
    }


@router.get("/{trace_id}")
async def get_trace_detail(trace_id: str):
    """Get all spans for a specific trace (for Gantt visualization)."""
    spans = await TraceSpanRepository.get_trace_spans(trace_id)
    if not spans:
        return JSONResponse(status_code=404, content={"detail": "Trace not found"})
    return {"trace_id": trace_id, "spans": spans}
