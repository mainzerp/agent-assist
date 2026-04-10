"""Conversation history admin API endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from app.security.auth import require_admin_session
from app.db.repository import ConversationRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin/conversations",
    tags=["admin-conversations"],
    dependencies=[Depends(require_admin_session)],
)


@router.get("")
async def list_conversations(
    agent_id: str | None = Query(None),
    search: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
):
    """List/search conversation history with filters."""
    conversations = await ConversationRepository.search(
        agent_id=agent_id,
        search_text=search,
        start_date=start_date,
        end_date=end_date,
        page=page,
        per_page=per_page,
    )
    total = await ConversationRepository.count(
        agent_id=agent_id,
        search_text=search,
        start_date=start_date,
        end_date=end_date,
    )
    return {
        "conversations": conversations,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page if per_page else 0,
    }


@router.get("/{conversation_id}")
async def get_conversation_detail(conversation_id: str):
    """Get full thread detail for a conversation."""
    turns = await ConversationRepository.get_by_conversation_id(conversation_id)
    if not turns:
        return JSONResponse(status_code=404, content={"detail": "Conversation not found"})
    return {"conversation_id": conversation_id, "turns": turns}
