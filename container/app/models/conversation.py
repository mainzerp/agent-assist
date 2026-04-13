"""Conversation request and response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConversationRequest(BaseModel):
    """Incoming conversation request from HA integration."""

    text: str = Field(..., description="User input text", max_length=5000)
    conversation_id: str | None = Field(None, description="Conversation ID for multi-turn", max_length=64)
    language: str = Field("en", description="User language code", max_length=10)


class ConversationResponse(BaseModel):
    """Full conversation response sent back to HA integration."""

    speech: str = Field(..., description="Response text for TTS or display")
    conversation_id: str | None = None
    action_executed: ActionResult | None = None


class ActionResult(BaseModel):
    """Result of an HA action execution."""

    service: str = Field(..., description="HA service called (e.g., light/turn_on)")
    entity_id: str = Field(..., description="Target entity ID")
    result: str = Field("success", description="Execution result: success or error message")
    service_data: dict | None = Field(None, description="Additional service data sent")


class StreamToken(BaseModel):
    """Single token in a streaming response."""

    token: str
    done: bool = False
    conversation_id: str | None = None
