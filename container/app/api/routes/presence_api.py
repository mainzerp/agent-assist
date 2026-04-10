"""Presence detection status and configuration API endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.security.auth import require_admin_session
from app.db.repository import SettingsRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin/presence",
    tags=["admin-presence"],
    dependencies=[Depends(require_admin_session)],
)


class PresenceConfigUpdate(BaseModel):
    enabled: bool | None = None
    decay_timeout: float | None = None


@router.get("/status")
async def get_presence_status(request: Request) -> dict[str, Any]:
    """Get room confidence, sensors, and config."""
    presence_detector = request.app.state.presence_detector
    enabled = (await SettingsRepository.get_value("presence.enabled", "true")) == "true"
    decay_timeout = float(await SettingsRepository.get_value("presence.decay_timeout", "300"))

    room_confidence: dict[str, float] = {}
    sensors: list[dict[str, Any]] = []

    if presence_detector:
        room_confidence = presence_detector.get_room_confidence()
        sensors = [
            {
                "entity_id": s.entity_id,
                "sensor_type": s.sensor_type,
                "area": s.area,
            }
            for s in presence_detector._sensors
        ]

    return {
        "enabled": enabled,
        "decay_timeout": decay_timeout,
        "room_confidence": room_confidence,
        "sensors": sensors,
    }


@router.put("/config")
async def update_presence_config(body: PresenceConfigUpdate) -> dict[str, str]:
    """Update presence detection settings."""
    if body.enabled is not None:
        await SettingsRepository.set(
            "presence.enabled",
            "true" if body.enabled else "false",
            value_type="boolean",
            category="presence",
        )
    if body.decay_timeout is not None:
        await SettingsRepository.set(
            "presence.decay_timeout",
            str(body.decay_timeout),
            value_type="number",
            category="presence",
        )
    return {"status": "updated"}
