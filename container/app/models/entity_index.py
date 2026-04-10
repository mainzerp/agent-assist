"""Entity index entry models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EntityIndexEntry(BaseModel):
    """A Home Assistant entity stored in the pre-embedded entity index."""

    entity_id: str = Field(..., description="HA entity ID (e.g., light.kitchen_ceiling)")
    friendly_name: str = Field("", description="Human-readable entity name")
    domain: str = Field("", description="HA domain (e.g., light, switch, climate)")
    area: str | None = Field(None, description="HA area/room assignment")
    device_class: str | None = Field(None, description="HA device class")
    aliases: list[str] = Field(default_factory=list, description="User-defined aliases")

    @property
    def embedding_text(self) -> str:
        """Text representation used for embedding."""
        parts = [self.friendly_name]
        if self.area:
            parts.append(self.area)
        if self.domain:
            parts.append(self.domain)
        if self.device_class:
            parts.append(self.device_class)
        return " ".join(parts)
