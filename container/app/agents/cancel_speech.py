"""Short spoken acknowledgement when the user dismisses the interaction only."""

from __future__ import annotations


def cancel_interaction_ack(language: str | None) -> str:
    """Return a brief TTS-safe line after cancel-interaction classification."""
    lang = (language or "en").lower()
    if lang.startswith("de"):
        return "Alles klar."
    return "Okay."
