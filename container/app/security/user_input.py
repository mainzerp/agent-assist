"""Live user input preparation for conversation ingress."""

from __future__ import annotations

from dataclasses import dataclass

from app.security.sanitization import check_injection_patterns, sanitize_input


@dataclass(frozen=True)
class PreparedUserInput:
    """Sanitized live user text plus prompt-injection detection metadata."""

    text: str
    injection_detected: bool


def prepare_user_text(text: str) -> PreparedUserInput:
    """Sanitize one live user turn and detect prompt-injection patterns."""
    sanitized = sanitize_input(text or "")
    injection_detected = check_injection_patterns(sanitized)
    return PreparedUserInput(text=sanitized, injection_detected=injection_detected)
