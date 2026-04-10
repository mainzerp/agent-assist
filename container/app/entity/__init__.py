"""Entity subsystem -- index, alias resolution, matching signals, and matcher."""

from app.entity.index import EntityIndex
from app.entity.aliases import AliasResolver
from app.entity.signals import LevenshteinSignal, EmbeddingSignal, AliasSignal
from app.entity.matcher import EntityMatcher, MatchResult

__all__ = [
    "EntityIndex",
    "AliasResolver",
    "LevenshteinSignal",
    "EmbeddingSignal",
    "AliasSignal",
    "EntityMatcher",
    "MatchResult",
]
