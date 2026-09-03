"""Source-backed research retrieval."""

from .knowledge import (
    KnowledgeIndexError,
    build_index,
    index_stats,
    query_index,
)

__all__ = [
    "KnowledgeIndexError",
    "build_index",
    "index_stats",
    "query_index",
]
