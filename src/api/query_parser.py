from __future__ import annotations

__all__ = ["parse_query"]


def parse_query(query: str) -> str:
    """Strip surrounding whitespace and validate the query is non-empty."""
    normalized = query.strip()
    if not normalized:
        raise ValueError("query must not be empty or whitespace-only")
    return normalized
