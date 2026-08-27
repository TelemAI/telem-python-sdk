"""Shared helpers for integrations that expose Telem as a model tool."""

from __future__ import annotations

from telem.models import SearchResponse

DEFAULT_RESULT_MAX_LEN = 2048


def format_search_results(response: SearchResponse, max_len: int) -> str:
    """Format search results for a model, one block per result, content truncated."""
    blocks = [
        f"URL: {result.url}\nTitle: {result.title}\nContent: {result.content[:max_len]}"
        for result in response.results
    ]
    return "\n\n".join(blocks) if blocks else "No results found."
