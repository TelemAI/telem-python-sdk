"""Shared helpers for integrations that expose Telem as a model tool."""

from __future__ import annotations

from typing import Any

from telem.models import SearchResponse

DEFAULT_RESULT_MAX_LEN = 2048

#: The model-facing description of the optional per-call ``topic`` tool argument, shared
#: by every integration that exposes ``telem_search``.
TOPIC_DESCRIPTION = (
    "Optional. Set it only when the answer must come from one site: linkedin, reddit, or x "
    "(twitter is also accepted). Leave it unset otherwise."
)


def topic_search_kwargs(topic: Any) -> dict[str, Any]:
    """``search()`` keyword arguments for a model's per-call ``topic``.

    The topic is trimmed, and anything but a non-blank string is unset: no kwargs, so
    ``TELEM_TOPIC`` still applies. The routing MODE is never taken from anything the
    model sent — it comes from the ``autoRouting`` config key or ``TELEM_AUTO_ROUTING``.
    """
    topic = topic.strip() if isinstance(topic, str) else ""
    return {"topic": topic} if topic else {}


def format_search_results(response: SearchResponse, max_len: int) -> str:
    """Format search results for a model, one block per result, content truncated."""
    blocks = [
        f"URL: {result.url}\nTitle: {result.title}\nContent: {result.content[:max_len]}"
        for result in response.results
    ]
    return "\n\n".join(blocks) if blocks else "No results found."
