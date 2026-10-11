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
#: The descriptions of the optional per-call publication window, as ``TOPIC_DESCRIPTION``.
PUBLISHED_AFTER_DESCRIPTION = (
    "Optional. Only return results published on or after this date or date-time, in ISO "
    "8601 (for example 2026-10-01 or 2026-10-01T09:00:00Z). A value with no time zone is "
    "read as UTC. Results with no known publication date are kept. Set it only when the "
    "task needs sources from a time period; leave it unset otherwise."
)
PUBLISHED_BEFORE_DESCRIPTION = (
    "Optional. Only return results published before this date or date-time, in ISO 8601. "
    "A value with no time zone is read as UTC, and a date with no time includes that whole "
    "day. Results with no known publication date are kept. Set it only when the task needs "
    "sources from a time period; leave it unset otherwise."
)

#: The ``search()`` options a model may set per call.
_MODEL_ARGS = ("topic", "published_after", "published_before")


def model_search_kwargs(args: Any) -> dict[str, Any]:
    """``search()`` keyword arguments for a model's per-call tool arguments: ``topic``
    and the publication window.

    Each value is trimmed, and anything but a non-blank string is unset: no kwarg, so
    ``TELEM_TOPIC`` still applies. The routing MODE is never taken from anything the
    model sent — it comes from the ``autoRouting`` config key or ``TELEM_AUTO_ROUTING``.
    """
    if not isinstance(args, dict):
        return {}
    kwargs = {}
    for name in _MODEL_ARGS:
        value = args.get(name)
        value = value.strip() if isinstance(value, str) else ""
        if value:
            kwargs[name] = value
    return kwargs


def format_search_results(response: SearchResponse, max_len: int) -> str:
    """Format search results for a model, one block per result, content truncated."""
    blocks = [
        f"URL: {result.url}\nTitle: {result.title}\nContent: {result.content[:max_len]}"
        for result in response.results
    ]
    return "\n\n".join(blocks) if blocks else "No results found."
