"""Translate the OpenAI wrap's recorded messages into Telem trajectory-v5 metadata.

The wrap records messages as plain dicts (``_serialize_message``), so this module
reads dicts where the LangChain counterpart reads message objects. Everything from
the flat-history shape downward is shared with it via ``_trajectory_v5``.
"""

from __future__ import annotations

import json
from typing import Any

from telem.integrations._trajectory_v5 import (
    HISTORY_TEXT_CAP,
    NONE,
    build_metadata,
    build_snapshot,
    content_anchor,
    tool_marker,
)

HARNESS_ID = "openai"

#: Maps an OpenAI role onto the backend history role, mirroring the LangChain
#: translator's ``role_by_type``. A role absent here carries no conversation
#: content and is dropped: ``tool`` (and legacy ``function``) results are already
#: represented by their call's ``completed`` marker, and a result can be a whole
#: file. ``developer`` is OpenAI's system role on reasoning models, so it maps to
#: ``system`` rather than reaching the wire as a fourth role no other harness emits.
_ROLE_BY_NAME = {
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "developer": "system",
}


def _text(content: Any) -> str:
    """Reduce OpenAI content (string, block list, or None) to plain text."""
    if isinstance(content, str):
        return content[:HISTORY_TEXT_CAP]
    if isinstance(content, list):
        pieces: list[str] = []
        for block in content:
            if isinstance(block, str):
                pieces.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    pieces.append(text)
        return "\n".join(pieces)[:HISTORY_TEXT_CAP]
    return "" if content is None else str(content)[:HISTORY_TEXT_CAP]


def _call_name(call: dict[str, Any]) -> str:
    """Return a tool call's name, from either the function or custom payload."""
    for key in ("function", "custom"):
        block = call.get(key)
        if isinstance(block, dict):
            return str(block.get("name", ""))
    return ""


def _call_args(call: dict[str, Any]) -> Any:
    """Return a tool call's arguments, parsed when they are JSON.

    OpenAI ships arguments as a JSON *string*; parsing and letting ``tool_marker``
    re-serialize compactly keeps the marker byte-identical to every other harness.
    Unparsable arguments are forwarded verbatim rather than dropped.
    """
    function = call.get("function")
    if isinstance(function, dict):
        raw = function.get("arguments", "")
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return raw
    custom = call.get("custom")
    return custom.get("input", "") if isinstance(custom, dict) else ""


def _answered(messages: list[dict[str, Any]]) -> set[str]:
    """Ids of tool calls that already have a tool message answering them.

    OpenAI tool messages carry no status field, so an answered call is reported as
    ``completed``; a failure the caller expressed in the result text is invisible here.
    """
    return {
        str(call_id)
        for message in messages
        if message.get("role") == "tool" and (call_id := message.get("tool_call_id"))
    }


def _reasoning(message: dict[str, Any]) -> str | None:
    """Return provider reasoning text (OpenRouter/deepseek extra), if present."""
    value = message.get("reasoning")
    return value[:HISTORY_TEXT_CAP] if isinstance(value, str) and value else None


def message_history(
    messages: list[dict[str, Any]], *, current_tool_call_id: str | None = None
) -> list[dict[str, str]]:
    """Flatten recorded OpenAI messages into the backend history shape."""
    answered = _answered(messages)
    history: list[dict[str, str]] = []

    for message in messages:
        role = _ROLE_BY_NAME.get(str(message.get("role", "")))
        if role is None:
            continue

        pieces: list[str] = []
        text = _text(message.get("content"))
        if text:
            pieces.append(text)
        for call in message.get("tool_calls") or []:
            call_id = str(call.get("id", ""))
            if current_tool_call_id and call_id == current_tool_call_id:
                status = "running"
            else:
                status = "completed" if call_id in answered else "pending"
            pieces.append(tool_marker(_call_name(call), status, _call_args(call)))

        content = "\n".join(pieces)[:HISTORY_TEXT_CAP]
        reasoning = _reasoning(message)
        if not content and not reasoning:
            continue
        entry = {"role": role, "content": content}
        if reasoning:
            entry["reasoning"] = reasoning
        history.append(entry)
    return history


def window_anchor(messages: list[dict[str, Any]]) -> str:
    """Identify this context-window generation by its first message.

    OpenAI messages have no ids, so the anchor is a content hash. When a caller
    trims or summarizes the front of the history — the OpenAI equivalent of a
    compaction — the anchor moves and a new generation begins.
    """
    for message in messages:
        return content_anchor(str(message.get("role", "")), _text(message.get("content")))
    return NONE


def trajectory_metadata(
    *,
    conversation_id: str,
    window_id: str,
    message_id: str,
    tool_call_id: str,
    messages: list[dict[str, Any]],
    ancestors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the trajectory-v5 metadata for one search node."""
    return build_metadata(
        harness=HARNESS_ID,
        conversation_id=conversation_id,
        window_id=window_id,
        message_id=message_id or NONE,
        tool_call_id=tool_call_id or NONE,
        history=message_history(messages, current_tool_call_id=tool_call_id or None),
        ancestors=ancestors,
        kind="search",
    )


def parent_snapshot(
    *,
    conversation_id: str,
    window_id: str,
    message_id: str,
    messages: list[dict[str, Any]],
    ancestors: list[dict[str, Any]],
    spawned_at: str | None,
) -> dict[str, Any]:
    """Freeze a parent wrapped client as the child's newest ancestor."""
    return build_snapshot(
        harness=HARNESS_ID,
        conversation_id=conversation_id,
        window_id=window_id,
        message_id=message_id or NONE,
        context=message_history(messages),
        ancestors=ancestors,
        spawned_at=spawned_at,
    )
