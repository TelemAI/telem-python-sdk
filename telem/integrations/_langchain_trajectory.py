"""Translate LangGraph runtime state into Telem trajectory-v5 metadata."""

from __future__ import annotations

from typing import Any

from telem.integrations._trajectory_v5 import (
    HISTORY_TEXT_CAP,
    NONE,
    build_metadata,
    build_snapshot,
    tool_marker,
)

HARNESS_ID = "langgraph"
_TRAJECTORY_CONFIG_KEY = "_telem_trajectory"


def _messages_from_state(state: Any) -> list[Any]:
    if isinstance(state, dict):
        messages = state.get("messages", [])
    else:
        messages = getattr(state, "messages", [])
    return list(messages) if messages else []


def _message_id(message: Any) -> str | None:
    value = getattr(message, "id", None)
    return str(value) if value else None


def _message_type(message: Any) -> str:
    return str(getattr(message, "type", ""))


def _text_content(message: Any) -> str:
    content = getattr(message, "content", "")
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
    return str(content)[:HISTORY_TEXT_CAP] if content is not None else ""


def _tool_statuses(messages: list[Any]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for message in messages:
        if _message_type(message) != "tool":
            continue
        call_id = getattr(message, "tool_call_id", None)
        if call_id:
            status = getattr(message, "status", "success")
            statuses[str(call_id)] = "error" if status == "error" else "completed"
    return statuses


def _reasoning(message: Any) -> str | None:
    extras = getattr(message, "additional_kwargs", {}) or {}
    for key in ("reasoning", "reasoning_content"):
        value = extras.get(key)
        if isinstance(value, str) and value:
            return value[:HISTORY_TEXT_CAP]
    return None


def message_history(
    messages: list[Any], *, current_tool_call_id: str | None
) -> list[dict[str, str]]:
    """Flatten the active LangGraph message branch into the backend history shape."""
    statuses = _tool_statuses(messages)
    history: list[dict[str, str]] = []
    role_by_type = {"human": "user", "ai": "assistant", "system": "system"}

    for message in messages:
        message_type = _message_type(message)
        role = role_by_type.get(message_type)
        if role is None:
            continue

        pieces: list[str] = []
        text = _text_content(message)
        if text:
            pieces.append(text)
        if message_type == "ai":
            for call in getattr(message, "tool_calls", []) or []:
                call_id = str(call.get("id", ""))
                status = (
                    "running"
                    if current_tool_call_id and call_id == current_tool_call_id
                    else statuses.get(call_id, "pending")
                )
                pieces.append(tool_marker(str(call.get("name", "")), status, call.get("args")))

        content = "\n".join(pieces)[:HISTORY_TEXT_CAP]
        reasoning = _reasoning(message)
        if not content and not reasoning:
            continue
        entry = {"role": role, "content": content}
        if reasoning:
            entry["reasoning"] = reasoning
        history.append(entry)
    return history


def _current_message_id(messages: list[Any], tool_call_id: str | None) -> str:
    for message in reversed(messages):
        if _message_type(message) != "ai":
            continue
        calls = getattr(message, "tool_calls", []) or []
        if tool_call_id and any(str(call.get("id", "")) == tool_call_id for call in calls):
            return _message_id(message) or NONE
    return NONE


def _first_message_id(messages: list[Any]) -> str | None:
    for message in messages:
        if message_id := _message_id(message):
            return message_id
    return None


def _trajectory_config(configurable: dict[str, Any]) -> dict[str, Any]:
    value = configurable.get(_TRAJECTORY_CONFIG_KEY)
    return value if isinstance(value, dict) else {}


def _identity(
    messages: list[Any],
    configurable: dict[str, Any],
    tool_call_id: str,
) -> tuple[str, str]:
    private = _trajectory_config(configurable)
    first_message_id = _first_message_id(messages)
    conversation_id = str(
        private.get("conversation_id")
        or configurable.get("thread_id")
        or first_message_id
        or tool_call_id
        or NONE
    )
    window_id = str(
        private.get("context_window_id")
        or configurable.get("telem_context_window_id")
        or first_message_id
        or NONE
    )
    return conversation_id, window_id


def _ancestors(configurable: dict[str, Any]) -> list[dict[str, Any]]:
    value = _trajectory_config(configurable).get("ancestors")
    if not isinstance(value, list):
        return []
    return [dict(entry) for entry in value if isinstance(entry, dict)]


def create_child_config(
    parent_runtime: Any,
    *,
    child_thread_id: str,
    child_context_window_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze the parent runtime at delegation and return config for a child graph."""
    parent_messages = _messages_from_state(parent_runtime.state)
    parent_configurable = parent_runtime.config.get("configurable", {})
    parent_call_id = str(parent_runtime.tool_call_id) if parent_runtime.tool_call_id else ""
    parent_conversation_id, parent_window_id = _identity(
        parent_messages,
        parent_configurable,
        parent_call_id,
    )
    parent_ancestors = _ancestors(parent_configurable)
    snapshot = build_snapshot(
        harness=HARNESS_ID,
        conversation_id=parent_conversation_id,
        window_id=parent_window_id,
        message_id=_current_message_id(parent_messages, parent_call_id),
        context=message_history(parent_messages, current_tool_call_id=parent_call_id or None),
        ancestors=parent_ancestors,
    )

    child_config = dict(config or {})
    configurable = dict(child_config.get("configurable") or {})
    configurable["thread_id"] = child_thread_id
    private: dict[str, Any] = {
        "conversation_id": child_thread_id,
        "ancestors": [*parent_ancestors, snapshot],
    }
    if child_context_window_id is not None:
        private["context_window_id"] = child_context_window_id
    configurable[_TRAJECTORY_CONFIG_KEY] = private
    child_config["configurable"] = configurable
    return child_config


def trajectory_metadata(
    caller_metadata: dict[str, Any] | None,
    runtime: Any,
) -> dict[str, Any]:
    """Build trajectory-v5 metadata from an injected LangGraph runtime."""
    messages = _messages_from_state(runtime.state)
    configurable = runtime.config.get("configurable", {})
    tool_call_id = str(runtime.tool_call_id) if runtime.tool_call_id else ""
    conversation_id, window_id = _identity(messages, configurable, tool_call_id)
    return build_metadata(
        harness=HARNESS_ID,
        conversation_id=conversation_id,
        window_id=window_id,
        message_id=_current_message_id(messages, tool_call_id),
        tool_call_id=tool_call_id or NONE,
        history=message_history(messages, current_tool_call_id=tool_call_id or None),
        ancestors=_ancestors(configurable),
        kind="search",
        caller_metadata=caller_metadata,
    )
