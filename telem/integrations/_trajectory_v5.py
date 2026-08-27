"""Framework-neutral identity helpers for the trajectory-v5 wire protocol."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import UUID, uuid5

NS_TRAJECTORY = UUID("443866ab-1b45-5ed8-979e-52fdad07b810")
NONE = "none"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _lp(value: str) -> str:
    return f"{len(value.encode())}:{value}"


def fingerprint(harness: str, conversation_id: str) -> str:
    """Return a stable, non-reversible conversation identity."""
    return _sha256(_lp(harness) + _lp(conversation_id))


def session_key(harness: str, conversation_id: str, window_id: str) -> str:
    """Return the deterministic identity of one context-window generation."""
    name = _lp(harness) + _lp(_sha256(conversation_id)) + _lp(window_id) + _lp(NONE)
    return str(uuid5(NS_TRAJECTORY, name))


def event_node_key(
    harness: str,
    session: str,
    message_id: str,
    tool_call_id: str,
) -> str:
    """Return a replay-stable event node key for one model tool call."""
    name = _lp(harness) + _lp(session) + _lp(message_id) + _lp(tool_call_id) + _lp("event")
    return str(uuid5(NS_TRAJECTORY, name))


def snapshot_node_key(harness: str, conversation_id: str, message_id: str) -> str:
    """Return the identity of an agent snapshot at a delegation boundary."""
    name = _lp(harness) + _lp(_sha256(conversation_id)) + _lp(message_id) + _lp("snap")
    return str(uuid5(NS_TRAJECTORY, name))


HISTORY_TEXT_CAP = 128_000
TOOL_INPUT_CAP = 128_000

#: Metadata keys the trajectory payload owns. A caller-supplied value for any of
#: these is dropped rather than allowed to forge identity.
RESERVED_METADATA = frozenset(
    {
        "message_history",
        "session_key",
        "fingerprint",
        "node_key",
        "kind",
        "parent_node_key",
        "ancestors",
    }
)


def content_anchor(*parts: str) -> str:
    """Return a stable anchor id for content a host did not assign an id to.

    Length-prefixed like every other key component, so concatenation stays injective.
    """
    return _sha256("".join(_lp(part) for part in parts))


def tool_marker(name: str, status: str, args: Any) -> str:
    """Render one tool call as the compact acting-trace marker used in history."""
    try:
        serialized = json.dumps(args, ensure_ascii=False, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        serialized = ""
    suffix = f" {serialized[:TOOL_INPUT_CAP]}" if serialized else ""
    return f"[tool {name or '?'}: {status}{suffix}]"


def build_metadata(
    *,
    harness: str,
    conversation_id: str,
    window_id: str,
    message_id: str,
    tool_call_id: str,
    history: list[dict[str, str]],
    ancestors: list[dict[str, Any]],
    kind: str = "search",
    caller_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the trajectory-v5 metadata block for one event node.

    Caller metadata is preserved except for the reserved keys, which the generated
    values always win. ``parent_node_key`` is the DIRECT parent's snapshot key — the
    last entry of the root-first ancestor chain — or ``None`` when there is no parent.
    """
    metadata = {
        key: value for key, value in (caller_metadata or {}).items() if key not in RESERVED_METADATA
    }
    session = session_key(harness, conversation_id, window_id)
    metadata.update(
        {
            "message_history": history,
            "session_key": session,
            "fingerprint": fingerprint(harness, conversation_id),
            "node_key": event_node_key(harness, session, message_id, tool_call_id),
            "kind": kind,
            "parent_node_key": ancestors[-1].get("node_key") if ancestors else None,
            "ancestors": ancestors,
        }
    )
    return metadata


def build_snapshot(
    *,
    harness: str,
    conversation_id: str,
    window_id: str,
    message_id: str,
    context: list[dict[str, str]],
    ancestors: list[dict[str, Any]],
    spawned_at: str | None = None,
) -> dict[str, Any]:
    """Freeze one agent at a delegation boundary as an ancestor entry."""
    return {
        "session_key": session_key(harness, conversation_id, window_id),
        "fingerprint": fingerprint(harness, conversation_id),
        "node_key": snapshot_node_key(harness, conversation_id, message_id),
        "parent_node_key": ancestors[-1].get("node_key") if ancestors else None,
        "context": context,
        "spawned_at": spawned_at,
    }
