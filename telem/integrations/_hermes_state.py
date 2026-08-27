"""Per-session trajectory state for the Hermes plugin.

Hermes hands a plugin its conversation through *hooks*, which fire on threads the
tool handler never runs on: ``pre_api_request`` on the agent loop's thread,
``subagent_start`` on the **parent's** thread for a child that has not started
yet. A :class:`~contextvars.ContextVar` cannot carry any of that —
``propagate_context_to_thread`` copies a context parent → worker and never back,
and nothing at all connects a parent to its child — so state is keyed by the
Hermes session id in a module-level map behind a lock. Concurrency is real: the
gateway runs turns on a shared worker pool and delegate children run in their
own, so one process holds a parent, its children, and unrelated sessions all
firing the same global callbacks.

Nothing here imports Hermes, a Telem client, or the trajectory helpers: it is a
bounded dict of opaque window ids, and it is tested as one.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, NamedTuple
from uuid import uuid4

#: Hard cap on tracked sessions. ``on_session_finalize`` eviction is best effort
#: — a crash or SIGKILL fires nothing, and child sessions have no teardown hook
#: at all — so this bound, not the hook, is what keeps the map finite.
#:
#: One delegating prompt costs several entries (each subagent is its own session
#: and none of them are ever finalized), so the headroom is smaller than the
#: number suggests.
MAX_SESSIONS = 128


class Trajectory(NamedTuple):
    """What one session contributes to a trajectory node."""

    messages: list[dict[str, Any]]
    window_id: str
    ancestors: list[dict[str, Any]]


class _Entry:
    __slots__ = ("messages", "window_id", "ancestors", "baseline")

    def __init__(self, window_id: str) -> None:
        self.messages: list[dict[str, Any]] = []
        #: Opaque id of this context-window generation, straight through to
        #: ``session_key``. Deliberately not a sequence number — see
        #: :meth:`SessionState._next_window`.
        self.window_id = window_id
        self.ancestors: list[dict[str, Any]] = []
        #: Length of the last ``pre_api_request`` snapshot — what the compaction
        #: check compares against. ``len(self.messages)`` cannot serve once
        #: :meth:`SessionState.append_message` has added the in-flight assistant
        #: turn: the next retry re-delivers the *pre*-append list, so every retry
        #: would read as a compaction and burn a generation.
        self.baseline = 0


class SessionState:
    """A bounded, lock-guarded map of Hermes session id to trajectory state."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, _Entry] = OrderedDict()
        self._max = max(1, max_sessions)

    # -- writes ------------------------------------------------------------ #
    def record_history(self, session_id: str, messages: Any) -> None:
        """Replace this session's history; a *shorter* list starts a new window.

        Overwrite, never append: ``pre_api_request`` fires once per API call and
        again on every retry. The length drop is the only compaction signal a
        plugin gets — in-place compaction swaps the live message set for a
        shorter one and fires no hook, only an event callback we cannot reach.
        """
        rows = (
            [row for row in messages if isinstance(row, dict)] if isinstance(messages, list) else []
        )
        with self._lock:
            entry = self._touch(session_id)
            if entry is None:
                return
            if len(rows) < entry.baseline:
                entry.window_id = self._next_window()
            entry.baseline = len(rows)
            entry.messages = rows

    def append_message(self, session_id: str, message: dict[str, Any]) -> None:
        """Add the turn the model just produced, ahead of the tools it calls.

        ``pre_api_request`` fires *before* the API call, so its snapshot stops
        short of the assistant turn that call returns — the very turn that
        decides to search, and the one carrying the reasoning that motivated it.
        Hermes appends that turn to its own list before dispatching tools
        (``agent/conversation_loop.py:6699``, tools at ``:6775``), so appending it
        here on ``post_api_request`` is what makes a tool's snapshot match what
        the model could actually see when it called the tool.

        The baseline is deliberately left alone: it tracks what the *hook*
        delivered, so a retry re-delivering the pre-append list is not mistaken
        for a compaction.
        """
        if not isinstance(message, dict) or not message:
            return
        with self._lock:
            entry = self._touch(session_id)
            if entry is not None:
                # Rebind rather than mutate: `read` hands out list copies, but a
                # caller holding one from before the append should not see it grow.
                entry.messages = [*entry.messages, message]

    def set_ancestors(self, session_id: str, ancestors: list[dict[str, Any]]) -> None:
        """Freeze a child's root-first ancestor chain at the moment it is spawned."""
        chain = list(ancestors)
        with self._lock:
            entry = self._touch(session_id)
            if entry is not None:
                entry.ancestors = chain

    def drop(self, session_id: str) -> None:
        """Forget a retired session."""
        with self._lock:
            self._sessions.pop(session_id or "", None)

    # -- reads ------------------------------------------------------------- #
    def read(self, session_id: str) -> Trajectory:
        """Return this session's state as copies, or empty for an unknown id."""
        with self._lock:
            entry = self._sessions.get(session_id or "")
            if entry is None:
                return Trajectory(messages=[], window_id="", ancestors=[])
            return Trajectory(
                messages=list(entry.messages),
                window_id=entry.window_id,
                ancestors=list(entry.ancestors),
            )

    def tracked(self) -> int:
        """How many sessions are held right now (the bound's test surface)."""
        with self._lock:
            return len(self._sessions)

    # -- internals --------------------------------------------------------- #
    def _touch(self, session_id: str) -> _Entry | None:
        """Return the entry for *session_id*, creating it and enforcing the cap.

        The caller holds the lock. A blank id is refused rather than stored:
        Hermes passes ``agent.session_id or ""``, and one shared ``""`` key would
        pool unrelated conversations into a single history.
        """
        if not session_id:
            return None
        entry = self._sessions.get(session_id)
        if entry is None:
            entry = self._sessions[session_id] = _Entry(self._next_window())
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)
        return entry

    @staticmethod
    def _next_window() -> str:
        """Mint a context-window id that nothing has ever issued or will reissue.

        Hermes is the only surface that has to invent this. Every other one reads
        it from its host — opencode hashes the session's own compaction and revert
        markers, LangChain takes ``context_window_id`` from config, the MCP server
        reads the host's ``_meta``. Hermes exposes nothing: compaction fires no
        hook, only an ``agent.event_callback("session:compress", …)`` a plugin
        cannot register, so a shrinking history is the only signal there is.

        Opaque and random rather than a counter, because every counter we can keep
        is one we can also LOSE, and losing it regenerates a used id. ``session_key``
        hashes (harness, conversation_id, window_id) with the first two fixed for a
        conversation, so a restarted count reissues an old key, and the backend —
        which binds ``session_key`` to a backend session id, unique on
        (account, session_key), ON CONFLICT DO NOTHING — then files the new context
        window onto the old session. Two windows as one, flagged nowhere.

        Two ways to lose a counter, both routine: the capacity cap evicts a session
        that is merely quiet rather than finished (children are never finalized), and
        a process restart resets everything while ``hermes -c`` resumes the same
        session id. A random id has no state to lose, so neither can reissue.

        The cost is deliberate: a resumed or evicted session gets a NEW key, so its
        trajectory shows a break rather than continuing. That is the honest reading —
        we lost the history that defined the old window — and a visible split beats a
        silent merge, which corrupts without a signal.

        Full 122 bits, never truncated. Shortening to 32 makes collisions real at a
        few tens of thousands of windows; at full width the birthday probability is
        ~1e-33 across a whole conversation, and a collision only matters WITHIN one
        conversation anyway, since ``conversation_id`` separates the rest.
        """
        return uuid4().hex
