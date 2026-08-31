"""Exa-style wrapping of OpenAI clients: Telem search exposed as a model tool.

``wrap_openai`` / ``wrap_async_openai`` patch an OpenAI client's
``chat.completions.create`` in place and return the same object. One wrapped client
is one agent conversation: the wrap passively records the conversation flowing
through it and sends it with every search as trajectory-v5 metadata (flat
``message_history`` plus ``session_key``/``fingerprint``/``node_key``).
"""

from __future__ import annotations

import inspect
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from telem.errors import APIStatusError
from telem.integrations._openai_trajectory import (
    parent_snapshot,
    trajectory_metadata,
    window_anchor,
)
from telem.integrations._tools import (
    DEFAULT_RESULT_MAX_LEN,
    format_search_results as _format_results,
)
from telem.integrations._trajectory_v5 import (
    CAPABILITY_CAP,
    DELIVERED_CAP,
    BoundedKeySet,
    DeliveryPlan,
    cache_scope,
    is_missing_snapshots_refusal,
    plan_ancestors,
    record_delivery,
    restore_omitted,
)
from telem.models import SearchResponse

TELEM_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "telem_search",
        "description": "Search the web for relevant information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query to search for."},
            },
            "required": ["query"],
        },
    },
}
# reasoning/reasoning_details are provider extras (OpenRouter, deepseek); the official
# OpenAI API never returns reasoning content, so they are simply absent there.
_MESSAGE_KEYS = ("role", "content", "name", "tool_call_id", "reasoning", "reasoning_details")


def _serialize_tool_call(call: Any) -> dict[str, Any]:
    """Reduce a tool call (dict or OpenAI object) to a plain dict.

    Handles function calls and the newer custom tool calls (which carry a ``custom``
    payload instead of ``function``); unknown shapes keep id/type only.
    """
    if isinstance(call, dict):
        data: dict[str, Any] = {"id": call.get("id", ""), "type": call.get("type", "function")}
        function = call.get("function")
        if isinstance(function, dict):
            data["function"] = {
                "name": function.get("name", ""),
                "arguments": function.get("arguments", ""),
            }
        custom = call.get("custom")
        if isinstance(custom, dict):
            data["custom"] = {"name": custom.get("name", ""), "input": custom.get("input", "")}
        return data
    data = {"id": getattr(call, "id", ""), "type": getattr(call, "type", "function")}
    function = getattr(call, "function", None)
    if function is not None:
        data["function"] = {"name": function.name, "arguments": function.arguments}
    custom = getattr(call, "custom", None)
    if custom is not None:
        data["custom"] = {
            "name": getattr(custom, "name", ""),
            "input": getattr(custom, "input", ""),
        }
    return data


def _serialize_message(message: Any) -> dict[str, Any]:
    """Reduce a message (dict or OpenAI object) to plain-dict role/content/tool fields."""
    if isinstance(message, dict):
        data = {key: message[key] for key in _MESSAGE_KEYS if message.get(key) is not None}
        tool_calls = message.get("tool_calls")
    else:
        data = {
            key: value
            for key in _MESSAGE_KEYS
            if (value := getattr(message, key, None)) is not None
        }
        tool_calls = getattr(message, "tool_calls", None)
    data.setdefault("role", "")
    data.setdefault("content", None)
    if tool_calls:
        data["tool_calls"] = [_serialize_tool_call(call) for call in tool_calls]
    return data


def _call_id(call: Any) -> str:
    """Return the tool-call id from a dict or OpenAI tool-call object."""
    return call.get("id", "") if isinstance(call, dict) else call.id


def _call_name(call: Any) -> str:
    """Return the function name of a tool call; "" for non-function (custom) calls.

    Only function calls can be telem_search, so "" safely routes custom tool calls
    to the caller's own handling.
    """
    function = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
    if function is None:
        return ""
    return function.get("name", "") if isinstance(function, dict) else function.name


def _query_from_call(call: Any) -> str:
    """Extract the query string from a telem_search tool call's JSON arguments."""
    function = (call.get("function") or {}) if isinstance(call, dict) else call.function
    arguments = function.get("arguments", "") if isinstance(function, dict) else function.arguments
    try:
        query = json.loads(arguments)["query"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"telem_search call has no usable query: {arguments!r}") from exc
    if not isinstance(query, str):
        raise ValueError(f"telem_search query must be a string, got {type(query)!r}")
    return query


def _tool_message(call_id: str, content: str) -> dict[str, Any]:
    """Build the tool message answering a telem_search call.

    Exactly role/tool_call_id/content — strict providers reject extra fields, so
    telem answers are tracked by call id in the wrap state, not marked on the message.
    """
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _normalize_tool_result(tool_call: Any, result: Any) -> dict[str, Any]:
    """Pass dict results through; wrap anything else into a proper tool message."""
    if isinstance(result, dict):
        return result
    return {"role": "tool", "tool_call_id": _call_id(tool_call), "content": str(result)}


@dataclass
class _WrapConfig:
    """Conversation-level search defaults fixed at wrap() time.

    Only the knobs a conversation-level agent needs: everything else (fields,
    provider exclusions, full content, ...) reaches the wire through the Telem
    client's own defaults, since the wrap calls the same ``search()``.
    """

    tier: str | None
    providers_include: list[str] | None
    num_results: int | None
    goal: str | None
    context: str | None
    result_max_len: int


@dataclass
class _WrapState:
    """Mutable wrap state attached to the patched client as ``_telem_wrap``."""

    telem: Any
    config: _WrapConfig
    conversation_id: str
    context_window_id: str | None = None
    ancestors: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    runner_mode: bool = False
    #: The id and unix timestamp of the most recent completion. They identify this
    #: agent's current message for node keys, and its spawn point when it delegates.
    last_completion_id: str = ""
    last_completion_created: int | None = None
    #: Incremental history, phase 1, PER WRAPPED CLIENT: the
    #: snapshot keys whose context this wrap has proven landed, and the scopes whose
    #: backend has proven it implements the contract guard. Both are keyed by
    #: ``(base_url, sha256(api_key))`` — node ids are derived from the account key
    #: server-side, so flipping either voids every belief about that world. A fresh
    #: wrap starts cold, which costs one full re-send per snapshot: the safe
    #: direction, and the reason this is not process-global.
    delivered: BoundedKeySet = field(default_factory=lambda: BoundedKeySet(DELIVERED_CAP))
    capability: BoundedKeySet = field(default_factory=lambda: BoundedKeySet(CAPABILITY_CAP))


def _attach(obj: Any, name: str, value: Any) -> None:
    """Set an attribute, falling back past pydantic __setattr__ guards if needed."""
    try:
        setattr(obj, name, value)
    except (AttributeError, TypeError, ValueError):
        object.__setattr__(obj, name, value)


def _snapshot(client: Any, state: _WrapState) -> None:
    """Publish a fresh copy of the recorded conversation as ``client.telem_messages``."""
    client.telem_messages = list(state.messages)


def _record_request(client: Any, state: _WrapState, messages: list[Any]) -> None:
    """Reset the recording to the full history the caller just passed to create()."""
    state.messages = [_serialize_message(message) for message in messages]
    _snapshot(client, state)


def _record(client: Any, state: _WrapState, message: Any) -> None:
    """Append one message to the recording and refresh the public snapshot."""
    state.messages.append(_serialize_message(message))
    _snapshot(client, state)


def _record_completion(client: Any, state: _WrapState, completion: Any) -> None:
    """Record a model reply and remember the completion that produced it.

    ``completion.id`` identifies this agent's current message for node keys, and
    ``completion.created`` is the true timestamp if this agent delegates next.
    """
    state.last_completion_id = str(getattr(completion, "id", "") or "")
    created = getattr(completion, "created", None)
    state.last_completion_created = created if isinstance(created, int) else None
    _record(client, state, completion.choices[0].message)


def _prepare_kwargs(state: _WrapState, kwargs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Pop ``use_telem`` and apply the tools policy.

    Default behavior replaces the caller's tools with the telem tool (exa parity);
    loop integration (runner_mode) merges instead. Returns (kwargs, telem_enabled).

    When replacing tools, a dict-shaped ``tool_choice`` is dropped too: it references
    tools that no longer exist in the request and the API would reject it with a 400.
    String forms ("auto"/"none"/"required") stay valid and are preserved. In runner
    mode the caller's tools all survive the merge, so ``tool_choice`` is left alone.
    """
    use_telem = kwargs.pop("use_telem", "auto")
    if use_telem == "none":
        return kwargs, False
    if state.runner_mode:
        merged = list(kwargs.get("tools") or [])
        names = {
            _call_name({"function": tool.get("function")})
            for tool in merged
            if isinstance(tool, dict)
        }
        if "telem_search" not in names:
            merged.append(TELEM_SEARCH_TOOL)
        kwargs["tools"] = merged
    else:
        kwargs["tools"] = [TELEM_SEARCH_TOOL]
        if not isinstance(kwargs.get("tool_choice"), str):
            kwargs.pop("tool_choice", None)
    return kwargs, True


def _window_id(state: _WrapState) -> str:
    """This agent's context-window generation: the explicit id, else the anchor."""
    return state.context_window_id or window_anchor(state.messages)


def _search_kwargs(
    state: _WrapState, tool_call_id: str, plan: DeliveryPlan | None = None
) -> dict[str, Any]:
    """Assemble search() keyword arguments: config defaults plus the v5 payload.

    No ``session`` is sent: under trajectory v5 ``session_key`` IS the session
    identity, and asserting a second one would give the backend two competing
    answers for the same question.

    With a *plan*, the ancestor chain is filtered through phase-1 omission and the
    call's promises are recorded on it. Without one the chain goes out in full —
    which is what the helper tests and any future caller with nothing to reconcile
    a response against should get.
    """
    config = state.config
    ancestors = (
        state.ancestors
        if plan is None
        else plan_ancestors(
            state.ancestors, plan, delivered=state.delivered, capability=state.capability
        )
    )
    return {
        "tier": config.tier,
        "providers_include": config.providers_include,
        "num_results": config.num_results,
        "goal": config.goal,
        "context": config.context,
        "metadata": trajectory_metadata(
            conversation_id=state.conversation_id,
            window_id=_window_id(state),
            message_id=state.last_completion_id,
            tool_call_id=tool_call_id,
            messages=state.messages,
            ancestors=ancestors,
        ),
    }


def _delivery_plan(state: _WrapState) -> DeliveryPlan:
    """Open this call's phase-1 ledger against the scope the POST will actually use.

    Resolved BEFORE the trajectory is built: the ``(base_url, key)`` scope
    decides which ancestor contexts may be withheld, and that decision has to be
    taken against the very world this request then reaches — resolving afterwards
    could omit against one world what was only ever delivered to another. The two
    attributes are read defensively because the wrap's ``telem`` is duck-typed.
    """
    return DeliveryPlan(
        scope=cache_scope(
            getattr(state.telem, "base_url", None), getattr(state.telem, "api_key", None)
        )
    )


def _run_search(state: _WrapState, query: str, tool_call_id: str) -> SearchResponse:
    """One search, plus the single in-call retry the guard's 409 asks for.

    The retried request is the SAME kwargs object — same ``node_key`` (the message
    and tool-call ids that derive it have not moved), same ``message_history``,
    same search block — with ``context_omitted`` swapped back for ``context``. The
    refusal is a PRE-EXECUTION one (nothing ran, nothing billed, nothing persisted),
    so the retry is this call's only execution and the wrap's "a search is not
    idempotent, never re-run it" stance is untouched in substance.
    """
    plan = _delivery_plan(state)
    kwargs = _search_kwargs(state, tool_call_id, plan)
    try:
        response = state.telem.search(query, **kwargs)
    except APIStatusError as error:
        if not is_missing_snapshots_refusal(error.status_code, error.body, plan):
            raise
        restore_omitted(plan, delivered=state.delivered)
        # Exactly once: `omitted` is now empty, so a second 409 cannot re-enter
        # this branch even in principle and is surfaced like any other error.
        response = state.telem.search(query, **kwargs)
    _mark_delivered(state, plan, response)
    return response


async def _arun_search(state: _WrapState, query: str, tool_call_id: str) -> SearchResponse:
    """The async twin of :func:`_run_search`; see it for the reasoning."""
    plan = _delivery_plan(state)
    kwargs = _search_kwargs(state, tool_call_id, plan)
    try:
        response = await state.telem.search(query, **kwargs)
    except APIStatusError as error:
        if not is_missing_snapshots_refusal(error.status_code, error.body, plan):
            raise
        restore_omitted(plan, delivered=state.delivered)
        response = await state.telem.search(query, **kwargs)
    _mark_delivered(state, plan, response)
    return response


def _mark_delivered(state: _WrapState, plan: DeliveryPlan, response: Any) -> None:
    """Fold a proven response into the phase-1 caches.

    Reached only once ``search()`` has returned, which means a 2xx whose body
    parsed AND passed the V2 envelope gate. The gate is stricter than the spec
    asks — a pre-V2 2xx still committed its trajectory write — but the SDK raises
    that from inside ``search()``, so the surface simply learns nothing from it and
    re-sends the contexts in full next call: the safe direction.
    """
    record_delivery(
        plan,
        getattr(response, "raw", None),
        delivered=state.delivered,
        capability=state.capability,
    )


def _spawned_at(created: int | None) -> str | None:
    """Render a completion's unix timestamp as the delegation moment, or None."""
    if not isinstance(created, int) or isinstance(created, bool):
        return None
    try:
        return datetime.fromtimestamp(created, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except (OSError, OverflowError, ValueError):
        return None


def _freeze_parent(parent: Any) -> list[dict[str, Any]]:
    """Freeze a parent wrapped client into this child's root-first ancestor chain.

    The freeze happens HERE, at wrap() time — so wrap the child at the moment you
    delegate, not at startup. Wrapping early records a delegation whose parent had
    not spoken yet, which the wrap cannot detect and will not invent context for.
    An unwrapped (or already-detached) parent contributes no ancestors.
    """
    state = getattr(parent, "_telem_wrap", None)
    if not isinstance(state, _WrapState):
        return []
    snapshot = parent_snapshot(
        conversation_id=state.conversation_id,
        window_id=_window_id(state),
        message_id=state.last_completion_id,
        messages=state.messages,
        ancestors=state.ancestors,
        spawned_at=_spawned_at(state.last_completion_created),
    )
    return [dict(entry) for entry in state.ancestors] + [snapshot]


def _adopt_session_id(client: Any, response: SearchResponse) -> None:
    """Adopt the server's session id as a read-only handle for the sessions API."""
    if response.session_id:
        client.telem_session_id = response.session_id


def _tool_calls(completion: Any) -> list[Any]:
    """Return the completion's tool calls (empty list when there are none)."""
    return getattr(completion.choices[0].message, "tool_calls", None) or []


def _all_telem_calls(calls: list[Any]) -> bool:
    """True when there is at least one call and every call is telem_search."""
    return bool(calls) and all(_call_name(call) == "telem_search" for call in calls)


_IMPORT_HINT = (
    "could not wrap this object as an OpenAI client and the 'openai' package is not "
    "installed; install it with: pip install 'telem-sdk[openai]'"
)


def _warn_streaming() -> None:
    """Warn loudly that streaming calls bypass Telem entirely."""
    warnings.warn(
        "telem-sdk does not support streaming through a wrapped client yet: this "
        "stream=True call bypasses Telem entirely (no telem_search tool, no session "
        "tracking, no reply recording).",
        UserWarning,
        stacklevel=3,
    )


def _stream_passthrough_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Strip Telem-only kwargs for a streaming passthrough call."""
    passthrough = dict(kwargs)
    passthrough.pop("use_telem", None)
    return passthrough


def _validate_client(client: Any) -> None:
    """Duck-type check for chat.completions.create; map failures to Import/TypeError."""
    create = getattr(getattr(getattr(client, "chat", None), "completions", None), "create", None)
    if callable(create):
        return
    try:
        import openai  # noqa: F401
    except ImportError as exc:
        raise ImportError(_IMPORT_HINT) from exc
    raise TypeError(
        f"expected an OpenAI client exposing chat.completions.create, got {type(client)!r}"
    )


def _install(client: Any, state: _WrapState, patched_create: Any, runner_factory: Any) -> Any:
    """Attach wrap state, the patched create, and the runner factory, exa-style, in place."""
    client._telem_wrap = state
    client.telem_conversation_id = state.conversation_id
    client.telem_session_id = None
    client.telem_messages = []
    client.chat.completions.create = patched_create
    client.wrap_tool_runner = runner_factory
    return client


def wrap_openai(
    telem: Any,
    client: Any,
    *,
    tier: str | None = None,
    providers_include: list[str] | None = None,
    num_results: int | None = None,
    goal: str | None = None,
    context: str | None = None,
    conversation_id: str | None = None,
    context_window_id: str | None = None,
    parent: Any | None = None,
    result_max_len: int = DEFAULT_RESULT_MAX_LEN,
) -> Any:
    """Patch a sync OpenAI client in place with Telem search; returns the same object.

    Args:
        telem: The Telem client used to run searches.
        client: An ``openai.OpenAI``-shaped client (duck-typed).
        tier: Result tier for every search (``minimalist``/``default``/``extended``/
            ``max``). ``None`` falls through to the Telem client's default.
        providers_include: Provider allow-list forwarded to every search. ``None``
            falls through to the Telem client's default, then the server's.
        num_results: Results per provider for every search (server default 5). The
            only count knob a wrapped caller has — each rendered row is still capped
            by ``result_max_len``.
        goal: Optional agent goal merged into every search's metadata.
        context: Optional context string merged into every search's metadata.
        conversation_id: Stable id for this agent's conversation. Auto-minted when
            omitted — supply your own (a thread id, a request-scoped chat id) when the
            conversation outlives a single wrapped client, e.g. a web server wrapping a
            fresh client per request.
        context_window_id: Explicit context-window generation id. Omitted, the wrap
            anchors on the first message so trimming or summarizing the history starts a
            new generation on its own.
        parent: Optional parent wrapped client. Its conversation is frozen into this
            client's ``ancestors`` right now, at wrap() time — wrap the child at the
            moment it is delegated to, not at startup, or its lineage snapshot will
            predate anything the parent actually said.
        result_max_len: Max characters of each result's content fed to the model.

    Returns:
        The same client object, patched. Wrapping an already-wrapped client is a no-op.
    """
    if getattr(client, "_telem_wrap", None) is not None:
        return client
    _validate_client(client)
    state = _WrapState(
        telem=telem,
        config=_WrapConfig(
            tier=tier,
            providers_include=providers_include,
            num_results=num_results,
            goal=goal,
            context=context,
            result_max_len=result_max_len,
        ),
        conversation_id=conversation_id or str(uuid4()),
        context_window_id=context_window_id,
        ancestors=_freeze_parent(parent) if parent is not None else [],
    )
    real_create = client.chat.completions.create

    def create(**kwargs: Any) -> Any:
        messages = kwargs.get("messages") or []
        _record_request(client, state, messages)
        if kwargs.get("stream"):
            _warn_streaming()
            return real_create(**_stream_passthrough_kwargs(kwargs))
        prepared, enabled = _prepare_kwargs(state, dict(kwargs))
        completion = real_create(**prepared)
        _record_completion(client, state, completion)
        calls = _tool_calls(completion)
        if not enabled or state.runner_mode or not _all_telem_calls(calls):
            _attach(completion, "telem_responses", [])
            return completion
        responses: list[SearchResponse] = []
        tool_messages: list[dict[str, Any]] = []
        for call in calls:
            response = _run_search(state, _query_from_call(call), _call_id(call))
            _adopt_session_id(client, response)
            call_id = _call_id(call)
            message = _tool_message(call_id, _format_results(response, state.config.result_max_len))
            responses.append(response)
            tool_messages.append(message)
            _record(client, state, message)
        prepared["messages"] = [*messages, completion.choices[0].message, *tool_messages]
        final = real_create(**prepared)
        _record_completion(client, state, final)
        _attach(final, "telem_responses", responses)
        return final

    def wrap_tool_runner(fn: Any = None) -> Any:
        """Wrap a tool-dispatch function; switches this client to loop integration.

        Args:
            fn: Callable taking a tool call and returning a tool message dict or a
                plain string. ``None`` when the agent has no tools of its own.

        Returns:
            ``run_tool(tool_call)``: executes telem_search calls via the SDK and
            delegates everything else to ``fn``.
        """
        state.runner_mode = True

        def run_tool(tool_call: Any) -> dict[str, Any]:
            if _call_name(tool_call) == "telem_search":
                response = _run_search(state, _query_from_call(tool_call), _call_id(tool_call))
                _adopt_session_id(client, response)
                call_id = _call_id(tool_call)
                message = _tool_message(
                    call_id, _format_results(response, state.config.result_max_len)
                )
                _record(client, state, message)
                return message
            if fn is None:
                raise ValueError(
                    f"no handler for tool call {_call_name(tool_call)!r}; pass your tool "
                    "function to wrap_tool_runner()"
                )
            message = _normalize_tool_result(tool_call, fn(tool_call))
            _record(client, state, message)
            return message

        return run_tool

    return _install(client, state, create, wrap_tool_runner)


def wrap_async_openai(
    telem: Any,
    client: Any,
    *,
    tier: str | None = None,
    providers_include: list[str] | None = None,
    num_results: int | None = None,
    goal: str | None = None,
    context: str | None = None,
    conversation_id: str | None = None,
    context_window_id: str | None = None,
    parent: Any | None = None,
    result_max_len: int = DEFAULT_RESULT_MAX_LEN,
) -> Any:
    """Patch an async OpenAI client in place with Telem search; returns the same object.

    Args:
        telem: The AsyncTelem client used to run searches.
        client: An ``openai.AsyncOpenAI``-shaped client (duck-typed).
        tier: Result tier for every search (``minimalist``/``default``/``extended``/
            ``max``). ``None`` falls through to the Telem client's default.
        providers_include: Provider allow-list forwarded to every search. ``None``
            falls through to the Telem client's default, then the server's.
        num_results: Results per provider for every search (server default 5). The
            only count knob a wrapped caller has — each rendered row is still capped
            by ``result_max_len``.
        goal: Optional agent goal merged into every search's metadata.
        context: Optional context string merged into every search's metadata.
        conversation_id: Stable id for this agent's conversation. Auto-minted when
            omitted — supply your own (a thread id, a request-scoped chat id) when the
            conversation outlives a single wrapped client, e.g. a web server wrapping a
            fresh client per request.
        context_window_id: Explicit context-window generation id. Omitted, the wrap
            anchors on the first message so trimming or summarizing the history starts a
            new generation on its own.
        parent: Optional parent wrapped client. Its conversation is frozen into this
            client's ``ancestors`` right now, at wrap() time — wrap the child at the
            moment it is delegated to, not at startup, or its lineage snapshot will
            predate anything the parent actually said.
        result_max_len: Max characters of each result's content fed to the model.

    Returns:
        The same client object, patched. Wrapping an already-wrapped client is a no-op.
    """
    if getattr(client, "_telem_wrap", None) is not None:
        return client
    _validate_client(client)
    state = _WrapState(
        telem=telem,
        config=_WrapConfig(
            tier=tier,
            providers_include=providers_include,
            num_results=num_results,
            goal=goal,
            context=context,
            result_max_len=result_max_len,
        ),
        conversation_id=conversation_id or str(uuid4()),
        context_window_id=context_window_id,
        ancestors=_freeze_parent(parent) if parent is not None else [],
    )
    real_create = client.chat.completions.create

    async def create(**kwargs: Any) -> Any:
        messages = kwargs.get("messages") or []
        _record_request(client, state, messages)
        if kwargs.get("stream"):
            _warn_streaming()
            return await real_create(**_stream_passthrough_kwargs(kwargs))
        prepared, enabled = _prepare_kwargs(state, dict(kwargs))
        completion = await real_create(**prepared)
        _record_completion(client, state, completion)
        calls = _tool_calls(completion)
        if not enabled or state.runner_mode or not _all_telem_calls(calls):
            _attach(completion, "telem_responses", [])
            return completion
        responses: list[SearchResponse] = []
        tool_messages: list[dict[str, Any]] = []
        for call in calls:
            response = await _arun_search(state, _query_from_call(call), _call_id(call))
            _adopt_session_id(client, response)
            call_id = _call_id(call)
            message = _tool_message(call_id, _format_results(response, state.config.result_max_len))
            responses.append(response)
            tool_messages.append(message)
            _record(client, state, message)
        prepared["messages"] = [*messages, completion.choices[0].message, *tool_messages]
        final = await real_create(**prepared)
        _record_completion(client, state, final)
        _attach(final, "telem_responses", responses)
        return final

    def wrap_tool_runner(fn: Any = None) -> Any:
        """Wrap a tool-dispatch function; switches this client to loop integration.

        Args:
            fn: Sync or async callable taking a tool call and returning a tool message
                dict or a plain string. ``None`` when the agent has no tools of its own.

        Returns:
            ``run_tool(tool_call)`` coroutine function: executes telem_search calls via
            the SDK and delegates everything else to ``fn``.
        """
        state.runner_mode = True

        async def run_tool(tool_call: Any) -> dict[str, Any]:
            if _call_name(tool_call) == "telem_search":
                response = await _arun_search(
                    state, _query_from_call(tool_call), _call_id(tool_call)
                )
                _adopt_session_id(client, response)
                call_id = _call_id(tool_call)
                message = _tool_message(
                    call_id, _format_results(response, state.config.result_max_len)
                )
                _record(client, state, message)
                return message
            if fn is None:
                raise ValueError(
                    f"no handler for tool call {_call_name(tool_call)!r}; pass your tool "
                    "function to wrap_tool_runner()"
                )
            result = fn(tool_call)
            if inspect.isawaitable(result):
                result = await result
            message = _normalize_tool_result(tool_call, result)
            _record(client, state, message)
            return message

        return run_tool

    return _install(client, state, create, wrap_tool_runner)
