"""Monitors: a query that Telem runs on a schedule, with an agent that keeps what matches.

A monitor has the agent unless it is created with ``agent=False``, and a monitor with the
agent must end, so its create needs a ``ttl``. Read what the agent finds with ``changes`` or
``iter_changes``, and the changes of many monitors in one read with ``changes_across`` or
``iter_changes_across``. ``poll`` reads a monitor's own results and is deprecated.

A monitor created with ``inputs`` watches a list of companies: ``onboarding`` reads what
the first read of each company found and ``apply_onboarding`` applies your review of it,
``expansions`` lists what the monitor proposes to add (``confirm_expansion``,
``revert_expansion``), ``data_map`` says what it collects, and ``companies`` lists its
companies. ``feedback`` labels a fact of any monitor with an agent."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import time
import warnings
from collections.abc import AsyncIterator, Generator, Iterator, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from telem._base import (
    _iso,
    batch_retry_wait,
    create_retry_wait,
    path_segment,
    resolve_idempotency_key,
)
from telem._normalize import build_search_response
from telem.errors import TelemError, TelemTimeoutError
from telem.models import (
    AgentDocument,
    ApplyResult,
    ChangeRecord,
    ChangesPage,
    CompanyList,
    DataMap,
    ExpansionsPage,
    FeedbackResult,
    Monitor,
    MonitorBatchResult,
    MonitorInputs,
    MonitorPage,
    MonitorsChangesPage,
    OnboardingSheet,
    SearchResponse,
)

if TYPE_CHECKING:
    from telem.async_client import AsyncTelem
    from telem.client import Telem


#: Why a create with the agent and no ``ttl`` is refused before anything is sent.
_NEEDS_TTL = (
    'A monitor with an agent must end: pass ttl, such as ttl="30d" (at most 30 days), '
    "or agent=False for a monitor without an agent."
)
#: The same, for a create with ``inputs``. The router refuses ``inputs`` with
#: ``agent=False`` (400); the SDK does not check that before it sends.
_NEEDS_TTL_INPUTS = (
    'A monitor with inputs must end: pass ttl, such as ttl="30d" (at most 365 days).'
)


def _inputs_body(inputs: dict[str, Any] | MonitorInputs) -> dict[str, Any]:
    """``inputs`` for the wire: a dict as given, or a model's fields that are set. The
    router checks every value."""
    if isinstance(inputs, MonitorInputs):
        dumped = inputs.model_dump(by_alias=True)
        return {key: value for key, value in dumped.items() if value is not None}
    return inputs


def _create_body(
    query: str,
    frequency: str | None,
    metadata: dict[str, Any] | None,
    webhook_url: str | None,
    ttl: str | None,
    search: dict[str, Any] | None,
    include_domains: list[str] | None,
    agent: bool | None,
    inputs: dict[str, Any] | MonitorInputs | None,
    *,
    preset: str | None = None,
    tags: list[str] | None = None,
    default_vertical: str | None = None,
) -> dict[str, Any]:
    # An explicit ``None`` is the default, as it was before: the agent is on.
    agent_on = True if agent is None else bool(agent)
    if agent_on and not ttl:
        raise ValueError(_NEEDS_TTL if inputs is None else _NEEDS_TTL_INPUTS)
    options: dict[str, Any] = {}
    if frequency is not None:
        options["frequency"] = frequency
    if include_domains is not None:
        options["include_domains"] = include_domains
    if search is not None:
        options["search"] = _search_block(search)
    body: dict[str, Any] = {"query": query, "options": options}
    if metadata is not None:
        body["metadata"] = metadata
    if webhook_url is not None:
        body["webhook_url"] = webhook_url
    if ttl is not None:
        body["ttl"] = ttl
    if inputs is not None:
        body["inputs"] = _inputs_body(inputs)
        # Unpublished: the client's default vertical, on inputs that do not set one.
        # A copy, so the caller's dict is never changed. Unset adds nothing.
        if default_vertical is not None and "vertical" not in body["inputs"]:
            body["inputs"] = {**body["inputs"], "vertical": default_vertical}
    body["agent"] = agent_on
    if preset is not None:
        body["preset"] = preset
    if tags is not None:
        body["tags"] = tags
    return body


def _search_block(search: dict[str, Any]) -> dict[str, Any]:
    """The monitor's search options for the wire: the keys of ``search()``, with
    ``providers_include``/``providers_exclude`` as ``providers``. No client
    default applies: a monitor's search is set on the monitor alone."""
    block = {k: v for k, v in search.items() if not k.startswith("providers_")}
    providers = {
        k.removeprefix("providers_"): v for k, v in search.items() if k.startswith("providers_")
    }
    if providers:
        block["providers"] = providers
    return block


def _update_body(
    status: str | None,
    frequency: str | None,
    webhook_url: str | None,
    ttl: str | None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    body = {"status": status, "frequency": frequency, "ttl": ttl}
    body = {key: value for key, value in body.items() if value is not None}
    if webhook_url is not None:
        body["webhook_url"] = webhook_url or None  # "" removes the webhook
    if tags is not None:
        body["tags"] = tags  # [] clears them; the router refuses null
    return body


_SIGNATURE = re.compile(r"[0-9a-f]{64}")


def verify_webhook(body: bytes | str, header: str, secret: str, *, tolerance: int = 300) -> bool:
    """Check a monitor webhook: the ``Telem-Signature`` header against the raw body.

    Args:
        body: The raw request body, exactly as received.
        header: The ``Telem-Signature`` header.
        secret: The monitor's ``webhook_secret``.
        tolerance: The oldest signature to accept, in seconds.

    Returns:
        True when any ``v1`` signature is correct and not older than ``tolerance``.
        A header can carry several: during a secret's rotation, one for each secret.
        A header with no ``t``, or more than one, is False.
    """
    parts = [item.strip().split("=", 1) for item in header.split(",") if "=" in item]
    timestamps = [value for key, value in parts if key == "t"]
    signatures = [value for key, value in parts if key == "v1" and _SIGNATURE.fullmatch(value)]
    if len(timestamps) != 1 or not signatures:
        return False
    timestamp = timestamps[0]
    if (
        not (timestamp.isascii() and timestamp.isdigit())
        or abs(time.time() - int(timestamp)) > tolerance
    ):
        return False
    raw = body.encode() if isinstance(body, str) else body
    expected = hmac.new(secret.encode(), f"{timestamp}.".encode() + raw, hashlib.sha256).hexdigest()
    return any(hmac.compare_digest(expected, signature) for signature in signatures)


#: Whether this process has been told that ``poll`` is deprecated: it is told once.
_poll_warned = False

_POLL_DEPRECATED = (
    "monitors.poll() is deprecated: read what the monitor agent finds with "
    "monitors.iter_changes() or monitors.changes()."
)


def _warn_poll_deprecated() -> None:
    """Warn that ``poll`` is deprecated, the first time this process polls.

    The warning points at the caller's line, not at this module. It is marked as given
    only once it is given, so a caller that turns warnings into errors gets the error
    on every poll, not on the first one alone.
    """
    global _poll_warned
    if _poll_warned:
        return
    # 1 is this function, 2 the poll method, 3 the code that called it.
    warnings.warn(_POLL_DEPRECATED, DeprecationWarning, stacklevel=3)
    _poll_warned = True


def _poll_body(session: str | None, metadata: dict[str, Any] | None) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if session is not None:
        body["session_id"] = session
    if metadata is not None:
        body["metadata"] = metadata
    return body


def _list_params(
    limit: int | None, cursor: str | None, tag: str | None = None
) -> dict[str, Any] | None:
    params: dict[str, Any] = {}
    if limit is not None:
        params["limit"] = limit
    if cursor is not None:
        params["cursor"] = cursor
    if tag is not None:
        params["tag"] = tag
    return params or None


def _changes_params(
    cursor: int | None,
    since: str | datetime | None,
    kinds: list[str] | None,
    limit: int | None,
) -> dict[str, Any] | None:
    """The query of a changes read: what the caller set, and nothing else."""
    params: dict[str, Any] = {}
    if cursor is not None:
        params["cursor"] = cursor
    if since is not None:
        params["since"] = _iso(since)
    if kinds is not None:
        # One kind written as a bare string is one kind, not one per letter.
        params["kinds"] = kinds if isinstance(kinds, str) else ",".join(kinds)
    if limit is not None:
        params["limit"] = limit
    return params or None


def _has_more(page: ChangesPage, cursor: int) -> bool:
    """Whether to read on from ``page.next_cursor``, after a read that started at ``cursor``.

    Only while ``next_cursor`` is below the watermark, which the router answers only to a
    full page: a shorter page ends at the watermark, and there the reader is caught up.
    Never after an empty page or a page that did not move the cursor, whatever else it
    says: the router sends neither below the watermark, and reading on could bring the
    same page back for ever."""
    return bool(page.changes) and cursor < page.next_cursor < page.watermark


def _across_params(
    tag: str | None,
    ids: str | list[str] | None,
    cursor: str | None,
    since: str | datetime | None,
    kinds: list[str] | None,
    limit: int | None,
) -> dict[str, Any]:
    """The query of a read across monitors: ``tag`` or ``ids``, then what the caller set."""
    if (tag is None) == (ids is None):
        raise ValueError("send tag or ids, one of them")
    if tag is not None:
        params: dict[str, Any] = {"tag": tag}
    else:
        if not ids:
            raise ValueError("ids needs at least one monitor id")
        # One id written as a bare string is one id, not one per letter.
        params = {"ids": ids if isinstance(ids, str) else ",".join(ids)}
    if cursor is not None:
        params["cursor"] = cursor
    params.update(_changes_params(None, since, kinds, limit) or {})
    return params


def _reads_across_on(page: MonitorsChangesPage, cursor: str | None) -> bool:
    """Whether to read on from ``page.next_cursor``: only after a full page (``more``) that
    served changes and moved the cursor, so a stalled router cannot repeat a page for ever."""
    return page.more and bool(page.changes) and page.next_cursor != cursor


#: Added to a create that timed out: the router may have made the monitor anyway.
_MAY_EXIST = "The monitor may exist: list monitors before you create it again."

_MAY_EXIST_MANY = "The monitors may exist: list monitors before you create them again."

_sleep = time.sleep
_async_sleep = asyncio.sleep
_clock = time.monotonic

_BATCH_PATH = "/v1/monitors:batch"
#: The most monitors one ``POST /v1/monitors:batch`` takes.
_BATCH_SIZE = 10
_ITEM_FIELDS = frozenset(
    {
        "query",
        "frequency",
        "metadata",
        "webhook_url",
        "ttl",
        "search",
        "include_domains",
        "agent",
        "inputs",
        "preset",
        "tags",
        "idempotency_key",
    }
)

#: A step of a bulk create: ``("post", body)`` or ``("sleep", seconds)``. The post's
#: answer is sent back in, or its ``TelemError`` thrown in.
_Steps = Generator[tuple[str, Any], Any, list[MonitorBatchResult]]


def _batch_items(items: Sequence[Mapping[str, Any]], default_vertical: str | None) -> list[dict]:
    """Every monitor's wire body with its ``idempotency_key``, all checked before any is sent."""
    if isinstance(items, (str, bytes, Mapping)):
        raise ValueError("create_many takes a list of monitors, each a dict of create's arguments")
    bodies = []
    given: dict[str, int] = {}
    for index, item in enumerate(items):
        try:
            if not isinstance(item, Mapping):
                raise ValueError("a monitor must be a dict of create's arguments")
            unknown = sorted(set(item) - _ITEM_FIELDS)
            if unknown:
                raise ValueError(f"unknown argument {unknown[0]!r}")
            if item.get("query") is None:
                raise ValueError("query is required")
            if not isinstance(item["query"], str):
                raise ValueError("query must be a string")
            body = _create_body(
                item["query"],
                item.get("frequency"),
                item.get("metadata"),
                item.get("webhook_url"),
                item.get("ttl"),
                item.get("search"),
                item.get("include_domains"),
                item.get("agent"),
                item.get("inputs"),
                preset=item.get("preset"),
                tags=item.get("tags"),
                default_vertical=default_vertical,
            )
            body["idempotency_key"] = resolve_idempotency_key(item.get("idempotency_key"))
            try:
                json.dumps(body, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"not JSON: {exc}") from None
        except ValueError as exc:
            raise ValueError(f"monitor {index}: {exc}") from exc
        bodies.append(body)
        if item.get("idempotency_key") is not None:
            first = given.setdefault(body["idempotency_key"], index)
            if first != index:
                raise ValueError(f"monitors {first} and {index} have the same idempotency_key")
    return bodies


def _batch_answer(
    data: Any, sent: list[int], keys: list[str]
) -> tuple[dict[int, MonitorBatchResult], TelemError | None]:
    """The results of one batch call by their place in the whole list, and the first
    malformed entry's error, if any. ``sent`` holds the place of each monitor sent; the
    answer's ``index`` is a position in it."""
    results = data.get("results") if isinstance(data, dict) else None
    count = f"the batch answer does not have one result for each of the {len(sent)} monitors"
    if not isinstance(results, list):
        return {}, TelemError(f"{count} sent")
    answer: dict[int, MonitorBatchResult] = {}
    # A wrong count still keeps each entry it can place.
    problem = TelemError(f"{count} sent") if len(results) != len(sent) else None
    for entry in results:
        try:
            position, result = _batch_entry(entry, sent, keys, answer)
        except TelemError as exc:
            problem = problem or exc
            continue
        answer[position] = result
    return answer, problem


def _batch_entry(
    entry: Any, sent: list[int], keys: list[str], answer: dict[int, MonitorBatchResult]
) -> tuple[int, MonitorBatchResult]:
    index = entry.get("index") if isinstance(entry, dict) else None
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(sent):
        raise TelemError("a batch result has no valid index")
    status = entry.get("status")
    if not isinstance(status, int) or isinstance(status, bool):
        raise TelemError(f"batch result {index} has no status")
    position = sent[index]
    if position in answer:
        raise TelemError(f"two batch results have index {index}")
    fields = {**entry, "index": position, "idempotency_key": keys[position]}
    try:
        return position, MonitorBatchResult.model_validate(fields)
    except ValidationError as exc:
        # Named by field only: the values can hold a webhook_secret.
        bad = sorted({".".join(map(str, e["loc"])) for e in exc.errors(include_input=False)})
        raise TelemError(f"batch result {index} is not valid: {', '.join(bad)}") from None


def _comes_back(result: MonitorBatchResult | None) -> bool:
    """Whether this monitor has no final answer: none yet, a ``429``, or a ``409``
    ``idempotency_in_progress`` (its first create is still running)."""
    if result is None or result.status == 429:
        return True
    return result.status == 409 and (result.error or {}).get("code") == "idempotency_in_progress"


def _failed(
    error: TelemError, results: list[MonitorBatchResult | None], keys: list[str]
) -> TelemError:
    """``error`` as raised by a bulk create once a call was sent: monitors may exist, so it
    says so, with the results so far and every key."""
    note = "" if error.message.endswith(".") else "."
    message = f"{error.message}{note} {_MAY_EXIST_MANY}"
    raised = type(error)(message, status_code=error.status_code, body=error.body)
    raised.retry_after = error.retry_after
    raised.results = [result for result in results if result is not None]
    raised.idempotency_keys = list(keys)
    return raised


def _keep(error: BaseException, results: list[MonitorBatchResult | None], keys: list[str]) -> None:
    """Put the results so far and every key on an exception that stopped a bulk create after
    a call was sent, such as a cancel or a ``KeyboardInterrupt``, where it takes them."""
    if isinstance(error, GeneratorExit):
        return
    try:
        error.results = [result for result in results if result is not None]
        error.idempotency_keys = list(keys)
    except AttributeError:
        pass


def _create_many_steps(bodies: list[dict]) -> _Steps:
    """A bulk create, ten monitors per call, one call at a time. Each call is retried as
    a create is; then only its monitors that Telem asked for again are sent again, with
    their keys, until none is left or the next wait would end past the budget. A call that
    fails whole raises, with the answers so far and every key on the error."""
    keys = [body["idempotency_key"] for body in bodies]
    results: list[MonitorBatchResult | None] = [None] * len(bodies)
    for offset in range(0, len(bodies), _BATCH_SIZE):
        pending = list(range(offset, min(offset + _BATCH_SIZE, len(bodies))))
        attempt, started = 0, _clock()
        while True:
            try:
                data = yield ("post", {"monitors": [bodies[i] for i in pending]})
            except TelemError as exc:
                elapsed = _clock() - started
                wait = create_retry_wait(exc, attempt, elapsed, gateway_timeouts=True)
                if wait is None:
                    raise _failed(exc, results, keys) from exc
            except BaseException as exc:
                _keep(exc, results, keys)
                raise
            else:
                answer, problem = _batch_answer(data, pending, keys)
                for i, result in answer.items():
                    results[i] = result
                if problem is not None:
                    raise _failed(problem, results, keys) from None
                pending = [i for i in pending if _comes_back(results[i])]
                if not pending:
                    break
                retry_afters = [results[i].retry_after for i in pending]
                wait = batch_retry_wait(retry_afters, attempt, _clock() - started)
                if wait is None:
                    break
            try:
                yield ("sleep", wait)
            except BaseException as exc:
                _keep(exc, results, keys)
                raise
            attempt += 1
    return results


def _page_from(data: Any) -> MonitorPage:
    return MonitorPage.model_validate(data if isinstance(data, dict) else {})


def _agent_from(data: Any) -> AgentDocument:
    # The model ignores what it does not name; ``raw`` keeps the whole document.
    return AgentDocument.model_validate({**data, "raw": data} if isinstance(data, dict) else data)


def _path(monitor_id: str, suffix: str = "") -> str:
    if not monitor_id:
        raise ValueError("a monitor id is required")
    return f"/v1/monitors/{path_segment(monitor_id)}{suffix}"


def _expansion_path(monitor_id: str, expansion_id: str, action: str) -> str:
    """The path of ``:confirm`` or ``:revert`` of one expansion, its id one encoded segment."""
    if not expansion_id:
        raise ValueError("an expansion id is required")
    return _path(monitor_id, f"/agent/expansions/{path_segment(expansion_id)}:{action}")


def _status_params(status: str | None) -> dict[str, Any] | None:
    return None if status is None else {"status": status}


def _page_params(offset: int | None, limit: int | None) -> dict[str, Any] | None:
    params = _set_fields({"offset": offset, "limit": limit})
    return params or None


def _set_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """The fields the caller set: a ``None`` is not sent."""
    return {key: value for key, value in fields.items() if value is not None}


def _feedback_body(fact_id: str, verdict: str, category: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {"fact_id": fact_id, "verdict": verdict}
    if category is not None:
        body["category"] = category
    return body


class MonitorsResource:
    """Synchronous monitor operations bound to a :class:`~telem.client.Telem` client."""

    def __init__(self, client: Telem) -> None:
        self._client = client

    def create(
        self,
        query: str,
        *,
        frequency: str | None = None,
        metadata: dict[str, Any] | None = None,
        webhook_url: str | None = None,
        ttl: str | None = None,
        search: dict[str, Any] | None = None,
        include_domains: list[str] | None = None,
        agent: bool = True,
        inputs: dict[str, Any] | MonitorInputs | None = None,
        preset: str | None = None,
        tags: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> Monitor:
        """Create a monitor. Telem runs ``query`` on the schedule, and its agent checks
        every result against ``query`` and keeps the ones that match.

        Args:
            query: What to watch for, as a plain sentence.
            frequency: ``"1h"``, ``"6h"``, ``"12h"``, ``"1d"`` or ``"7d"``. If you leave it
                out, the monitor runs once a day.
            metadata: Query lineage keys, as in a search. They choose the session that
                records the polls of the monitor.
            webhook_url: An ``https`` URL that receives the new results.
            ttl: End the monitor after this long, such as ``"7d"`` or ``"30d"`` (at most
                ``"30d"``, or ``"365d"`` with ``inputs``). Required with the agent. Without
                the agent, the monitor never ends if you leave it out.
            search: Also run a web search for ``query`` on the schedule, with these
                search options, named as in :meth:`Telem.search`: ``num_results`` and ``tier``.
                ``{}`` uses the defaults. It adds to what a poll returns.
            include_domains: Only results from these sites and their subdomains, such as
                ``["sec.gov"]``, in what a poll returns.
            agent: ``True`` (the default) starts the monitor agent with the monitor. Read
                what it keeps with :meth:`iter_changes` or :meth:`changes`. ``False``
                creates a monitor without an agent, which you read with the deprecated
                :meth:`poll`.
            inputs: Watch a list of companies: the companies, and the schema that says
                what to collect about them, such as the preset ``"biopharma-v0"``. A
                :class:`~telem.models.MonitorInputs` or a dict, with the router's field
                names. It needs the agent (the router refuses ``inputs`` with
                ``agent=False``) and a ``ttl`` of at most ``"365d"``. Read what its first
                read of each company found with :meth:`onboarding`.
            preset: The fields every result has: ``"article"`` (``headline``,
                ``publication``, ``published_at``, ``summary``, ``kind``), ``"person"``,
                ``"company"`` or ``"listing"``. The agent keeps those fields as they are and
                may add more. Leave it out and the agent picks the fields from ``query``.
                It needs the agent and is refused with ``inputs``.
            tags: Labels of your own, to list the monitor by with ``list(tag=...)``: at
                most 10. Telem strips and lower-cases each one, which must then be 1 to 32
                of ``a-z``, ``0-9``, ``-`` and ``_``, starting with a letter or digit, and
                drops repeats. Anything else is refused with a 422.
            idempotency_key: The ``Idempotency-Key`` header, 1 to 255 visible ASCII
                characters. By default each call makes a new one. Pass your own to
                retry a create across a restart.

        The create is retried on a ``429``, on a ``409`` ``idempotency_in_progress``,
        and after a timeout or a lost connection, for up to about 120 seconds, waiting
        for ``Retry-After`` plus a little random extra. Every retry sends the same key,
        and Telem answers a repeat of the key with the first monitor for 24 hours. A
        key is scoped to its API key: under another API key the same key is a new
        create.

        Returns:
            The new :class:`Monitor`. Its ``agent_state`` shows the agent. When the agent
            did not start, ``agent_state`` is ``None`` and ``agent_note`` says why. With
            ``webhook_url``, its ``webhook_secret`` signs the deliveries; Telem shows it
            only in this response.

        Raises:
            ValueError: The agent is on and ``ttl`` is missing, or ``idempotency_key`` is
                not valid. Nothing was sent.
            TelemTimeoutError: The last retry timed out too. The monitor may exist.
        """
        body = _create_body(
            query,
            frequency,
            metadata,
            webhook_url,
            ttl,
            search,
            include_domains,
            agent,
            inputs,
            preset=preset,
            tags=tags,
            default_vertical=self._client.default_vertical,
        )
        headers = {"Idempotency-Key": resolve_idempotency_key(idempotency_key)}
        attempt, started = 0, _clock()
        while True:
            try:
                data = self._client._post("/v1/monitors", body, headers=headers)
                return Monitor.model_validate(data)
            except TelemError as exc:
                wait = create_retry_wait(exc, attempt, _clock() - started)
                if wait is None:
                    if isinstance(exc, TelemTimeoutError):
                        raise TelemTimeoutError(f"{exc.message}. {_MAY_EXIST}") from exc
                    raise
            _sleep(wait)
            attempt += 1

    def create_many(self, items: Sequence[Mapping[str, Any]]) -> list[MonitorBatchResult]:
        """Create many monitors, ten per call. A monitor that fails does not stop the others.

        Args:
            items: One dict per monitor, with the arguments of :meth:`create`: ``query``,
                and any of ``frequency``, ``ttl``, ``agent``, ``inputs``, ``preset``,
                ``tags``, ``webhook_url``, ``metadata``, ``search``, ``include_domains``
                and ``idempotency_key``. Each monitor gets its own key: a new one, or
                yours. Pass your own keys to send the list again after a restart.

        More than ten monitors go in calls of ten, one after the other. Each call is
        retried as :meth:`create` is, and after a ``502`` or ``504`` too: the call may have
        outlasted the edge. Then the monitors that Telem answered with a
        ``429`` (or a ``409`` ``idempotency_in_progress``) are sent again, with the same
        keys, after the longest ``retry_after`` among them, for up to about 120 seconds
        per call. Any other error of a monitor is its result and is not retried.

        Returns:
            One :class:`MonitorBatchResult` per monitor, in the order of ``items``: its
            ``status``, its ``monitor`` or its ``error``, and its ``idempotency_key``. A
            monitor still answered with a ``429`` or a ``409`` ``idempotency_in_progress``
            (it is still being made) when the time is up keeps that result: send it again
            later with the same key. A made monitor's ``webhook_secret`` is in its
            result, here and in an error's ``results``: do not log them whole.

        Raises:
            ValueError: A monitor is not valid, as in :meth:`create`, is not JSON, or has
                the same ``idempotency_key`` as another; the message names its index.
                Nothing was sent.
            TelemError: A call failed whole, a ``429`` past the time included, or its
                answer was not valid. The error's ``results`` holds the last answer of
                each monitor answered so far and its ``idempotency_keys`` the key of every
                monitor, in the order of ``items``: send the monitors without a ``201``
                or ``200`` again with the same keys. Once a call was sent, its message
                always says that monitors may exist. An entry of an answer that cannot be
                read, a made monitor's included, is left out of ``results``; its key is
                still in ``idempotency_keys``.

        A ``create_many`` stopped part way after a call was sent, cancelled on
        :class:`~telem.AsyncTelem` or by a ``KeyboardInterrupt``, raises that exception
        unchanged, with ``results`` and ``idempotency_keys`` put on it.
        """
        steps = _create_many_steps(_batch_items(items, self._client.default_vertical))
        reply: Any = None
        while True:
            try:
                kind, value = (
                    steps.throw(reply) if isinstance(reply, BaseException) else steps.send(reply)
                )
            except StopIteration as stop:
                return stop.value
            try:
                if kind == "sleep":
                    reply = _sleep(value)
                else:
                    reply = self._client._post(_BATCH_PATH, value)
            except BaseException as exc:  # the steps keep the keys on it, then re-raise it
                reply = exc

    def list(
        self, *, limit: int | None = None, cursor: str | None = None, tag: str | None = None
    ) -> list[Monitor]:
        """List your monitors, newest first. :meth:`list_page` also gives the next cursor.

        Args:
            limit: How many to return (server default 50, maximum 200).
            cursor: The ``next_cursor`` of the page before.
            tag: Only your key's monitors that carry this one tag.
        """
        return self.list_page(limit=limit, cursor=cursor, tag=tag).monitors

    def list_page(
        self, *, limit: int | None = None, cursor: str | None = None, tag: str | None = None
    ) -> MonitorPage:
        """One page of your monitors, newest first, and the cursor of the next page.

        Args:
            limit: How many to return (server default 50, maximum 200).
            cursor: The ``next_cursor`` of the page before.
            tag: Only your key's monitors that carry this one tag. Page on with
                ``cursor`` and the same ``tag``.
        """
        params = _list_params(limit, cursor, tag)
        return _page_from(self._client._get("/v1/monitors", params=params))

    def get(self, monitor_id: str) -> Monitor:
        """Get one monitor."""
        return Monitor.model_validate(self._client._get(_path(monitor_id)))

    def update(
        self,
        monitor_id: str,
        *,
        status: str | None = None,
        frequency: str | None = None,
        webhook_url: str | None = None,
        ttl: str | None = None,
        tags: list[str] | None = None,
    ) -> Monitor:
        """Pause, resume, renew or change a monitor. Send at least one field.

        Args:
            monitor_id: The monitor id.
            status: ``"paused"`` or ``"active"``.
            frequency: A new frequency.
            webhook_url: A new webhook URL, with a new ``webhook_secret`` in the
                response. ``""`` removes the webhook.
            ttl: A new end, this long from now, such as ``"30d"`` (at most ``"30d"``).
                It renews a monitor before it ends; an ended monitor cannot be renewed.
            tags: The monitor's new tags, in place of all its tags (the rule is that of
                :meth:`create`). ``[]`` removes them all; leave it out to keep them.

        Returns:
            The updated :class:`Monitor`.
        """
        body = _update_body(status, frequency, webhook_url, ttl, tags)
        return Monitor.model_validate(self._client._request("PATCH", _path(monitor_id), body))

    def poll(
        self,
        monitor_id: str,
        *,
        session: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Get the monitor's own results that are new since the last poll.

        .. deprecated:: 0.4.0
            Read what the monitor agent finds with :meth:`iter_changes` or
            :meth:`changes`. A poll still works, and warns once per process
            (``DeprecationWarning``).

        Args:
            monitor_id: The monitor id.
            session: Record this poll in another session (default: the monitor's own).
            metadata: Request metadata (query lineage keys), as in a search.

        Returns:
            A :class:`SearchResponse`.
        """
        _warn_poll_deprecated()
        data = self._client._post(_path(monitor_id, "/poll"), _poll_body(session, metadata))
        return build_search_response(data)

    def agent(self, monitor_id: str) -> AgentDocument:
        """Get the monitor agent: its status, plan version and spend.

        Args:
            monitor_id: The monitor id.

        Returns:
            An :class:`AgentDocument`. Its ``raw`` has everything Telem sent.

        Raises:
            NotFoundError: The monitor has no agent, or it is not yours.
        """
        return _agent_from(self._client._get(_path(monitor_id, "/agent")))

    def changes(
        self,
        monitor_id: str,
        *,
        cursor: int | None = None,
        since: str | datetime | None = None,
        kinds: list[str] | None = None,
        limit: int | None = None,
    ) -> ChangesPage:
        """One page of what the monitor agent changed, oldest first.

        Keep the ``next_cursor`` of the page and pass it as ``cursor`` to read on: a
        change never comes twice, and none is skipped. You are caught up when
        ``next_cursor`` reaches ``watermark``, or when a page is empty.
        :meth:`iter_changes` does this for you.

        Args:
            monitor_id: The monitor id.
            cursor: Read the changes after this ``seq``. Without it, the read starts
                at the first change.
            since: Start at a time instead of a cursor, and not together with one. A
                ``datetime`` with no zone is local time. A string needs a zone, such as
                ``"2026-11-03T09:00:00Z"``.
            kinds: Only these kinds: ``entity_added``, ``value_changed``,
                ``fact_status_changed``, ``alert``, ``run`` or ``plan_changed``. Any
                other name is a ``BadRequestError``; an empty list keeps every kind.
            limit: How many changes to return at most (server default 200, maximum 500).

        Returns:
            A :class:`ChangesPage`.

        Raises:
            BadRequestError: Both ``cursor`` and ``since`` were sent (``cursor_conflict``),
                or ``kinds`` names a kind that is not one of the six (``kinds_invalid``).
            NotFoundError: The monitor has no agent, or it is not yours.
        """
        params = _changes_params(cursor, since, kinds, limit)
        data = self._client._get(_path(monitor_id, "/agent/changes"), params=params)
        return ChangesPage.model_validate(data)

    def iter_changes(self, monitor_id: str, *, cursor: int = 0) -> Iterator[ChangeRecord]:
        """Every change after ``cursor``, oldest first, page by page until you are caught up.

        It reads what is there now and then stops; it does not wait for more. To follow
        the agent, call it again later from the ``seq`` of the last change you handled::

            for change in client.monitors.iter_changes(monitor.id, cursor=saved):
                handle(change)
                saved = change.seq

        Args:
            monitor_id: The monitor id.
            cursor: Start after this ``seq``. ``0`` starts at the first change.

        Yields:
            :class:`ChangeRecord`, in the order of ``seq``.
        """
        while True:
            page = self.changes(monitor_id, cursor=cursor)
            yield from page.changes
            if not _has_more(page, cursor):
                return
            cursor = page.next_cursor

    def changes_across(
        self,
        tag: str | None = None,
        ids: str | list[str] | None = None,
        *,
        cursor: str | None = None,
        since: str | datetime | None = None,
        kinds: list[str] | None = None,
        limit: int | None = None,
    ) -> MonitorsChangesPage:
        """One page of the changes of up to 100 monitors, by ``seq``.

        Send ``tag`` or ``ids``, one of them. Keep the ``next_cursor`` of every page, the
        last one too, and pass it as ``cursor`` with the same ``tag`` or ``ids`` to read
        on. Read on now while ``more`` is true; ``pending`` monitors are read by a later
        call with the cursor. A later page can bring lower seqs: dedupe by ``seq``.

        Args:
            tag: Your key's own live monitors that carry this tag. A monitor with no agent
                is in ``skipped``.
            ids: These monitors, at most 100: any monitor with an agent in your key's
                project, ended ones too. One that is not is a 404 for the whole read.
            cursor: The ``next_cursor`` of the page before. Without it, the read starts at
                the first change.
            since: Start at a time instead of a cursor, and not together with one. A
                monitor that joins later starts at the same time. As in :meth:`changes`.
            kinds: Only these kinds, as in :meth:`changes`.
            limit: How many changes to return at most (server default 200, maximum 500).

        Returns:
            A :class:`MonitorsChangesPage`.

        Raises:
            ValueError: Neither or both of ``tag`` and ``ids``, or an empty ``ids``.
                Nothing is sent.
            BadRequestError: ``invalid_request``, ``cursor_conflict`` (``cursor`` and
                ``since``), ``cursor_invalid`` or ``kinds_invalid``.
            NotFoundError: A monitor of ``ids`` has no agent, or is not in your project.
            APIStatusError: A ``422``: ``too_many_monitors`` (more than 100; read them in
                groups of ``ids``), or a parameter is malformed.
        """
        params = _across_params(tag, ids, cursor, since, kinds, limit)
        return MonitorsChangesPage.model_validate(
            self._client._get("/v1/monitors/changes", params=params)
        )

    def iter_changes_across(
        self,
        tag: str | None = None,
        ids: str | list[str] | None = None,
        *,
        cursor: str | None = None,
        since: str | datetime | None = None,
        kinds: list[str] | None = None,
        limit: int | None = None,
    ) -> Iterator[ChangeRecord]:
        """The changes of up to 100 monitors, page by page while ``more`` is true.

        It reads what is there now in one pass and then stops. It cannot hand back the
        cursor: to keep your place between reads, call :meth:`changes_across` page by
        page and keep each ``next_cursor``. Dedupe by ``seq``. The arguments are those of
        :meth:`changes_across`; ``since`` is sent with the first page only.

        Yields:
            :class:`ChangeRecord`, as the router serves them.
        """
        while True:
            page = self.changes_across(
                tag, ids, cursor=cursor, since=since, kinds=kinds, limit=limit
            )
            yield from page.changes
            if not _reads_across_on(page, cursor):
                return
            cursor, since = page.next_cursor, None

    def onboarding(self, monitor_id: str) -> OnboardingSheet:
        """The onboarding sheet of each company of a monitor created with ``inputs``.

        A company's sheet says what the first read of its pages found: candidate kinds of
        items and of events, each with a count, the number of sites it was found on and an
        example quote with its page. Labels, names and quotes are text from those pages.

        Args:
            monitor_id: The monitor id.

        Returns:
            An :class:`OnboardingSheet`, one company each. A company is ``pending``,
            ``running`` or ``done``, and has its ``sheet`` once it is ``done``.

        Raises:
            APIStatusError: ``409`` (``not_an_engine_monitor``): the monitor was created
                without ``inputs``.
            NotFoundError: The monitor does not exist, or it is not yours.
        """
        return OnboardingSheet.model_validate(
            self._client._get(_path(monitor_id, "/agent/onboarding"))
        )

    def apply_onboarding(self, monitor_id: str, marks: dict[str, Any]) -> ApplyResult:
        """Apply your review of one company's onboarding sheet.

        The accepted candidates join what the monitor collects, and the company's pages are
        read again for them.

        Args:
            monitor_id: The monitor id.
            marks: The review, sent as given: ``company_id`` (required), and any of
                ``accept`` and ``reject`` (lists of candidate ids), ``rename`` (``{id:
                label}``) and ``merge`` (``{id: id}``).

        Returns:
            An :class:`ApplyResult`, with the ``job_id`` of the job that applies it.

        Raises:
            BadRequestError: ``marks`` names a candidate the sheet does not have.
            APIStatusError: ``409`` ``onboarding_pending`` (the sheet is not done yet) or
                ``compile_in_progress`` (the monitor is making its plan: try again soon),
                ``429`` (too many changes this hour: wait for ``Retry-After``). Nothing
                changed then.
        """
        data = self._client._post(_path(monitor_id, "/agent/onboarding:apply"), dict(marks))
        return ApplyResult.model_validate(data)

    def expansions(self, monitor_id: str, *, status: str | None = None) -> ExpansionsPage:
        """What a monitor created with ``inputs`` proposes to add from its own results, or
        has added: items, people, categories, sources and pages, newest first.

        Args:
            monitor_id: The monitor id.
            status: Only the records with this status: ``proposed``, ``active``,
                ``demoted``, ``retired`` or ``reverted``.

        Returns:
            An :class:`ExpansionsPage`.
        """
        path = _path(monitor_id, "/agent/expansions")
        return ExpansionsPage.model_validate(self._client._get(path, params=_status_params(status)))

    def confirm_expansion(self, monitor_id: str, expansion_id: str, **fields: Any) -> ApplyResult:
        """Use a proposed addition: an item becomes active, a category joins the schema.

        Args:
            monitor_id: The monitor id.
            expansion_id: The record's ``id``, from :meth:`expansions`.
            **fields: For an added item, its ``domain`` or ``handle``, if you know them.

        Returns:
            An :class:`ApplyResult`, with the ``job_id`` of the job that applies it.

        Raises:
            APIStatusError: ``409`` ``expansion_settled`` (it is reverted or demoted
                already) or ``compile_in_progress``, ``429``. Nothing changed then.
        """
        path = _expansion_path(monitor_id, expansion_id, "confirm")
        return ApplyResult.model_validate(self._client._post(path, _set_fields(fields)))

    def revert_expansion(self, monitor_id: str, expansion_id: str) -> ApplyResult:
        """Remove an addition: it is never proposed again, and the facts it brought are
        taken back.

        Args:
            monitor_id: The monitor id.
            expansion_id: The record's ``id``, from :meth:`expansions`.

        Returns:
            An :class:`ApplyResult`, with the ``job_id`` of the job that applies it.

        Raises:
            APIStatusError: ``409`` ``expansion_settled`` (it is reverted or demoted
                already) or ``compile_in_progress``, ``429``. Nothing changed then.
        """
        path = _expansion_path(monitor_id, expansion_id, "revert")
        return ApplyResult.model_validate(self._client._post(path, {}))

    def feedback(
        self, monitor_id: str, fact_id: str, verdict: str, *, category: str | None = None
    ) -> FeedbackResult:
        """Label one fact of the monitor.

        Args:
            monitor_id: The monitor id.
            fact_id: The fact's id, as a change names it.
            verdict: ``confirmed`` (it is right), ``refuted`` (it is wrong) or
                ``restored`` (a refuted fact counts again).
            category: With ``refuted``, on a monitor created with ``inputs``: the event
                category the fact belongs to instead, such as ``other_notable``.

        Returns:
            A :class:`FeedbackResult`: the fact's status now and, with ``category``, the
            ``job_id`` of the job that files it there.

        Raises:
            BadRequestError: ``category`` is not an event category of the monitor
                (``category_unknown``).
        """
        body = _feedback_body(fact_id, verdict, category)
        data = self._client._post(_path(monitor_id, "/agent/feedback"), body)
        return FeedbackResult.model_validate(data)

    def data_map(self, monitor_id: str) -> DataMap:
        """Every kind of item and category of event the monitor collects, the categories it
        found in its own results, and how it finds people.

        Args:
            monitor_id: The monitor id.

        Returns:
            A :class:`DataMap`. Its ``raw`` has everything Telem sent.
        """
        return DataMap.model_validate(self._client._get(_path(monitor_id, "/agent/data-map")))

    def companies(
        self, monitor_id: str, *, offset: int | None = None, limit: int | None = None
    ) -> CompanyList:
        """One page of the companies a monitor created with ``inputs`` follows, in rank
        order: the order they were given.

        Args:
            monitor_id: The monitor id.
            offset: How many companies to skip, from the top (server default 0).
            limit: How many companies to return (server default 100, maximum 1,000).

        Returns:
            A :class:`CompanyList`. Pass its ``next_offset`` as ``offset`` for the next
            page; it is ``None`` on the last page.

        Raises:
            APIStatusError: ``409`` (``not_an_engine_monitor``): the monitor was created
                without ``inputs``.
            NotFoundError: The monitor does not exist, it is not yours, or it was created
                with ``agent=False``.
        """
        path = _path(monitor_id, "/agent/companies")
        return CompanyList.model_validate(
            self._client._get(path, params=_page_params(offset, limit))
        )

    def delete(self, monitor_id: str) -> None:
        """Delete a monitor. A repeat deletes what a failed delete left."""
        self._client._request("DELETE", _path(monitor_id))


class AsyncMonitorsResource:
    """Asynchronous monitor operations bound to an :class:`~telem.async_client.AsyncTelem`."""

    def __init__(self, client: AsyncTelem) -> None:
        self._client = client

    async def create(
        self,
        query: str,
        *,
        frequency: str | None = None,
        metadata: dict[str, Any] | None = None,
        webhook_url: str | None = None,
        ttl: str | None = None,
        search: dict[str, Any] | None = None,
        include_domains: list[str] | None = None,
        agent: bool = True,
        inputs: dict[str, Any] | MonitorInputs | None = None,
        preset: str | None = None,
        tags: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> Monitor:
        """Create a monitor. It has the agent unless ``agent=False``, and with the agent
        ``ttl`` is required. Retried as :meth:`MonitorsResource.create` is."""
        body = _create_body(
            query,
            frequency,
            metadata,
            webhook_url,
            ttl,
            search,
            include_domains,
            agent,
            inputs,
            preset=preset,
            tags=tags,
            default_vertical=self._client.default_vertical,
        )
        headers = {"Idempotency-Key": resolve_idempotency_key(idempotency_key)}
        attempt, started = 0, _clock()
        while True:
            try:
                data = await self._client._post("/v1/monitors", body, headers=headers)
                return Monitor.model_validate(data)
            except TelemError as exc:
                wait = create_retry_wait(exc, attempt, _clock() - started)
                if wait is None:
                    if isinstance(exc, TelemTimeoutError):
                        raise TelemTimeoutError(f"{exc.message}. {_MAY_EXIST}") from exc
                    raise
            await _async_sleep(wait)
            attempt += 1

    async def create_many(self, items: Sequence[Mapping[str, Any]]) -> list[MonitorBatchResult]:
        """Create many monitors, ten per call. See :meth:`MonitorsResource.create_many`."""
        steps = _create_many_steps(_batch_items(items, self._client.default_vertical))
        reply: Any = None
        while True:
            try:
                kind, value = (
                    steps.throw(reply) if isinstance(reply, BaseException) else steps.send(reply)
                )
            except StopIteration as stop:
                return stop.value
            try:
                if kind == "sleep":
                    reply = await _async_sleep(value)
                else:
                    reply = await self._client._post(_BATCH_PATH, value)
            except BaseException as exc:  # the steps keep the keys on it, then re-raise it
                reply = exc

    async def list(
        self, *, limit: int | None = None, cursor: str | None = None, tag: str | None = None
    ) -> list[Monitor]:
        """List your monitors, newest first. See :meth:`MonitorsResource.list`."""
        return (await self.list_page(limit=limit, cursor=cursor, tag=tag)).monitors

    async def list_page(
        self, *, limit: int | None = None, cursor: str | None = None, tag: str | None = None
    ) -> MonitorPage:
        """One page of your monitors. See :meth:`MonitorsResource.list_page`."""
        params = _list_params(limit, cursor, tag)
        data = await self._client._get("/v1/monitors", params=params)
        return _page_from(data)

    async def get(self, monitor_id: str) -> Monitor:
        """Get one monitor."""
        return Monitor.model_validate(await self._client._get(_path(monitor_id)))

    async def update(
        self,
        monitor_id: str,
        *,
        status: str | None = None,
        frequency: str | None = None,
        webhook_url: str | None = None,
        ttl: str | None = None,
        tags: list[str] | None = None,
    ) -> Monitor:
        """Pause, resume, renew or change a monitor. See :meth:`MonitorsResource.update`."""
        body = _update_body(status, frequency, webhook_url, ttl, tags)
        data = await self._client._request("PATCH", _path(monitor_id), body)
        return Monitor.model_validate(data)

    async def poll(
        self,
        monitor_id: str,
        *,
        session: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Get the monitor's own results that are new since the last poll.

        .. deprecated:: 0.4.0
            Read the agent with :meth:`iter_changes` or :meth:`changes`. See
            :meth:`MonitorsResource.poll`.
        """
        _warn_poll_deprecated()
        body = _poll_body(session, metadata)
        return build_search_response(await self._client._post(_path(monitor_id, "/poll"), body))

    async def agent(self, monitor_id: str) -> AgentDocument:
        """Get the monitor agent. See :meth:`MonitorsResource.agent`."""
        return _agent_from(await self._client._get(_path(monitor_id, "/agent")))

    async def changes(
        self,
        monitor_id: str,
        *,
        cursor: int | None = None,
        since: str | datetime | None = None,
        kinds: list[str] | None = None,
        limit: int | None = None,
    ) -> ChangesPage:
        """One page of what the monitor agent changed. See :meth:`MonitorsResource.changes`."""
        params = _changes_params(cursor, since, kinds, limit)
        data = await self._client._get(_path(monitor_id, "/agent/changes"), params=params)
        return ChangesPage.model_validate(data)

    async def iter_changes(
        self, monitor_id: str, *, cursor: int = 0
    ) -> AsyncIterator[ChangeRecord]:
        """Every change after ``cursor``, until you are caught up.

        Use it with ``async for``. See :meth:`MonitorsResource.iter_changes`.
        """
        while True:
            page = await self.changes(monitor_id, cursor=cursor)
            for change in page.changes:
                yield change
            if not _has_more(page, cursor):
                return
            cursor = page.next_cursor

    async def changes_across(
        self,
        tag: str | None = None,
        ids: str | list[str] | None = None,
        *,
        cursor: str | None = None,
        since: str | datetime | None = None,
        kinds: list[str] | None = None,
        limit: int | None = None,
    ) -> MonitorsChangesPage:
        """One page of the changes of up to 100 monitors. See
        :meth:`MonitorsResource.changes_across`."""
        params = _across_params(tag, ids, cursor, since, kinds, limit)
        return MonitorsChangesPage.model_validate(
            await self._client._get("/v1/monitors/changes", params=params)
        )

    async def iter_changes_across(
        self,
        tag: str | None = None,
        ids: str | list[str] | None = None,
        *,
        cursor: str | None = None,
        since: str | datetime | None = None,
        kinds: list[str] | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[ChangeRecord]:
        """The changes of up to 100 monitors, while ``more`` is true.

        Use it with ``async for``. See :meth:`MonitorsResource.iter_changes_across`.
        """
        while True:
            page = await self.changes_across(
                tag, ids, cursor=cursor, since=since, kinds=kinds, limit=limit
            )
            for change in page.changes:
                yield change
            if not _reads_across_on(page, cursor):
                return
            cursor, since = page.next_cursor, None

    async def onboarding(self, monitor_id: str) -> OnboardingSheet:
        """The onboarding sheet of each company. See :meth:`MonitorsResource.onboarding`."""
        data = await self._client._get(_path(monitor_id, "/agent/onboarding"))
        return OnboardingSheet.model_validate(data)

    async def apply_onboarding(self, monitor_id: str, marks: dict[str, Any]) -> ApplyResult:
        """Apply your review of one company's sheet. See
        :meth:`MonitorsResource.apply_onboarding`."""
        path = _path(monitor_id, "/agent/onboarding:apply")
        return ApplyResult.model_validate(await self._client._post(path, dict(marks)))

    async def expansions(self, monitor_id: str, *, status: str | None = None) -> ExpansionsPage:
        """What the monitor proposes to add, or added. See :meth:`MonitorsResource.expansions`."""
        path = _path(monitor_id, "/agent/expansions")
        data = await self._client._get(path, params=_status_params(status))
        return ExpansionsPage.model_validate(data)

    async def confirm_expansion(
        self, monitor_id: str, expansion_id: str, **fields: Any
    ) -> ApplyResult:
        """Use a proposed addition. See :meth:`MonitorsResource.confirm_expansion`."""
        path = _expansion_path(monitor_id, expansion_id, "confirm")
        return ApplyResult.model_validate(await self._client._post(path, _set_fields(fields)))

    async def revert_expansion(self, monitor_id: str, expansion_id: str) -> ApplyResult:
        """Remove an addition. See :meth:`MonitorsResource.revert_expansion`."""
        path = _expansion_path(monitor_id, expansion_id, "revert")
        return ApplyResult.model_validate(await self._client._post(path, {}))

    async def feedback(
        self, monitor_id: str, fact_id: str, verdict: str, *, category: str | None = None
    ) -> FeedbackResult:
        """Label one fact of the monitor. See :meth:`MonitorsResource.feedback`."""
        body = _feedback_body(fact_id, verdict, category)
        data = await self._client._post(_path(monitor_id, "/agent/feedback"), body)
        return FeedbackResult.model_validate(data)

    async def data_map(self, monitor_id: str) -> DataMap:
        """Every kind and category the monitor collects. See :meth:`MonitorsResource.data_map`."""
        data = await self._client._get(_path(monitor_id, "/agent/data-map"))
        return DataMap.model_validate(data)

    async def companies(
        self, monitor_id: str, *, offset: int | None = None, limit: int | None = None
    ) -> CompanyList:
        """One page of the monitor's companies. See :meth:`MonitorsResource.companies`."""
        path = _path(monitor_id, "/agent/companies")
        data = await self._client._get(path, params=_page_params(offset, limit))
        return CompanyList.model_validate(data)

    async def delete(self, monitor_id: str) -> None:
        """Delete a monitor. A repeat deletes what a failed delete left."""
        await self._client._request("DELETE", _path(monitor_id))
