"""Monitors: a query that Telem runs on a schedule, with an agent that keeps what matches.

A monitor has the agent unless it is created with ``agent=False``, and a monitor with the
agent must end, so its create needs a ``ttl``. Read what the agent finds with ``changes`` or
``iter_changes``. ``poll`` reads a monitor's own results and is deprecated.

A monitor created with ``inputs`` watches a list of companies: ``onboarding`` reads what
the first read of each company found and ``apply_onboarding`` applies your review of it,
``expansions`` lists what the monitor proposes to add (``confirm_expansion``,
``revert_expansion``), and ``data_map`` says what it collects. ``feedback`` labels a fact
of any monitor with an agent."""

from __future__ import annotations

import hashlib
import hmac
import time
import warnings
from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any

from telem._base import _iso, path_segment
from telem._normalize import build_search_response
from telem.errors import TelemTimeoutError
from telem.models import (
    AgentDocument,
    ApplyResult,
    ChangeRecord,
    ChangesPage,
    DataMap,
    ExpansionsPage,
    FeedbackResult,
    Monitor,
    MonitorInputs,
    MonitorPage,
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
    body["agent"] = agent_on
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
) -> dict[str, Any]:
    body = {"status": status, "frequency": frequency, "ttl": ttl}
    body = {key: value for key, value in body.items() if value is not None}
    if webhook_url is not None:
        body["webhook_url"] = webhook_url or None  # "" removes the webhook
    return body


def verify_webhook(body: bytes | str, header: str, secret: str, *, tolerance: int = 300) -> bool:
    """Check a monitor webhook: the ``Telem-Signature`` header against the raw body.

    Args:
        body: The raw request body, exactly as received.
        header: The ``Telem-Signature`` header.
        secret: The monitor's ``webhook_secret``.
        tolerance: The oldest signature to accept, in seconds.

    Returns:
        True when the signature is correct and not older than ``tolerance``.
    """
    parts = dict(item.split("=", 1) for item in header.split(",") if "=" in item)
    timestamp, signature = parts.get("t", ""), parts.get("v1", "")
    if not timestamp.isdigit() or abs(time.time() - int(timestamp)) > tolerance:
        return False
    raw = body.encode() if isinstance(body, str) else body
    expected = hmac.new(secret.encode(), f"{timestamp}.".encode() + raw, hashlib.sha256)
    return hmac.compare_digest(expected.hexdigest(), signature)


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


def _list_params(limit: int | None, cursor: str | None) -> dict[str, Any] | None:
    params: dict[str, Any] = {}
    if limit is not None:
        params["limit"] = limit
    if cursor is not None:
        params["cursor"] = cursor
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


#: Added to a create that timed out: the router may have made the monitor anyway.
_MAY_EXIST = "The monitor may exist: list monitors before you create it again."


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

        Returns:
            The new :class:`Monitor`. Its ``agent_state`` shows the agent. When the agent
            did not start, ``agent_state`` is ``None`` and ``agent_note`` says why. With
            ``webhook_url``, its ``webhook_secret`` signs the deliveries; Telem shows it
            only in this response.

        Raises:
            ValueError: The agent is on and ``ttl`` is missing. Nothing was sent.
        """
        body = _create_body(
            query, frequency, metadata, webhook_url, ttl, search, include_domains, agent, inputs
        )
        try:
            data = self._client._post("/v1/monitors", body)
        except TelemTimeoutError as exc:
            raise TelemTimeoutError(f"{exc.message}. {_MAY_EXIST}") from exc
        return Monitor.model_validate(data)

    def list(self, *, limit: int | None = None, cursor: str | None = None) -> list[Monitor]:
        """List your monitors, newest first. :meth:`list_page` also gives the next cursor.

        Args:
            limit: How many to return (server default 50, maximum 200).
            cursor: The ``next_cursor`` of the page before.
        """
        return self.list_page(limit=limit, cursor=cursor).monitors

    def list_page(self, *, limit: int | None = None, cursor: str | None = None) -> MonitorPage:
        """One page of your monitors, newest first, and the cursor of the next page.

        Args:
            limit: How many to return (server default 50, maximum 200).
            cursor: The ``next_cursor`` of the page before.
        """
        return _page_from(self._client._get("/v1/monitors", params=_list_params(limit, cursor)))

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

        Returns:
            The updated :class:`Monitor`.
        """
        body = _update_body(status, frequency, webhook_url, ttl)
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
    ) -> Monitor:
        """Create a monitor. It has the agent unless ``agent=False``, and with the agent
        ``ttl`` is required. See :meth:`MonitorsResource.create`."""
        body = _create_body(
            query, frequency, metadata, webhook_url, ttl, search, include_domains, agent, inputs
        )
        try:
            data = await self._client._post("/v1/monitors", body)
        except TelemTimeoutError as exc:
            raise TelemTimeoutError(f"{exc.message}. {_MAY_EXIST}") from exc
        return Monitor.model_validate(data)

    async def list(self, *, limit: int | None = None, cursor: str | None = None) -> list[Monitor]:
        """List your monitors, newest first. See :meth:`MonitorsResource.list`."""
        return (await self.list_page(limit=limit, cursor=cursor)).monitors

    async def list_page(
        self, *, limit: int | None = None, cursor: str | None = None
    ) -> MonitorPage:
        """One page of your monitors. See :meth:`MonitorsResource.list_page`."""
        data = await self._client._get("/v1/monitors", params=_list_params(limit, cursor))
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
    ) -> Monitor:
        """Pause, resume, renew or change a monitor. See :meth:`MonitorsResource.update`."""
        body = _update_body(status, frequency, webhook_url, ttl)
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

    async def delete(self, monitor_id: str) -> None:
        """Delete a monitor. A repeat deletes what a failed delete left."""
        await self._client._request("DELETE", _path(monitor_id))
