"""Pydantic v2 models for the Telem SDK public API."""

from __future__ import annotations

import logging
from typing import Any, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


def _holds_models(annotation: Any) -> bool:
    """Whether a field's type has a model in it (``Foo``, ``list[Foo]``, ``Foo | None``)."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True
    return any(_holds_models(arg) for arg in get_args(annotation))


def _tolerant(cls: type[BaseModel], data: Any, handler: ModelWrapValidatorHandler, raw: bool):
    """``handler(data)``; on a ValidationError, the record again without the optional fields
    that failed (each takes its default), logged at debug: one malformed item never fails a
    whole list. A required field that fails still raises. With ``raw``, ``raw`` holds the
    record as sent, the dropped fields too."""
    extra = {"raw": data} if raw and isinstance(data, dict) else {}
    try:
        return handler({**data, **extra} if extra else data)
    except ValidationError as exc:
        if not isinstance(data, dict):
            raise
        by_key = {info.alias or name: info for name, info in cls.model_fields.items()}
        bad: set[str] = set()
        for error in exc.errors():
            loc = error.get("loc") or ()
            if not loc:
                continue
            key = str(loc[0])
            info = by_key.get(key)
            # An error deeper inside a field that holds models is a nested model's own: it
            # left its optional fields unset already, so what is left (a required field, a
            # non-object item) raises as before rather than emptying a list or a sub-record.
            if len(loc) > 1 and info is not None and _holds_models(info.annotation):
                raise
            bad.add(key)
        required = {key for key, info in by_key.items() if info.is_required()}
        if not bad or bad & required or "raw" in bad:
            raise
        logger.debug("telem: %s dropped malformed fields %s", cls.__name__, sorted(bad))
        return handler({**{k: v for k, v in data.items() if k not in bad}, **extra})


class _Tolerant(BaseModel):
    """A list item: a malformed optional field is left unset on this item only."""

    @model_validator(mode="wrap")
    @classmethod
    def _drop_malformed(cls, data: Any, handler: ModelWrapValidatorHandler) -> Any:
        return _tolerant(cls, data, handler, raw=False)


class SearchResult(BaseModel):
    """A single normalized search result row from a provider envelope."""

    model_config = ConfigDict(extra="ignore")

    url: str = ""
    title: str = ""
    summary: str | None = None
    excerpt: list[str] | None = None
    full_content: dict[str, Any] | None = None
    publish_date: str | None = None
    rank: int | None = None
    thumbnail: str | None = None
    favicon: str | None = None
    source: dict[str, Any] | None = None
    enrichments: dict[str, Any] | None = None
    fetch_meta: dict[str, Any] | None = None
    # Image searches (``topic="images"``) only: the original image, its width and height.
    image_url: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    provider: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def content(self) -> str:
        """Legacy alias: ``summary``, else ``full_content["content"]``, else ``""``.

        Not a stored field — it never appears in :meth:`model_dump`.
        """
        return self.summary or (self.full_content or {}).get("content") or ""


class ProviderRun(BaseModel):
    """Results and status for one provider (preprocessor run) in an interaction."""

    model_config = ConfigDict(extra="ignore")

    provider: str
    status: str
    results: list[SearchResult] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    latency_ms: int | None = None
    preprocessor_run_id: str | None = None
    tier: str | None = None
    fields: list[str] = Field(default_factory=list)
    query: str = ""
    batch_index: int = 0
    answer: str | None = None
    entities: dict[str, Any] | list[Any] | None = None
    # {"questions": [...], "searches": [...]} — a DICT per server v1 (caught live: serpapi)
    related: dict[str, Any] | None = None
    verticals: dict[str, Any] | None = None
    usage: dict[str, Any] | list[Any] | None = None
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    # Monitor polls: results folded as the same story, {lead url: [{url, title}]}
    same_story: dict[str, list[dict[str, Any]]] | None = None
    raw: dict[str, Any] | None = None


class SearchResponse(BaseModel):
    """The result of a :meth:`Telem.search` call.

    ``results`` is the flattened list across all providers; ``by_provider`` preserves the
    per-provider breakdown including any partial failures.
    """

    model_config = ConfigDict(extra="ignore")

    results: list[SearchResult] = Field(default_factory=list)
    by_provider: list[ProviderRun] = Field(default_factory=list)
    session_id: str
    interaction_id: str
    status: str
    normalized_schema_version: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class FetchResult(BaseModel):
    """One fetched URL's outcome from a :meth:`Telem.fetch` call.

    Fields mirror the server's ``fetched_results`` row; ``status``/``error`` fall back
    to the run row for URLs whose fetch failed before producing a row.
    """

    model_config = ConfigDict(extra="ignore")

    url: str = ""
    canonical_url: str | None = None
    title: str = ""
    status: str = ""
    content: str | None = None
    content_truncated: bool | None = None
    content_format: str | None = None
    content_sha256: str | None = None
    content_type: str | None = None
    content_ref: dict[str, Any] | None = None
    provider: str = ""
    http_status: int | None = None
    fetched_at: str | None = None
    latency_ms: int | None = None
    provider_metadata: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    batch_index: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)


class FetchResponse(BaseModel):
    """The result of a :meth:`Telem.fetch` call.

    One :class:`FetchResult` per requested URL, ordered by the request's URL order
    (``batch_index``). Fetch responses carry no ``normalized_schema_version`` echo.
    """

    model_config = ConfigDict(extra="ignore")

    results: list[FetchResult] = Field(default_factory=list)
    session_id: str
    interaction_id: str
    status: str
    raw: dict[str, Any] = Field(default_factory=dict)


class ProviderInfo(_Tolerant):
    """Metadata describing an available search provider (preprocessor)."""

    model_config = ConfigDict(extra="ignore")

    name: str
    type: str = ""
    active_by_default: bool = False
    description: str | None = None
    method: str | None = None
    url: str | None = None
    normalized: bool = False
    tiers: list[str] = Field(default_factory=list)
    #: Whether auto-routing can pick it. ``None`` from a router that does not say.
    auto_routed: bool | None = None


class SessionSummary(_Tolerant):
    """Summary metadata for a session listed by :meth:`SessionsResource.list`."""

    model_config = ConfigDict(extra="ignore")

    id: str
    created_at: str | None = None
    updated_at: str | None = None
    interaction_count: int = 0
    latest_interaction_at: str | None = None


class SessionHistory(BaseModel):
    """Interaction history for a session (pass-through of backend records)."""

    model_config = ConfigDict(extra="ignore")

    session_id: str
    instance_id: str | None = None
    interactions: list[dict[str, Any]] = Field(default_factory=list)


class SessionResults(BaseModel):
    """Aggregated websearch preprocessor results for a session."""

    model_config = ConfigDict(extra="ignore")

    session_id: str
    query: str = ""
    goal: str = ""
    context: str = ""
    previous_preprocessor_results: list[dict[str, Any]] = Field(default_factory=list)


class AgentCompileState(_Tolerant):
    """The job that makes the agent's plan: ``AgentState.compile``."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    status: str
    #: Why the plan could not be made; ``None`` when it did not fail.
    error: str | None = None


class AgentState(_Tolerant):
    """What the monitor agent is doing: ``Monitor.agent_state``."""

    model_config = ConfigDict(extra="ignore")

    #: The agent engine: ``"v4"``, or ``"engine"`` for a monitor created with ``inputs``.
    engine: str
    #: Such as ``compiling``, ``ready``, ``paused`` or ``failed``.
    status: str
    #: ``True`` when the agent needs you to look at it, for example after it failed.
    needs_attention: bool = False
    #: The version of the agent's plan; ``None`` until the first plan is made.
    plan_version: int | None = None
    #: When the agent runs next, as an ISO 8601 time; ``None`` when it is not scheduled.
    next_run_at: str | None = None
    #: The ``seq`` of the newest change you can read. ``0``: there is none yet.
    watermark: int = 0
    #: The latest job that made the agent's plan; ``None`` when there is none.
    compile: AgentCompileState | None = None


class Monitor(_Tolerant):
    """A monitor: a query that Telem runs on a schedule."""

    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    id: str
    query: str
    frequency: str
    #: ``active``, ``paused`` (you paused it) or ``ended`` (past ``expires_at``).
    status: str
    session_id: str
    created_at: str
    last_polled_at: str | None = None
    #: When the monitor ends; ``None``: never.
    expires_at: str | None = None
    webhook_url: str | None = None
    #: Signs the webhook deliveries. Only in the response that set ``webhook_url``. Not in
    #: ``repr()``.
    webhook_secret: str | None = Field(default=None, repr=False)
    #: The monitor agent; ``None`` for a monitor without one (``agent=False``), and from
    #: a router that does not say.
    agent_state: AgentState | None = None
    #: Only in the answer to a create: why the monitor agent did not start, or not yet.
    #: ``None`` when it started, and in every other answer.
    agent_note: str | None = None
    #: Your labels for the monitor; empty when it has none.
    tags: list[str] = Field(default_factory=list)


# The page models with cursors (MonitorPage, ChangesPage, MonitorsChangesPage) stay strict: a
# malformed cursor that read as its default would end pagination silently. Their items are
# _Tolerant.
class MonitorPage(BaseModel):
    """One page of :meth:`~telem.resources.monitors.MonitorsResource.list_page`."""

    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    monitors: list[Monitor] = []
    #: Pass as ``cursor`` to get the next page; ``None`` on the last page.
    next_cursor: str | None = None


class MonitorBatchResult(_Tolerant):
    """One monitor of :meth:`~telem.resources.monitors.MonitorsResource.create_many`."""

    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    #: The monitor's position in the list you passed.
    index: int
    #: ``201``: made. ``200`` with ``replayed``: made before, under the same key. Any
    #: other status is this monitor's own error.
    status: int
    monitor: Monitor | None = None
    replayed: bool = False
    #: ``{"code", "message"}``, and ``"param"`` when a field is at fault.
    error: dict[str, Any] | None = None
    #: With a ``429``: the seconds to wait before you send this monitor again.
    retry_after: int | None = None
    #: The key this monitor was sent with: send it again with the same key.
    idempotency_key: str


class FetchNewResults(BaseModel):
    """Whether the monitor agent fetches each run's new search results before it reads
    them (``on``), and at most how many pages a run (``cap_per_run``)."""

    model_config = ConfigDict(extra="ignore")

    on: bool = Field(strict=True)
    cap_per_run: int = Field(strict=True)


class AgentDocument(BaseModel):
    """The agent of a monitor: :meth:`~telem.resources.monitors.MonitorsResource.agent`.

    The router sends more than these fields, such as the plan and the counts. All of it
    is in ``raw``. The document has no ``watermark``: a :class:`ChangesPage` has it, and
    so does ``Monitor.agent_state``.
    """

    model_config = ConfigDict(extra="ignore")

    #: Such as ``compiling``, ``ready``, ``paused`` or ``failed``.
    status: str | None = None
    query: str | None = None
    frequency: str | None = None
    #: The version of the agent's plan; ``0`` until the first plan is made.
    spec_version: int | None = None
    #: The agent's spending limits; ``None`` when none are set.
    budget: dict[str, Any] | None = None
    #: What the agent has spent so far, in US dollars.
    spend_usd: float | None = None
    #: The plan's ``fetch_new_results``; ``None`` before the first plan, or when the router
    #: does not send it.
    fetch_new_results: FetchNewResults | None = None
    #: The whole document, as the router sent it.
    raw: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _plan_fetch_new(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "spec" not in data:
            return data
        spec = data["spec"]
        held = spec.get("fetch_new_results") if isinstance(spec, dict) else None
        try:
            found = FetchNewResults.model_validate(held) if isinstance(held, dict) else None
        except ValidationError:
            found = None
        return {**data, "fetch_new_results": found}


class ChangeSubject(_Tolerant):
    """The item a change is about.

    ``entity_id`` is ``None`` for a change to the whole monitor, such as a run or a plan
    change.
    """

    model_config = ConfigDict(extra="ignore")

    entity_id: str | None = None
    name: str | None = None


class ChangeRecord(_Tolerant):
    """One change of the monitor agent: the records of :class:`ChangesPage`."""

    model_config = ConfigDict(extra="ignore")

    #: Where the change is in the stream. It only grows. Pass the last one you handled
    #: as ``cursor`` to read on from there.
    seq: int
    #: The agent's own name for the change, such as ``new_object`` or ``tick``.
    type: str
    #: ``entity_added``, ``value_changed``, ``fact_status_changed``, ``alert``, ``run``
    #: or ``plan_changed``.
    kind: str
    monitor_id: str
    subject: ChangeSubject = Field(default_factory=ChangeSubject)
    #: The value before and after the change. Any JSON, and ``None`` when there is none.
    before: Any = None
    after: Any = None
    fact_ids: list[str] = Field(default_factory=list)
    #: When the change happened, as an ISO 8601 time.
    at: str


class ChangesPage(BaseModel):
    """One page of :meth:`~telem.resources.monitors.MonitorsResource.changes`."""

    model_config = ConfigDict(extra="ignore")

    #: Oldest first.
    changes: list[ChangeRecord] = Field(default_factory=list)
    #: Keep it, and pass it as ``cursor`` to read on. On a full page (``limit`` changes) it
    #: is the ``seq`` of the last change. On a shorter page it is ``watermark``: the read
    #: got that far, past the changes that ``kinds`` left out too.
    next_cursor: int
    #: The ``seq`` of the newest change you can read now. You are caught up when
    #: ``next_cursor`` reaches it.
    watermark: int


class MonitorsChangesPage(BaseModel):
    """One page of :meth:`~telem.resources.monitors.MonitorsResource.changes_across`.

    Ordered by ``seq``, but a monitor that joins later or catches up can bring lower seqs
    on a later page: dedupe by ``seq``.
    """

    model_config = ConfigDict(extra="ignore")

    changes: list[ChangeRecord] = Field(default_factory=list)
    #: Keep it, and pass it as ``cursor`` to read on. It is opaque, and holds the place of
    #: every monitor of the read, the ``pending`` ones too.
    next_cursor: str
    #: ``True`` only for a full page: read on now with ``next_cursor``.
    more: bool = False
    #: The monitors read for this page.
    monitors: list[str] = Field(default_factory=list)
    #: The monitors not read on this call. A later call with ``next_cursor`` reads them.
    pending: list[str] = Field(default_factory=list)
    #: The monitors of the tag that have no agent.
    skipped: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# A monitor created with ``inputs``: a list of companies, and a schema that says what to
# collect about them.
# --------------------------------------------------------------------------- #


class MonitorInputs(BaseModel):
    """What a monitor of a list of companies collects: the ``inputs`` of
    :meth:`~telem.resources.monitors.MonitorsResource.create`.

    A field is sent only when it is set, and the router checks every value. A field this
    model does not name is sent too, as given. Rows are plain dicts with the router's own
    field names.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    #: The companies, 1 to 20: rows of ``company_id``, ``name`` and ``domain``.
    companies: list[dict[str, Any]] | None = None
    #: People to watch: rows of ``person_id``, ``name``, ``company_id`` and ``company``.
    roster: list[dict[str, Any]] | None = None
    #: What to collect: the preset ``"biopharma-v0"``, or a schema object. It is
    #: ``schema`` on the wire, and the constructor takes either name.
    schema_: str | dict[str, Any] | None = Field(default=None, alias="schema")
    #: Items to watch from the start: rows of ``entity_id``, ``kind`` and ``name``.
    entities: list[dict[str, Any]] | None = None
    #: Kinds of event to collect: rows with a ``label``.
    categories: list[dict[str, Any]] | None = None
    #: ``"propose"``: people found on company pages are proposed, for you to confirm.
    #: ``"auto"``: they are watched at once.
    discover: str | None = None
    #: How many days back the first read of each company looks.
    window_days: int | None = None
    #: Look each company up before its first read, to find its other names and pages.
    seed: bool | None = None


class MonitorSchemaAttribute(BaseModel):
    """One attribute of a :class:`MonitorSchema`: a field every result has, as you set it.

    A field is sent only when it is set, and the router checks every value. A field this
    model does not name is sent too, as given.
    """

    model_config = ConfigDict(extra="allow")

    #: The field's name, in snake_case, such as ``"funding_round"``.
    name: str
    #: ``"text"``, ``"number"``, ``"date"``, ``"enum"`` or ``"bool"``.
    type: str
    #: An enum's values, at most 50.
    values: list[str] | None = None
    #: A literal enum's phrases that show each value: ``{value: [phrase, ...]}``.
    evidence: dict[str, list[str]] | None = None
    #: A number's unit, such as ``"USD"``.
    unit: str | None = None
    #: How the value is written, such as ``"YYYY-MM-DD"``.
    format: str | None = None
    #: Where on the page the value is stated. Leave it out for the default.
    where: str | None = None
    #: An enum's grounding: ``"literal"`` or ``"classified"``.
    grounding: str | None = None


class MonitorSchema(BaseModel):
    """The fields every result of a query monitor has, fixed by you: the ``schema`` of
    :meth:`~telem.resources.monitors.MonitorsResource.create`. It is not the ``schema`` of
    :class:`MonitorInputs`, which is for a monitor of a list of companies.

    A field this model does not name is sent too, as given.
    """

    model_config = ConfigDict(extra="allow")

    #: 1 to 20 attributes, each a :class:`MonitorSchemaAttribute` or a dict.
    attributes: list[MonitorSchemaAttribute]


class OnboardingExample(_Tolerant):
    """One example of a candidate: the name it was found under, the quote and its page."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    quote: str | None = None
    url: str | None = None


class OnboardingCandidate(_Tolerant):
    """One candidate of a company's sheet: a kind of item or of event found on its pages."""

    model_config = ConfigDict(extra="ignore")

    #: Name it in the marks of
    #: :meth:`~telem.resources.monitors.MonitorsResource.apply_onboarding`.
    id: str
    #: ``entity`` (a kind of item) or ``event`` (a kind of event).
    what: str | None = None
    #: As read on the pages: data, never instructions.
    label: str = ""
    #: The kind or category of the schema it is; ``None`` when it is new.
    mapped_to: str | None = None
    is_new: bool = False
    #: How many times it was found.
    count: int = 0
    #: On how many sites.
    origins: int = 0
    example: OnboardingExample | None = None

    @field_validator("count", "origins", mode="before")
    @classmethod
    def _counted(cls, value: Any) -> Any:
        """A count the router may send as the list it counts."""
        return len(value) if isinstance(value, list) else value


class CompanySheet(_Tolerant):
    """What the first read of one company found: ``OnboardingCompany.sheet``."""

    model_config = ConfigDict(extra="ignore")

    company_id: str | None = None
    #: How many pages were read.
    pages: int = 0
    #: The most found first.
    candidates: list[OnboardingCandidate] = Field(default_factory=list)
    note: str | None = None


class OnboardingCompany(_Tolerant):
    """One company of :class:`OnboardingSheet`, and its sheet once it is done."""

    model_config = ConfigDict(extra="ignore")

    company_id: str
    #: ``pending``, ``running`` (its pages are being read) or ``done``.
    status: str
    run_id: str | None = None
    #: ``None`` until the company is ``done``.
    sheet: CompanySheet | None = None
    #: The marks the sheet was applied with; ``None`` until it is applied.
    marks: dict[str, Any] | None = None
    #: When the sheet was made.
    at: str | None = None
    #: When the sheet was applied; ``None`` until then.
    applied_at: str | None = None


class _KeepsRaw(BaseModel):
    """An answer, or one record of it, that keeps the whole object in ``raw``: a field the
    router adds is readable there before this SDK names it."""

    model_config = ConfigDict(extra="ignore")

    #: The whole object, as the router sent it.
    raw: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="wrap")
    @classmethod
    def _keep_raw(cls, data: Any, handler: ModelWrapValidatorHandler) -> Any:
        return _tolerant(cls, data, handler, raw=True)


class OnboardingSheet(_KeepsRaw):
    """The onboarding sheets of a monitor's companies, one per company:
    :meth:`~telem.resources.monitors.MonitorsResource.onboarding`."""

    model_config = ConfigDict(extra="ignore")

    monitor_id: str | None = None
    companies: list[OnboardingCompany] = Field(default_factory=list)


class ApplyResult(_KeepsRaw):
    """A change Telem accepted: the job that makes it. The answer to
    :meth:`~telem.resources.monitors.MonitorsResource.apply_onboarding`,
    ``confirm_expansion`` and ``revert_expansion``."""

    model_config = ConfigDict(extra="ignore")

    job_id: str


class ExpansionRecord(_KeepsRaw):
    """One thing a monitor proposed to add, or added: the records of :class:`ExpansionsPage`."""

    model_config = ConfigDict(extra="ignore")

    #: Pass it to ``confirm_expansion`` or ``revert_expansion``.
    id: str
    #: Such as ``entity``, ``person``, ``category``, ``source`` or ``watch``.
    kind: str
    #: What it adds: an item, a person, a category, a site or a page.
    target: str
    #: ``proposed``, ``active``, ``demoted``, ``retired`` or ``reverted``.
    status: str
    reason: dict[str, Any] = Field(default_factory=dict)
    delta: dict[str, Any] = Field(default_factory=dict)
    #: The counts behind the proposal.
    counts: dict[str, Any] = Field(default_factory=dict)
    #: One line that says what it is and why.
    note: str | None = None
    run_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ExpansionsPage(_KeepsRaw):
    """One page of :meth:`~telem.resources.monitors.MonitorsResource.expansions`."""

    model_config = ConfigDict(extra="ignore")

    #: The newest first.
    expansions: list[ExpansionRecord] = Field(default_factory=list)
    next_cursor: str | None = None


class CompanyListRow(_KeepsRaw):
    """One company of :class:`CompanyList`."""

    model_config = ConfigDict(extra="ignore")

    company_id: str
    name: str = ""
    domain: str = ""
    #: Whether the company is marked priority: it is checked on every routine run.
    priority: bool = False
    aliases: list[str] = Field(default_factory=list)
    ticker: str = ""
    x_handle: str = ""
    newsroom_url: str = ""
    team_page: str = ""
    #: ``daily``, ``weekly`` or ``off``, as you gave it; empty when its rank decides.
    tier: str = ""
    #: Its place among the monitor's companies, from 1.
    rank: int = 0


class CompanyList(_KeepsRaw):
    """One page of :meth:`~telem.resources.monitors.MonitorsResource.companies`, in rank
    order."""

    model_config = ConfigDict(extra="ignore")

    #: All the companies the monitor follows.
    total: int = 0
    #: How many of them are marked priority.
    priority_count: int = 0
    offset: int = 0
    limit: int = 0
    #: Pass it as ``offset`` for the next page; ``None`` on the last page.
    next_offset: int | None = None
    companies: list[CompanyListRow] = Field(default_factory=list)


class WebhookEndpoint(_KeepsRaw):
    """One of the project's webhook endpoints. Its secret is shown only by
    :class:`WebhookEndpointCreated` and :class:`WebhookEndpointRotated`."""

    model_config = ConfigDict(extra="ignore")

    id: str
    url: str
    #: The monitor tags it takes deliveries for, any one of them; ``[]`` takes every monitor.
    tags: list[str] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None


class WebhookEndpointCreated(WebhookEndpoint):
    """The answer to a create: the endpoint and its secret, shown only here and on rotate."""

    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    #: ``whsec_…``: check each delivery's ``Telem-Signature`` with it. Kept out of ``repr``
    #: and ``str``, as is ``raw``, which holds it too.
    secret: str = Field(repr=False)
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)


class WebhookEndpointRotated(_KeepsRaw):
    """The answer to a rotate: the new secret. The one it replaces still signs until
    ``previous_until``."""

    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    id: str
    #: Kept out of ``repr`` and ``str``, as is ``raw``, which holds it too.
    secret: str = Field(repr=False)
    previous_until: str
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)


class FeedbackResult(_KeepsRaw):
    """The labelled fact's status now: :meth:`~telem.resources.monitors.MonitorsResource.feedback`."""

    model_config = ConfigDict(extra="ignore")

    #: ``verified``, ``refuted`` or ``extracted``.
    status: str
    #: The job that files the fact under ``category``; ``None`` without one.
    job_id: str | None = None


class DataMap(_KeepsRaw):
    """Every kind and category a monitor collects:
    :meth:`~telem.resources.monitors.MonitorsResource.data_map`."""

    model_config = ConfigDict(extra="ignore")

    #: One per kind of item, such as ``{"kind": "program", "attributes": [...]}``.
    kinds: list[dict[str, Any]] = Field(default_factory=list)
    #: The categories of events it collects.
    categories: list[dict[str, Any]] = Field(default_factory=list)
    #: The categories it found in its own results and that wait for you to confirm them.
    induced_categories: list[dict[str, Any]] = Field(default_factory=list)
    #: How people are found: ``discover``, and how many are ``proposed`` and ``active``.
    people: dict[str, Any] = Field(default_factory=dict)
    note: str | None = None

    @field_validator("kinds", mode="before")
    @classmethod
    def _bare_kinds(cls, value: Any) -> Any:
        """A kind named bare, as a monitor created without inputs may have it."""
        if isinstance(value, list):
            return [{"kind": item} if isinstance(item, str) else item for item in value]
        return value
