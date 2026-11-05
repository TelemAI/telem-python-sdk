"""Pydantic v2 models for the Telem SDK public API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class ProviderInfo(BaseModel):
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


class SessionSummary(BaseModel):
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


class AgentCompileState(BaseModel):
    """The job that makes the agent's plan: ``AgentState.compile``."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    status: str
    #: Why the plan could not be made; ``None`` when it did not fail.
    error: str | None = None


class AgentState(BaseModel):
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


class Monitor(BaseModel):
    """A monitor: a query that Telem runs on a schedule."""

    model_config = ConfigDict(extra="ignore")

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
    #: Signs the webhook deliveries. Only in the response that set ``webhook_url``.
    webhook_secret: str | None = None
    #: The monitor agent; ``None`` for a monitor without one (``agent=False``), and from
    #: a router that does not say.
    agent_state: AgentState | None = None
    #: Only in the answer to a create: why the monitor agent did not start, or not yet.
    #: ``None`` when it started, and in every other answer.
    agent_note: str | None = None


class MonitorPage(BaseModel):
    """One page of :meth:`~telem.resources.monitors.MonitorsResource.list_page`."""

    model_config = ConfigDict(extra="ignore")

    monitors: list[Monitor] = []
    #: Pass as ``cursor`` to get the next page; ``None`` on the last page.
    next_cursor: str | None = None


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
    #: The whole document, as the router sent it.
    raw: dict[str, Any] = Field(default_factory=dict)


class ChangeSubject(BaseModel):
    """The item a change is about.

    ``entity_id`` is ``None`` for a change to the whole monitor, such as a run or a plan
    change.
    """

    model_config = ConfigDict(extra="ignore")

    entity_id: str | None = None
    name: str | None = None


class ChangeRecord(BaseModel):
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


class OnboardingExample(BaseModel):
    """One example of a candidate: the name it was found under, the quote and its page."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    quote: str | None = None
    url: str | None = None


class OnboardingCandidate(BaseModel):
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


class CompanySheet(BaseModel):
    """What the first read of one company found: ``OnboardingCompany.sheet``."""

    model_config = ConfigDict(extra="ignore")

    company_id: str | None = None
    #: How many pages were read.
    pages: int = 0
    #: The most found first.
    candidates: list[OnboardingCandidate] = Field(default_factory=list)
    note: str | None = None


class OnboardingCompany(BaseModel):
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

    @model_validator(mode="before")
    @classmethod
    def _keep_raw(cls, data: Any) -> Any:
        return {**data, "raw": data} if isinstance(data, dict) else data


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
