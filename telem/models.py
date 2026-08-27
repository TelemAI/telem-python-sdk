"""Pydantic v2 models for the Telem SDK public API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
