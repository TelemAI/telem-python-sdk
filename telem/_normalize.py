"""Parsing of a V2 search/fetch response body into the SDK models.

Under the V2 normalized contract each preprocessor run's ``output_payload`` IS the
server's normalized envelope, so there is nothing to normalize client-side: the
envelope is read, not reconstructed. Provenance is fixed — ``query``/``batch_index``/
``latency_ms``/``preprocessor_run_id``/``raw`` come from the RUN ROW, the query-level
blocks come from the envelope interior, and result rows land verbatim on
:class:`~telem.models.SearchResult`.

Four shape malformations degrade instead of raising, so one bad provider cannot cost
the caller the rest of a healthy response: a non-dict run, a missing or non-dict
envelope, and a non-list ``results`` each yield an empty run, and a non-dict result
row is skipped. That is the whole list — anything else (a row whose value has the
wrong type for its field, say) still surfaces pydantic's ``ValidationError``.
"""

from __future__ import annotations

from typing import Any

from telem.models import FetchResponse, FetchResult, ProviderRun, SearchResponse, SearchResult


def _results(envelope: dict[str, Any], provider: str) -> list[SearchResult]:
    """Read the envelope's ``results`` rows into :class:`SearchResult` objects.

    Each row is passed through verbatim (unknown keys are ignored by the model) with
    three overrides: ``url``/``title`` are coerced from ``null`` to ``""`` (the server
    emits ``title: null`` routinely and both fields are non-optional here), ``provider``
    is the run's own name, and ``raw`` is the row dict itself.

    Args:
        envelope: The run's normalized envelope (``output_payload``).
        provider: The run's ``preprocessor_name``.

    Returns:
        The parsed rows, in envelope order (empty when ``results`` is not a list).
    """
    rows = envelope.get("results")
    if not isinstance(rows, list):
        return []
    return [
        SearchResult.model_validate(
            {
                **row,
                "url": row.get("url") or "",
                "title": row.get("title") or "",
                "provider": provider,
                "raw": row,
            }
        )
        for row in rows
        if isinstance(row, dict)
    ]


def _provider_run(run: Any) -> ProviderRun:
    """Build one :class:`ProviderRun` from a ``preprocessor_runs`` entry.

    Args:
        run: One run object from the interaction body (anything, defensively).

    Returns:
        The run's :class:`ProviderRun`; an empty run when ``run`` is not a dict.
    """
    if not isinstance(run, dict):
        return ProviderRun(provider="", status="")

    provider = run.get("preprocessor_name", "")
    envelope = run.get("output_payload")
    if not isinstance(envelope, dict):
        # A failed run (``output_payload: null``) or a shape the SDK does not know:
        # the row-level provenance below still applies, there are simply no results.
        envelope = {}
    run_id = run.get("id")
    return ProviderRun(
        # -- from the run row --
        provider=provider,
        status=run.get("status", ""),
        error=run.get("error"),
        latency_ms=run.get("latency_ms"),
        preprocessor_run_id=str(run_id) if run_id else None,
        query=run.get("query", ""),
        batch_index=run.get("batch_index", 0),
        raw=run.get("raw_payload"),
        # -- from the envelope interior --
        results=_results(envelope, provider),
        tier=envelope.get("tier"),
        fields=envelope.get("fields", []),
        answer=envelope.get("answer"),
        entities=envelope.get("entities"),
        related=envelope.get("related"),
        verticals=envelope.get("verticals"),
        usage=envelope.get("usage"),
        warnings=envelope.get("warnings", []),
    )


def build_search_response(interaction: dict[str, Any]) -> SearchResponse:
    """Build a :class:`SearchResponse` from a parsed ``POST /v1/interactions`` body.

    Each entry in ``interaction["preprocessor_runs"]`` becomes a :class:`ProviderRun`;
    the flattened ``results`` list is the in-order concatenation of every provider's
    rows (providers in run order, rows in envelope order).

    Args:
        interaction: The parsed JSON body of an ``InteractionResponse``.

    Returns:
        The assembled :class:`SearchResponse`.
    """
    by_provider: list[ProviderRun] = []
    flat: list[SearchResult] = []

    for run in interaction.get("preprocessor_runs", []) or []:
        provider_run = _provider_run(run)
        by_provider.append(provider_run)
        flat.extend(provider_run.results)

    return SearchResponse(
        results=flat,
        by_provider=by_provider,
        session_id=str(interaction.get("session_id", "")),
        interaction_id=str(interaction.get("interaction_id", "")),
        status=interaction.get("status", ""),
        normalized_schema_version=interaction.get("normalized_schema_version"),
        raw=interaction,
    )


def _fetch_result(run: dict[str, Any]) -> FetchResult:
    """Build one :class:`FetchResult` from a ``web_fetch`` run.

    The engine emits one run per URL with a single ``fetched_results`` row; the row's
    keys land verbatim (unknown keys are ignored by the model). ``url``/``status``/
    ``error``/``latency_ms`` fall back to the RUN ROW (whose ``query`` is the URL) so a
    run that failed before producing a row still yields a result for its URL.

    Args:
        run: One ``web_fetch`` entry from ``preprocessor_runs``.

    Returns:
        The run's :class:`FetchResult`.
    """
    envelope = run.get("output_payload")
    if not isinstance(envelope, dict):
        envelope = {}
    rows = envelope.get("fetched_results")
    row: dict[str, Any] = {}
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        row = rows[0]
    return FetchResult.model_validate(
        {
            **row,
            "url": row.get("url") or run.get("query") or "",
            "title": row.get("title") or "",
            "status": row.get("status") or run.get("status") or "",
            "provider": row.get("provider") or "",
            "error": row.get("error") or run.get("error"),
            "latency_ms": row.get("latency_ms", run.get("latency_ms")),
            "batch_index": run.get("batch_index", 0),
            "raw": row,
        }
    )


def build_fetch_response(interaction: dict[str, Any]) -> FetchResponse:
    """Build a :class:`FetchResponse` from a parsed ``POST /v1/fetch`` body.

    Each ``web_fetch`` entry in ``interaction["preprocessor_runs"]`` (one per URL)
    becomes a :class:`FetchResult`; results are ordered by ``batch_index``, i.e. the
    request's URL order. Fetch responses carry no ``normalized_schema_version`` echo,
    so no version gate applies here.

    Args:
        interaction: The parsed JSON body of an ``InteractionResponse``.

    Returns:
        The assembled :class:`FetchResponse`.
    """
    runs = [
        run
        for run in (interaction.get("preprocessor_runs") or [])
        if isinstance(run, dict) and run.get("preprocessor_name") == "web_fetch"
    ]
    runs.sort(key=lambda run: run.get("batch_index") or 0)
    return FetchResponse(
        results=[_fetch_result(run) for run in runs],
        session_id=str(interaction.get("session_id", "")),
        interaction_id=str(interaction.get("interaction_id", "")),
        status=interaction.get("status", ""),
        raw=interaction,
    )
