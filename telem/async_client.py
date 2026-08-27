"""Asynchronous Telem client built on :class:`httpx.AsyncClient`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from telem._base import _DEFAULT_TIMEOUT, BaseClient
from telem._normalize import build_fetch_response, build_search_response
from telem.models import FetchResponse, ProviderInfo, SearchResponse
from telem.resources.sessions import AsyncSessionsResource


class AsyncTelem(BaseClient):
    """Asynchronous client for the Telem search orchestration API.

    Example:
        >>> from telem import AsyncTelem
        >>> async with AsyncTelem() as client:
        ...     response = await client.search("best python http client")
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        user_agent: str | None = None,
        default_tier: str | None = None,
        default_fields: list[str] | None = None,
        default_providers_include: list[str] | None = None,
        default_providers_exclude: list[str] | None = None,
        default_include_full_content: bool | None = None,
    ) -> None:
        """Create an asynchronous Telem client.

        Args:
            api_key: Bearer token. Falls back to the ``TELEM_API_KEY`` env var, then to
                ``~/.telem/credentials.json`` (the file the guided installer writes;
                ``TELEM_CONFIG_DIR`` relocates the directory), then anonymous.
            base_url: API base URL. Falls back to ``TELEM_BASE_URL``, then to the same
                credentials file, then the hosted deployment at router.telem.ai.
            timeout: Request timeout in seconds.
            user_agent: Optional User-Agent header override.
            default_tier: Default result tier for every search. Falls back to ``TELEM_TIER``.
            default_fields: Default explicit field list. Falls back to ``TELEM_FIELDS`` (csv).
            default_providers_include: Default provider allow-list. Falls back to
                ``TELEM_PROVIDERS_INCLUDE`` (csv).
            default_providers_exclude: Default provider deny-list. Falls back to
                ``TELEM_PROVIDERS_EXCLUDE`` (csv).
            default_include_full_content: Default full-content flag. Falls back to
                ``TELEM_FULL_CONTENT`` (only ``"1"`` enables it).
        """
        super().__init__(
            api_key,
            base_url=base_url,
            timeout=timeout,
            user_agent=user_agent,
            default_tier=default_tier,
            default_fields=default_fields,
            default_providers_include=default_providers_include,
            default_providers_exclude=default_providers_exclude,
            default_include_full_content=default_include_full_content,
        )
        self._client = httpx.AsyncClient(timeout=timeout, headers=self._headers())
        self._sessions = AsyncSessionsResource(self)

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Issue a GET request and return the parsed body (raising typed errors on failure)."""
        try:
            response = await self._client.get(self._url(path), params=params)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise self._connection_error(path, exc) from exc
        return self._result(response)

    async def _post(self, path: str, json: dict[str, Any]) -> Any:
        """Issue a POST request and return the parsed body (raising typed errors on failure)."""
        try:
            response = await self._client.post(self._url(path), json=json)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise self._connection_error(path, exc) from exc
        return self._result(response)

    async def search(
        self,
        query: str | Sequence[str],
        *,
        tier: str | None = None,
        fields: list[str] | None = None,
        providers_include: list[str] | None = None,
        providers_exclude: list[str] | None = None,
        provider_overrides: dict[str, dict[str, Any]] | None = None,
        num_results: int | None = None,
        include_raw: bool | None = None,
        include_full_content: bool | None = None,
        goal: str | None = None,
        context: str | None = None,
        session: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Run a search by creating an interaction and normalizing the provider results.

        Every option defaults to the client default (constructor argument or env var) and,
        failing that, to the server's own default: ``None`` means unset, while ``[]``/``{}``/
        ``False`` are explicit values sent to the server as-is. Values are not pre-validated
        client-side — tier names, field names and ``num_results`` bounds are the server's call.

        Args:
            query: The search query text, or a sequence of queries batched into ONE
                interaction — the backend runs them concurrently and tags each run in
                ``by_provider`` with ``batch_index``/``query``. A one-element sequence
                behaves exactly like a plain string.
            tier: Result tier (``minimalist``/``default``/``extended``/``max``).
            fields: Explicit field list; replaces ``tier`` when both would apply.
            providers_include: Provider allow-list.
            providers_exclude: Provider deny-list. When an allow-list also resolves, these
                names are subtracted from it and no deny-list is sent.
            provider_overrides: Per-provider raw request overrides, keyed by provider name.
            num_results: Results per provider (server default 5).
            include_raw: Ask the server to attach each provider's raw payload.
            include_full_content: Ask providers for full page content.
            goal: Optional agent goal, merged into request metadata.
            context: Optional context string, merged into request metadata.
            session: Optional session id to continue an existing session.
            metadata: Additional request metadata.

        Returns:
            A :class:`SearchResponse` with flattened results and a per-provider breakdown.
        """
        body = self._search_body(
            query=query,
            tier=tier,
            fields=fields,
            providers_include=providers_include,
            providers_exclude=providers_exclude,
            provider_overrides=provider_overrides,
            num_results=num_results,
            include_raw=include_raw,
            include_full_content=include_full_content,
            goal=goal,
            context=context,
            session=session,
            metadata=metadata,
        )
        data = await self._post("/v1/interactions", body)
        self._check_search_contract(data)
        return build_search_response(data)

    async def fetch(
        self,
        urls: Sequence[str],
        *,
        providers: list[str] | None = None,
        inline_content: bool | None = None,
        inline_max_chars: int | None = None,
        content_format: str | None = None,
        session: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FetchResponse:
        """Read web pages by URL via ``POST /v1/fetch``.

        The URLs are fetched together as one interaction (one run per URL); options are
        not pre-validated client-side — provider names, character bounds, formats, and
        the batch-size cap are the server's call. Fetch responses carry no
        ``normalized_schema_version`` echo, so no version gate runs here.

        Args:
            urls: The http(s) URLs to read (the server caps the batch size).
            providers: Fetch provider chain override (e.g. ``["tavily"]``).
            inline_content: Whether to inline page content in the response.
            inline_max_chars: Max inlined characters per page.
            content_format: Requested content format (e.g. ``"markdown"``).
            session: Optional session id to continue an existing session.
            metadata: Additional request metadata.

        Returns:
            A :class:`FetchResponse` with one :class:`FetchResult` per URL, in request
            order; failed URLs carry their run's ``error`` and ``status``.
        """
        body = self._fetch_body(
            urls=urls,
            providers=providers,
            inline_content=inline_content,
            inline_max_chars=inline_max_chars,
            content_format=content_format,
            session=session,
            metadata=metadata,
        )
        data = await self._post("/v1/fetch", body)
        return build_fetch_response(data)

    async def providers(self) -> list[ProviderInfo]:
        """List available search providers (preprocessors).

        Returns:
            A list of :class:`ProviderInfo` describing each configured provider.
        """
        data = await self._get("/v1/preprocessors")
        items = data.get("preprocessors", []) if isinstance(data, dict) else []
        return [ProviderInfo.model_validate(item) for item in items]

    def wrap(
        self,
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
        result_max_len: int = 2048,
    ) -> Any:
        """Patch an async OpenAI client in place so the model can call Telem search.

        One wrapped client = one agent conversation. The wrap records the conversation
        and sends it with every search as trajectory-v5 metadata (flat
        ``message_history`` plus ``session_key``/``fingerprint``/``node_key``).
        Requires the ``openai`` extra for real clients.

        The wrap fixes only the knobs below; every other search option (fields,
        provider exclusions, full content, ...) comes from this client's defaults,
        because the wrap runs the very same :meth:`search`.

        Args:
            client: An ``openai.AsyncOpenAI`` client (duck-typed).
            tier: Result tier for every search; ``None`` uses this client's default.
            providers_include: Provider allow-list forwarded to every search.
            num_results: Results per provider for every search (server default 5).
            goal: Optional agent goal merged into every search's metadata.
            context: Optional context string merged into every search's metadata.
            conversation_id: Stable id for this agent's conversation. Auto-minted when
                omitted — supply your own (a thread id, a request-scoped chat id) when the
                conversation outlives a single wrapped client, e.g. a web server wrapping a
                fresh client per request.
            context_window_id: Explicit context-window generation id. Omitted, the wrap
                anchors on the first message so trimming or summarizing the history starts a
                new generation on its own.
            parent: Optional parent wrapped client. Its conversation is frozen into
                this client's ``ancestors`` right now, at wrap() time — wrap the
                child at the moment it is delegated to, not at startup, or its
                lineage snapshot will predate anything the parent actually said.
            result_max_len: Max characters of each result's content fed to the model.

        Returns:
            The same client object, patched in place.
        """
        from telem.integrations.openai import wrap_async_openai

        return wrap_async_openai(
            self,
            client,
            tier=tier,
            providers_include=providers_include,
            num_results=num_results,
            goal=goal,
            context=context,
            conversation_id=conversation_id,
            context_window_id=context_window_id,
            parent=parent,
            result_max_len=result_max_len,
        )

    @property
    def sessions(self) -> AsyncSessionsResource:
        """Access session-scoped operations (list, history, results)."""
        return self._sessions

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> AsyncTelem:
        """Enter an async context manager, returning this client."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Exit the async context manager, closing the client."""
        await self.aclose()
