"""Shared configuration and request/response plumbing for the Telem clients.

:class:`BaseClient` centralizes credential/URL resolution, header building, request-body
construction, and response handling so the sync and async clients cannot drift in behavior.
It performs no HTTP itself — each concrete client owns its own httpx client.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import httpx

from telem._config_files import read_credentials
from telem._version import __version__
from telem.errors import TelemConnectionError, TelemServerVersionError, raise_for_status

#: The hosted deployment. A default rather than a requirement: an ordinary user
#: should not have to know where Telem runs, and `TELEM_BASE_URL` or the
#: `base_url` argument points the client at a staging or self-hosted one.
_DEFAULT_BASE_URL = "https://router.telem.ai"
#: Derived, never written out: a hardcoded copy here reported ``telem-sdk/0.1.0``
#: from the 0.1.1 and 0.1.2 wheels too, so server-side version telemetry could not
#: tell three releases apart. See :mod:`telem._version`.
_DEFAULT_USER_AGENT = f"telem-sdk/{__version__}"
#: The server grants provider timeouts of up to 60 s at the `max` tier / with full content.
_DEFAULT_TIMEOUT = 60.0
#: The lowest ``normalized_schema_version`` this SDK can parse. The whole
#: response surface is the V2 envelope, so an older echo is a hard stop, not a fallback.
_MIN_SCHEMA_VERSION = 2


def _csv_env(name: str) -> list[str] | None:
    """Parse a comma-separated env var into a list of names, or ``None`` when unset.

    Items are stripped and empty items dropped; an all-empty value is unset. An env var
    can therefore never express an explicit empty list — only a constructor kwarg or a
    call argument can (``[]`` is a deliberate "send nothing", which the server rejects).
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",")]
    return [item for item in items if item] or None


def _flag_env(name: str) -> bool | None:
    """Parse a boolean env var: exactly ``"1"`` means True, anything else is unset."""
    return True if os.environ.get(name) == "1" else None


class BaseClient:
    """Holds resolved configuration shared by the sync and async clients."""

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
        # Credentials: argument, else env var, else ``~/.telem/credentials.json`` — the
        # file the wizard/`--login` writes, so an installed user never has to export
        # anything (boto/gh precedent). The file is read ONLY when something is still
        # unresolved, so a fully-configured caller never touches the disk, and it is read
        # silently: a malformed one leaves the existing defaults exactly as they were.
        #
        # This is the SDK's ONE config file. Repo-local `.telem/telem.json` options stay
        # unread here on purpose: a library must not let a checked-out
        # repository steer an arbitrary program's spend.
        resolved_key = api_key or os.environ.get("TELEM_API_KEY") or None
        resolved_base = base_url or os.environ.get("TELEM_BASE_URL") or None
        if resolved_key is None or resolved_base is None:
            credentials = read_credentials(os.environ)
            resolved_key = resolved_key or credentials.get("apiKey")
            resolved_base = resolved_base or credentials.get("baseUrl")
        self.api_key = resolved_key
        self.base_url = (resolved_base or _DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.user_agent = user_agent or _DEFAULT_USER_AGENT
        # Search defaults: argument, else env var, else unset. Only ``None`` is unset —
        # ``[]``/``False`` are explicit values and are sent to the server verbatim.
        self.default_tier = (
            default_tier if default_tier is not None else (os.environ.get("TELEM_TIER") or None)
        )
        self.default_fields = (
            default_fields if default_fields is not None else _csv_env("TELEM_FIELDS")
        )
        self.default_providers_include = (
            default_providers_include
            if default_providers_include is not None
            else _csv_env("TELEM_PROVIDERS_INCLUDE")
        )
        self.default_providers_exclude = (
            default_providers_exclude
            if default_providers_exclude is not None
            else _csv_env("TELEM_PROVIDERS_EXCLUDE")
        )
        self.default_include_full_content = (
            default_include_full_content
            if default_include_full_content is not None
            else _flag_env("TELEM_FULL_CONTENT")
        )

    def _headers(self) -> dict[str, str]:
        """Build the default request headers, adding Bearer auth only when an api_key is set."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _url(self, path: str) -> str:
        """Join a request path onto the resolved base URL."""
        return f"{self.base_url}/{path.lstrip('/')}"

    def _resolve_search_options(
        self,
        *,
        tier: str | None,
        fields: list[str] | None,
        providers_include: list[str] | None,
        providers_exclude: list[str] | None,
        provider_overrides: dict[str, dict[str, Any]] | None,
        num_results: int | None,
        include_raw: bool | None,
        include_full_content: bool | None,
    ) -> dict[str, Any]:
        """Resolve call arguments against the client defaults into the wire ``search`` block.

        Each option resolves to the call argument when it is not ``None``, else the client
        default (constructor argument or env var), else unset. Two composition rules mirror
        the server's semantics:

        1. A resolved ``fields`` replaces ``tier``: the level that set it wins (a call-level
           ``fields`` drops a default ``tier``, a call-level ``tier`` drops default
           ``fields``), and when both come from the same level ``fields`` wins. The block
           never carries both.
        2. When both provider halves resolve, ``include`` fully determines the set: the
           excluded names are subtracted client-side and ``exclude`` is not sent. An include
           emptied by that subtraction is still sent, as ``[]``.

        Returns:
            The ``search`` block, containing exactly the options that are set (``{}`` when
            none are, in which case the caller omits the key entirely).
        """
        from_call_tier = tier is not None
        from_call_fields = fields is not None
        resolved_tier = tier if from_call_tier else self.default_tier
        resolved_fields = fields if from_call_fields else self.default_fields
        if resolved_tier is not None and resolved_fields is not None:
            if from_call_tier and not from_call_fields:
                resolved_fields = None
            else:
                resolved_tier = None

        include = (
            providers_include if providers_include is not None else self.default_providers_include
        )
        exclude = (
            providers_exclude if providers_exclude is not None else self.default_providers_exclude
        )
        if include is not None and exclude is not None:
            excluded = set(exclude)
            include = [name for name in include if name not in excluded]
            exclude = None

        if include_full_content is None:
            include_full_content = self.default_include_full_content

        providers: dict[str, Any] = {}
        if include is not None:
            providers["include"] = include
        if exclude is not None:
            providers["exclude"] = exclude

        block: dict[str, Any] = {}
        if resolved_tier is not None:
            block["tier"] = resolved_tier
        if resolved_fields is not None:
            block["fields"] = resolved_fields
        if providers:
            block["providers"] = providers
        if provider_overrides is not None:
            block["provider_overrides"] = provider_overrides
        if num_results is not None:
            block["num_results"] = num_results
        if include_raw is not None:
            block["include_raw"] = include_raw
        if include_full_content is not None:
            block["include_full_content"] = include_full_content
        return block

    def _search_body(
        self,
        *,
        query: str | Sequence[str],
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
    ) -> dict[str, Any]:
        """Build the JSON body for ``POST /v1/interactions``.

        The ``search`` block is attached only when at least one search option is set after
        resolution (see :meth:`_resolve_search_options`); ``preprocessor_names`` is never
        sent (under the V2 contract it is a 400 — provider selection is
        ``search.providers``). ``goal``/``context`` are merged into ``metadata`` only when
        provided as arguments, never overwriting caller-supplied metadata keys otherwise.
        ``session_id`` is included only when ``session`` is set.

        ``query`` may be a sequence of queries: they are batched into ONE interaction
        (``user_input`` becomes a list of ``{"query": ...}`` dicts) and the backend runs
        them concurrently, tagging each run with ``batch_index``/``query``. A one-element
        sequence collapses to the plain-string body.

        Raises:
            ValueError: If ``query`` is an empty sequence.
            TypeError: If a sequence element is not a string.
        """
        merged_metadata: dict[str, Any] = dict(metadata) if metadata else {}
        if goal is not None:
            merged_metadata["goal"] = goal
        if context is not None:
            merged_metadata["context"] = context

        user_input: dict[str, Any] | list[dict[str, Any]]
        if isinstance(query, str):
            user_input = {"query": query}
        else:
            queries = list(query)
            if not queries:
                raise ValueError("search() requires at least one query")
            for item in queries:
                if not isinstance(item, str):
                    raise TypeError(f"search() queries must be strings, got {type(item).__name__}")
            if len(queries) == 1:
                user_input = {"query": queries[0]}
            else:
                user_input = [{"query": item} for item in queries]

        body: dict[str, Any] = {
            "user_input": user_input,
            "postprocessor_names": [],
            "metadata": merged_metadata,
        }
        search = self._resolve_search_options(
            tier=tier,
            fields=fields,
            providers_include=providers_include,
            providers_exclude=providers_exclude,
            provider_overrides=provider_overrides,
            num_results=num_results,
            include_raw=include_raw,
            include_full_content=include_full_content,
        )
        if search:
            body["search"] = search
        if session is not None:
            body["session_id"] = session
        return body

    def _fetch_body(
        self,
        *,
        urls: Sequence[str],
        providers: list[str] | None = None,
        inline_content: bool | None = None,
        inline_max_chars: int | None = None,
        content_format: str | None = None,
        session: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the JSON body for ``POST /v1/fetch``.

        The endpoint's request model is ``extra="forbid"``: the body carries the
        top-level ``urls`` list plus an optional typed ``options`` block — never the
        legacy ``user_input``/``metadata.web_fetch`` shapes. ``options`` is attached
        only when at least one fetch option is set; there is no ``max_urls`` knob (the
        server cap is the cap). URL values are not pre-validated client-side beyond
        shape — non-http(s) items and over-cap batches are the server's 400s.

        Raises:
            ValueError: If ``urls`` is empty.
            TypeError: If ``urls`` is a bare string, or an element is not a string.
        """
        if isinstance(urls, str):
            raise TypeError("fetch() urls must be a sequence of URL strings, not a single string")
        url_list = list(urls)
        if not url_list:
            raise ValueError("fetch() requires at least one URL")
        for item in url_list:
            if not isinstance(item, str):
                raise TypeError(f"fetch() urls must be strings, got {type(item).__name__}")

        body: dict[str, Any] = {
            "urls": url_list,
            "metadata": dict(metadata) if metadata else {},
        }
        options: dict[str, Any] = {}
        if providers is not None:
            options["providers"] = providers
        if inline_content is not None:
            options["inline_content"] = inline_content
        if inline_max_chars is not None:
            options["inline_max_chars"] = inline_max_chars
        if content_format is not None:
            options["content_format"] = content_format
        if options:
            body["options"] = options
        if session is not None:
            body["session_id"] = session
        return body

    def _connection_error(self, path: str, exc: Exception) -> TelemConnectionError:
        """Wrap a transport-level failure (no HTTP response was produced) for ``path``."""
        return TelemConnectionError(
            f"could not reach {self._url(path)}: {type(exc).__name__}: {exc}"
        )

    @staticmethod
    def _check_search_contract(data: Any) -> None:
        """Verify a search response is V2-normalized before it is parsed.

        Runs on the search POST only — GETs (sessions, providers) never carry the echo.
        The body must be a dict whose ``normalized_schema_version`` is an int ``>= 2``;
        a non-dict body fails identically, carrying the observed type in the message.

        Args:
            data: The parsed body returned by ``_handle_response``.

        Raises:
            TelemServerVersionError: When the server did not echo the V2 contract.
        """
        if isinstance(data, dict):
            observed: Any = data.get("normalized_schema_version")
            if isinstance(observed, int) and observed >= _MIN_SCHEMA_VERSION:
                return
        else:
            observed = type(data)
        raise TelemServerVersionError(
            f"server echoed normalized_schema_version={observed!r}; expected "
            f">= {_MIN_SCHEMA_VERSION} — the server pre-dates the V2 normalized "
            "contract, or is a dev deployment with no adapter-backed search providers "
            "configured; point the SDK at a provider-configured V2 deployment"
        )

    @staticmethod
    def _handle_response(status_code: int, body: object) -> Any:
        """Raise a typed error on non-2xx, otherwise return the parsed body.

        Args:
            status_code: The HTTP status code of the response.
            body: The parsed JSON body (dict/list) or raw text.

        Returns:
            The parsed body unchanged on success.
        """
        raise_for_status(status_code, body)
        return body

    @classmethod
    def _result(cls, response: httpx.Response) -> Any:
        """Parse an httpx response into JSON (falling back to text) and apply error mapping.

        Args:
            response: The httpx response returned by either client.

        Returns:
            The parsed JSON body (or raw text) on success.
        """
        try:
            body: object = response.json()
        except ValueError:
            body = response.text
        return cls._handle_response(response.status_code, body)
