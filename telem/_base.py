"""Shared configuration and request/response plumbing for the Telem clients.

:class:`BaseClient` centralizes credential/URL resolution, header building, request-body
construction, and response handling so the sync and async clients cannot drift in behavior.
It performs no HTTP itself — each concrete client owns its own httpx client.
"""

from __future__ import annotations

import os
import random
import re
import uuid
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any
from urllib.parse import quote

import httpx

from telem._config_files import TELEM_OPTIONS, option_from_env, read_credentials
from telem._version import __version__
from telem.errors import (
    APIStatusError,
    TelemConnectionError,
    TelemError,
    TelemServerVersionError,
    TelemTimeoutError,
    raise_for_status,
)

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
#: How long a monitor create keeps retrying, from its first attempt: a burst of 64
#: creates on a 60/min key drains at about one a second. The count is a backstop.
_CREATE_RETRY_BUDGET = 120.0
_CREATE_MAX_RETRIES = 60
_MAX_BACKOFF_WAIT = 10.0
_RETRY_BASE_WAIT = 0.5
_IDEMPOTENCY_KEY = re.compile(r"[\x21-\x7e]{1,255}")
_RETRY_AFTER_SECONDS = re.compile(r"\s*[0-9]+\s*")
#: A GET is sent again up to this many times, as the JS SDK's ``maxRetries``.
_DEFAULT_MAX_RETRIES = 2
_MAX_RETRY_WAIT = 30.0


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


def path_segment(value: str) -> str:
    """Encode a caller-supplied id as ONE path segment, so ``../`` cannot change the route."""
    # quote() leaves dots alone, and httpx collapses a bare ``..`` segment.
    if value in (".", ".."):
        raise ValueError(f"invalid id: {value!r}")
    return quote(value, safe="")


def _providers_refusal(include: list[str], include_from_call: bool, exclude_from_call: bool) -> str:
    """The message for an ``include`` that ``exclude`` empties: where each list was set."""

    def where(half: str, from_call: bool) -> str:
        return f"the {half} argument" if from_call else f"default_{half} or TELEM_{half.upper()}"

    names = ", ".join(include)
    return (
        f"providers_exclude removes every provider in providers_include ({names}), so no "
        f"provider is left and no search was sent. Remove {names} from providers_include "
        f"(set by {where('providers_include', include_from_call)}) or from providers_exclude "
        f"(set by {where('providers_exclude', exclude_from_call)})."
    )


def _flag_env(name: str) -> bool | None:
    """Parse a boolean env var: exactly ``"1"`` means True, anything else is unset."""
    return True if os.environ.get(name) == "1" else None


def _fetch_env(key: str) -> Any | None:
    """Read one fetch key from its ``TELEM_FETCH_*`` var with the shared option table's
    parsing (``nameList`` as CSV, ``name`` trimmed, ``flag`` on with exactly ``"1"``)."""
    spec = next(spec for spec in TELEM_OPTIONS if spec.key == key)
    return option_from_env(spec, os.environ)


def _int_env(name: str) -> int | None:
    """Parse an integer env var, or ``None`` when it is unset or not an integer."""
    try:
        return int(os.environ.get(name, ""))
    except ValueError:
        return None


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
        default_auto_routing: str | None = None,
        default_max_routing_providers: int | None = None,
        default_topic: str | None = None,
        default_vertical: str | None = None,
        vertical: str | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError(f"max_retries must be a whole number of retries, got {max_retries!r}")
        self.max_retries = max_retries
        # Credentials: argument, else env var, else ``~/.telem/credentials.json`` — the
        # file the wizard/`--login` writes, so an installed user never has to export
        # anything (boto/gh precedent). The file is read ONLY when something is still
        # unresolved, so a fully-configured caller never touches the disk, and it is read
        # silently: a malformed one leaves the existing defaults exactly as they were.
        #
        # This is the SDK's ONE config file. Repo-local `.telem/telem.json` options stay
        # unread here on purpose: a library must not let a checked-out
        # repository steer an arbitrary program's spend.
        if api_key is not None and not api_key.strip():
            raise ValueError("api_key is empty")
        resolved_key = api_key or os.environ.get("TELEM_API_KEY") or None
        resolved_base = base_url or os.environ.get("TELEM_BASE_URL") or None
        if resolved_key is None or resolved_base is None:
            credentials = read_credentials(os.environ)
            resolved_key = resolved_key or credentials.get("apiKey")
            resolved_base = resolved_base or credentials.get("baseUrl")
        # Stripped: ``$(cat keyfile)`` leaves a trailing newline, which httpx rejects
        # as an illegal header value.
        self.api_key = resolved_key.strip() if resolved_key else resolved_key
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
        self.default_auto_routing = (
            default_auto_routing
            if default_auto_routing is not None
            else ((os.environ.get("TELEM_AUTO_ROUTING") or "").strip() or None)
        )
        self.default_max_routing_providers = (
            default_max_routing_providers
            if default_max_routing_providers is not None
            else _int_env("TELEM_MAX_ROUTING_PROVIDERS")
        )
        self.default_topic = (
            default_topic
            if default_topic is not None
            else ((os.environ.get("TELEM_TOPIC") or "").strip() or None)
        )
        # UNPUBLISHED (2026-11-27): the search vertical ("health", "general"; the router
        # validates the name and 400s an unknown one). Set, it becomes ``search.vertical``
        # on a search that does not set one, and ``inputs.vertical`` on a monitor create
        # with ``inputs`` that does not set one. Unset (the default), nothing is sent and
        # every body is byte-identical to before. ``vertical`` is an alias of
        # ``default_vertical``; ``TELEM_VERTICAL`` is the fallback. Deliberately NOT a
        # config-file key, a tool argument, or documented: a router that predates the
        # field rejects a request carrying it. ``TELEM_VERTICAL`` makes EVERY search and
        # monitor create of the process send it, so it must not be set in any environment
        # until the router that accepts ``vertical`` is live.
        if default_vertical is not None and vertical is not None and default_vertical != vertical:
            raise ValueError(
                f"vertical={vertical!r} and default_vertical={default_vertical!r} are the "
                "same option; pass one of them"
            )
        given_vertical = default_vertical if default_vertical is not None else vertical
        self.default_vertical = (
            given_vertical
            if given_vertical is not None
            else ((os.environ.get("TELEM_VERTICAL") or "").strip() or None)
        )
        # Fetch defaults: env var only, else unset; a ``fetch()`` argument wins per key.
        # The tier is lowercased, as the agent plugins send it. The config files stay
        # unread here too: ``resolve_fetch_options()`` is the opt-in for them.
        fetch_tier = _fetch_env("fetchTier")
        self.default_fetch_providers = _fetch_env("fetchProviders")
        self.default_fetch_tier = fetch_tier.lower() if fetch_tier is not None else None
        self.default_fetch_no_cache = _fetch_env("fetchNoCache")

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
        auto_routing: str | None = None,
        max_routing_providers: int | None = None,
        topic: str | None = None,
        published_after: str | date | None = None,
        published_before: str | date | None = None,
        include_domains: list[str] | None = None,
        vertical: str | None = None,
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
           emptied by that subtraction raises ``ValueError`` before anything is sent: no
           provider is left, and the deployment's default set could hold an excluded one.

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
            kept = [name for name in include if name not in excluded]
            if include and not kept:
                raise ValueError(
                    _providers_refusal(
                        include, providers_include is not None, providers_exclude is not None
                    )
                )
            include = kept
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
        if auto_routing is None:
            auto_routing = self.default_auto_routing
        # "off" (any case) means do not route: the router answers 422 for it as a
        # mode, and a per-call "off" also overrides a default mode.
        if auto_routing is not None and auto_routing.lower() != "off":
            block["auto_routing"] = auto_routing
        if max_routing_providers is None:
            max_routing_providers = self.default_max_routing_providers
        if max_routing_providers is not None:
            block["max_routing_providers"] = max_routing_providers
        if topic is None:
            topic = self.default_topic
        if topic is not None:
            block["topic"] = topic
        # Per call only: no client default, since a fixed window goes stale.
        if published_after is not None:
            block["published_after"] = _iso(published_after)
        if published_before is not None:
            block["published_before"] = _iso(published_before)
        if include_domains is not None:
            block["include_domains"] = include_domains
        # Unpublished; see ``__init__``. A call value wins over the client default.
        if vertical is None:
            vertical = self.default_vertical
        if vertical is not None:
            block["vertical"] = vertical
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
        auto_routing: str | None = None,
        max_routing_providers: int | None = None,
        topic: str | None = None,
        published_after: str | date | None = None,
        published_before: str | date | None = None,
        include_domains: list[str] | None = None,
        vertical: str | None = None,
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
            auto_routing=auto_routing,
            max_routing_providers=max_routing_providers,
            topic=topic,
            published_after=published_after,
            published_before=published_before,
            include_domains=include_domains,
            vertical=vertical,
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
        tier: str | None = None,
        no_cache: bool | None = None,
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
        # A call argument wins; ``None`` falls back to the ``TELEM_FETCH_*`` default.
        if providers is None and self.default_fetch_providers is not None:
            providers = list(self.default_fetch_providers)
        if tier is None:
            tier = self.default_fetch_tier
        if no_cache is None:
            no_cache = self.default_fetch_no_cache
        options: dict[str, Any] = {}
        if providers is not None:
            options["providers"] = providers
        if inline_content is not None:
            options["inline_content"] = inline_content
        if inline_max_chars is not None:
            options["inline_max_chars"] = inline_max_chars
        if content_format is not None:
            options["content_format"] = content_format
        if tier is not None:
            options["tier"] = tier
        if no_cache is not None:
            options["no_cache"] = no_cache
        if options:
            body["options"] = options
        if session is not None:
            body["session_id"] = session
        return body

    def _connection_error(self, path: str, exc: Exception) -> TelemConnectionError:
        """Wrap a transport-level failure (no HTTP response was produced) for ``path``."""
        # A LocalProtocolError is a fault in the request we built, and its text can quote
        # a header value (the Authorization key), so its text is never echoed.
        detail = "" if isinstance(exc, httpx.LocalProtocolError) else f": {exc}"
        return TelemConnectionError(
            f"could not reach {self._url(path)}: {type(exc).__name__}{detail}"
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

    def _timeout_error(self, path: str, exc: Exception) -> TelemTimeoutError:
        """Wrap a timeout for ``path``: the server may have got the request, so it is not
        "could not reach"."""
        return TelemTimeoutError(
            f"request to {self._url(path)} timed out after {self.timeout} s: "
            f"{type(exc).__name__}: {exc}"
        )

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
        try:
            return cls._handle_response(response.status_code, body)
        except TelemError as exc:
            exc.retry_after = response.headers.get("retry-after")
            raise


def resolve_idempotency_key(given: str | None) -> str:
    """The ``Idempotency-Key`` of one create: the caller's, checked, or a fresh uuid4."""
    if given is None:
        return str(uuid.uuid4())
    if not isinstance(given, str) or not _IDEMPOTENCY_KEY.fullmatch(given):
        raise ValueError("idempotency_key must be 1 to 255 visible ASCII characters")
    return given


def _jittered(wait: float) -> float:
    """``wait`` plus a random extra of up to 1 s or half the wait, whichever is more,
    so refused creates do not come back in lockstep."""
    return wait + random.random() * max(1.0, wait / 2)


#: What the edge answers a call that outlasted it: the router may still have done the work.
GATEWAY_TIMEOUTS = frozenset({502, 504})


def create_retry_wait(
    error: TelemError, attempt: int, elapsed: float, *, gateway_timeouts: bool = False
) -> float | None:
    """Seconds to wait before retrying a create sent with an ``Idempotency-Key``, or
    ``None`` when ``error`` is final or the wait would end past the budget. The key
    makes a retry after a lost reply return the first monitor, so a timeout or a
    connection failure is retried too, and with ``gateway_timeouts`` a 502 or 504."""
    if attempt >= _CREATE_MAX_RETRIES:
        return None
    if isinstance(error, TelemConnectionError):
        if isinstance(error.__cause__, (httpx.LocalProtocolError, httpx.UnsupportedProtocol)):
            return None
    elif not isinstance(error, APIStatusError):
        return None
    elif gateway_timeouts and error.status_code in GATEWAY_TIMEOUTS:
        pass
    elif error.status_code == 409:
        envelope = error.body.get("error") if isinstance(error.body, dict) else None
        if not isinstance(envelope, dict) or envelope.get("code") != "idempotency_in_progress":
            return None
    elif error.status_code != 429:
        return None
    retry_after = error.retry_after
    if retry_after is not None and _RETRY_AFTER_SECONDS.fullmatch(retry_after):
        return _create_wait(float(retry_after), attempt, elapsed)
    return _create_wait(None, attempt, elapsed)


def batch_retry_wait(retry_afters: list[int | None], attempt: int, elapsed: float) -> float | None:
    """Seconds to wait before re-sending the items of a bulk create that Telem asked to come
    back later: the longest of their ``retry_after``, or ``None`` past the budget."""
    if attempt >= _CREATE_MAX_RETRIES:
        return None
    known = [s for s in retry_afters if isinstance(s, int) and not isinstance(s, bool) and s >= 0]
    return _create_wait(float(max(known)) if known else None, attempt, elapsed)


def _create_wait(retry_after: float | None, attempt: int, elapsed: float) -> float | None:
    if retry_after is not None:
        wait = _jittered(retry_after)
    else:
        wait = min(_jittered(_RETRY_BASE_WAIT * 2**attempt), _MAX_BACKOFF_WAIT)
    return None if elapsed + wait > _CREATE_RETRY_BUDGET else wait


def read_retry_wait(error: TelemError, attempt: int, max_retries: int) -> float | None:
    """Seconds to wait before sending a GET again, or ``None`` when ``error`` is final.

    The JS SDK's rule for reads: a 5xx, a 429 or a connection failure is retried up to
    ``max_retries`` times, after ``Retry-After`` (seconds form) or else 0.5 s doubling,
    capped at 30 s, without jitter. A timeout is not retried, nor is a request httpx
    could not form. Writes are never retried here."""
    if attempt >= max_retries or isinstance(error, TelemTimeoutError):
        return None
    if isinstance(error, TelemConnectionError):
        if isinstance(error.__cause__, (httpx.LocalProtocolError, httpx.UnsupportedProtocol)):
            return None
    elif not isinstance(error, APIStatusError) or not (
        (error.status_code or 0) >= 500 or error.status_code == 429
    ):
        return None
    retry_after = error.retry_after
    if retry_after is not None and _RETRY_AFTER_SECONDS.fullmatch(retry_after):
        return min(float(retry_after), _MAX_RETRY_WAIT)
    return min(_RETRY_BASE_WAIT * 2**attempt, _MAX_RETRY_WAIT)


def _iso(value: str | date) -> str:
    """A ``published_after``/``published_before`` value for the wire. A string goes
    as written (no zone means UTC; the server parses and validates it). A naive
    ``datetime`` is local time, as everywhere in Python, so it goes with the local
    zone attached; a ``date`` goes as the day."""
    if isinstance(value, str):
        return value
    if isinstance(value, datetime) and value.tzinfo is None:
        value = value.astimezone()
    return value.isoformat()
