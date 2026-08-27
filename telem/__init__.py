"""Telem SDK — a Python client for the Telem search orchestration API."""

from telem.async_client import AsyncTelem
from telem.client import Telem
from telem.config import SearchOptionsResolution, resolve_search_options
from telem.errors import (
    APIStatusError,
    AuthError,
    BadRequestError,
    NotFoundError,
    TelemConnectionError,
    TelemError,
    TelemServerVersionError,
)
from telem.models import (
    FetchResponse,
    FetchResult,
    ProviderInfo,
    ProviderRun,
    SearchResponse,
    SearchResult,
    SessionHistory,
    SessionResults,
    SessionSummary,
)
from telem._version import __version__

__all__ = [
    "__version__",
    "Telem",
    "AsyncTelem",
    "SearchResult",
    "ProviderRun",
    "SearchResponse",
    "FetchResult",
    "FetchResponse",
    "ProviderInfo",
    "SessionSummary",
    "SessionHistory",
    "SessionResults",
    "TelemError",
    "AuthError",
    "BadRequestError",
    "NotFoundError",
    "APIStatusError",
    "TelemServerVersionError",
    "TelemConnectionError",
    # The unified config — read explicitly, never by the client itself.
    "resolve_search_options",
    "SearchOptionsResolution",
]
