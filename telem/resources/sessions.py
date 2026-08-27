"""Session-scoped resources: listing sessions, history, and preprocessor results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from telem.models import SessionHistory, SessionResults, SessionSummary

if TYPE_CHECKING:
    from telem.async_client import AsyncTelem
    from telem.client import Telem


def _sessions_from(data: Any) -> list[SessionSummary]:
    """Map a ``GET /v1/sessions`` body into a list of :class:`SessionSummary`."""
    items = data.get("sessions", []) if isinstance(data, dict) else []
    return [SessionSummary.model_validate(item) for item in items]


def _history_from(data: Any) -> SessionHistory:
    """Map a ``GET /v1/sessions/{id}/history`` body into a :class:`SessionHistory`."""
    data = data if isinstance(data, dict) else {}
    return SessionHistory(
        session_id=str(data.get("session_id", "")),
        instance_id=data.get("instance_id"),
        interactions=data.get("interactions", []) or [],
    )


def _results_from(data: Any) -> SessionResults:
    """Map a websearch-preprocessor-results body into a :class:`SessionResults`."""
    data = data if isinstance(data, dict) else {}
    return SessionResults(
        session_id=str(data.get("session_id", "")),
        query=data.get("query", "") or "",
        goal=data.get("goal", "") or "",
        context=data.get("context", "") or "",
        previous_preprocessor_results=data.get("previous_preprocessor_results", []) or [],
    )


def _history_params(full: bool) -> dict[str, str]:
    """Return the query params selecting full or short history."""
    return {"type": "full" if full else "short"}


class SessionsResource:
    """Synchronous session operations bound to a :class:`~telem.client.Telem` client."""

    def __init__(self, client: Telem) -> None:
        self._client = client

    def list(self) -> list[SessionSummary]:
        """List the caller's sessions (most recent first).

        Requires an api_key; the server returns 401 (surfaced as ``AuthError``) otherwise.

        Returns:
            A list of :class:`SessionSummary`.
        """
        data = self._client._get("/v1/sessions")
        return _sessions_from(data)

    def history(self, session_id: str, *, full: bool = False) -> SessionHistory:
        """Fetch a session's interaction history.

        Args:
            session_id: The session id.
            full: When ``True`` request detailed (``type=full``) records, else short summaries.

        Returns:
            A :class:`SessionHistory` with pass-through interaction records.
        """
        data = self._client._get(f"/v1/sessions/{session_id}/history", params=_history_params(full))
        return _history_from(data)

    def results(self, session_id: str) -> SessionResults:
        """Fetch aggregated websearch preprocessor results for a session.

        Args:
            session_id: The session id.

        Returns:
            A :class:`SessionResults` with the query/goal/context and per-step results.
        """
        data = self._client._get(f"/v1/sessions/{session_id}/websearch-preprocessor-results")
        return _results_from(data)


class AsyncSessionsResource:
    """Asynchronous session operations bound to an :class:`~telem.async_client.AsyncTelem`."""

    def __init__(self, client: AsyncTelem) -> None:
        self._client = client

    async def list(self) -> list[SessionSummary]:
        """List the caller's sessions (most recent first).

        Requires an api_key; the server returns 401 (surfaced as ``AuthError``) otherwise.

        Returns:
            A list of :class:`SessionSummary`.
        """
        data = await self._client._get("/v1/sessions")
        return _sessions_from(data)

    async def history(self, session_id: str, *, full: bool = False) -> SessionHistory:
        """Fetch a session's interaction history.

        Args:
            session_id: The session id.
            full: When ``True`` request detailed (``type=full``) records, else short summaries.

        Returns:
            A :class:`SessionHistory` with pass-through interaction records.
        """
        data = await self._client._get(
            f"/v1/sessions/{session_id}/history", params=_history_params(full)
        )
        return _history_from(data)

    async def results(self, session_id: str) -> SessionResults:
        """Fetch aggregated websearch preprocessor results for a session.

        Args:
            session_id: The session id.

        Returns:
            A :class:`SessionResults` with the query/goal/context and per-step results.
        """
        data = await self._client._get(f"/v1/sessions/{session_id}/websearch-preprocessor-results")
        return _results_from(data)
