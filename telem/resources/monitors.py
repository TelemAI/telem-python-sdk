"""Monitors: a query that Telem runs on a schedule, read back with ``poll``."""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import TYPE_CHECKING, Any

from telem._normalize import build_search_response
from telem.models import Monitor, SearchResponse

if TYPE_CHECKING:
    from telem.async_client import AsyncTelem
    from telem.client import Telem


def _create_body(
    query: str,
    frequency: str,
    metadata: dict[str, Any] | None,
    webhook_url: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"query": query, "options": {"frequency": frequency}}
    if metadata is not None:
        body["metadata"] = metadata
    if webhook_url is not None:
        body["webhook_url"] = webhook_url
    return body


def _update_body(
    status: str | None,
    frequency: str | None,
    webhook_url: str | None,
) -> dict[str, Any]:
    body = {"status": status, "frequency": frequency}
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


def _poll_body(session: str | None, metadata: dict[str, Any] | None) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if session is not None:
        body["session_id"] = session
    if metadata is not None:
        body["metadata"] = metadata
    return body


def _list_params(limit: int | None) -> dict[str, Any] | None:
    return {"limit": limit} if limit is not None else None


def _monitors_from(data: Any) -> list[Monitor]:
    items = data.get("monitors", []) if isinstance(data, dict) else []
    return [Monitor.model_validate(item) for item in items]


def _path(monitor_id: str, suffix: str = "") -> str:
    if not monitor_id:
        raise ValueError("a monitor id is required")
    return f"/v1/monitors/{monitor_id}{suffix}"


class MonitorsResource:
    """Synchronous monitor operations bound to a :class:`~telem.client.Telem` client."""

    def __init__(self, client: Telem) -> None:
        self._client = client

    def create(
        self,
        query: str,
        *,
        frequency: str,
        metadata: dict[str, Any] | None = None,
        webhook_url: str | None = None,
    ) -> Monitor:
        """Create a monitor. Telem runs ``query`` on the schedule.

        Args:
            query: What to watch for, as a plain sentence.
            frequency: ``"1h"``, ``"6h"``, ``"1d"`` or ``"7d"``.
            metadata: Request metadata (query lineage keys), as in a search.
            webhook_url: An ``https`` URL that receives the new results.

        Returns:
            The new :class:`Monitor`. With ``webhook_url``, its ``webhook_secret``
            signs the deliveries; Telem shows it only in this response.
        """
        body = _create_body(query, frequency, metadata, webhook_url)
        return Monitor.model_validate(self._client._post("/v1/monitors", body))

    def list(self, *, limit: int | None = None) -> list[Monitor]:
        """List your monitors, newest first.

        Args:
            limit: How many to return (server default 50, maximum 200).
        """
        return _monitors_from(self._client._get("/v1/monitors", params=_list_params(limit)))

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
    ) -> Monitor:
        """Pause, resume or change a monitor. Send at least one field.

        Args:
            monitor_id: The monitor id.
            status: ``"paused"`` or ``"active"``.
            frequency: A new frequency.
            webhook_url: A new webhook URL, with a new ``webhook_secret`` in the
                response. ``""`` removes the webhook.

        Returns:
            The updated :class:`Monitor`.
        """
        body = _update_body(status, frequency, webhook_url)
        return Monitor.model_validate(self._client._request("PATCH", _path(monitor_id), body))

    def poll(
        self,
        monitor_id: str,
        *,
        session: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Get the results that are new since the last poll.

        Args:
            monitor_id: The monitor id.
            session: Record this poll in another session (default: the monitor's own).
            metadata: Request metadata (query lineage keys), as in a search.

        Returns:
            A :class:`SearchResponse`.
        """
        data = self._client._post(_path(monitor_id, "/poll"), _poll_body(session, metadata))
        return build_search_response(data)

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
        frequency: str,
        metadata: dict[str, Any] | None = None,
        webhook_url: str | None = None,
    ) -> Monitor:
        """Create a monitor. See :meth:`MonitorsResource.create`."""
        body = _create_body(query, frequency, metadata, webhook_url)
        return Monitor.model_validate(await self._client._post("/v1/monitors", body))

    async def list(self, *, limit: int | None = None) -> list[Monitor]:
        """List your monitors, newest first. See :meth:`MonitorsResource.list`."""
        data = await self._client._get("/v1/monitors", params=_list_params(limit))
        return _monitors_from(data)

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
    ) -> Monitor:
        """Pause, resume or change a monitor. See :meth:`MonitorsResource.update`."""
        body = _update_body(status, frequency, webhook_url)
        data = await self._client._request("PATCH", _path(monitor_id), body)
        return Monitor.model_validate(data)

    async def poll(
        self,
        monitor_id: str,
        *,
        session: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Get the results that are new since the last poll. See :meth:`MonitorsResource.poll`."""
        body = _poll_body(session, metadata)
        return build_search_response(await self._client._post(_path(monitor_id, "/poll"), body))

    async def delete(self, monitor_id: str) -> None:
        """Delete a monitor. A repeat deletes what a failed delete left."""
        await self._client._request("DELETE", _path(monitor_id))
