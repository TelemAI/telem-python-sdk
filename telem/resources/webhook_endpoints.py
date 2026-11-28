"""Webhook endpoints: a project's URLs that get the webhook deliveries of many monitors,
all of them or those with one of the endpoint's tags. The secret is shown only by
``create`` and ``rotate``.

``create`` and ``rotate`` are not sent again by the SDK: a second create makes a second
endpoint, and a second rotate drops the secret the first one kept."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from telem._base import path_segment
from telem.models import WebhookEndpoint, WebhookEndpointCreated, WebhookEndpointRotated

if TYPE_CHECKING:
    from telem.async_client import AsyncTelem
    from telem.client import Telem

_ROOT = "/v1/webhook-endpoints"


def _path(endpoint_id: str, suffix: str = "") -> str:
    if not endpoint_id:
        raise ValueError("a webhook endpoint id is required")
    return f"{_ROOT}/{path_segment(endpoint_id)}{suffix}"


def _create_body(url: str, tags: list[str] | None) -> dict[str, Any]:
    body: dict[str, Any] = {"url": url}
    if tags is not None:
        body["tags"] = tags
    return body


def _update_body(url: str | None, tags: list[str] | None) -> dict[str, Any]:
    # The router refuses a null url or tags, so an unset field is not sent.
    body: dict[str, Any] = {}
    if url is not None:
        body["url"] = url
    if tags is not None:
        body["tags"] = tags
    return body


def _list_from(data: Any) -> list[WebhookEndpoint]:
    rows = data.get("endpoints") if isinstance(data, dict) else None
    return [WebhookEndpoint.model_validate(row) for row in rows or [] if isinstance(row, dict)]


class WebhookEndpointsResource:
    """Synchronous webhook endpoint operations bound to a :class:`~telem.client.Telem`."""

    def __init__(self, client: Telem) -> None:
        self._client = client

    def create(self, url: str, *, tags: list[str] | None = None) -> WebhookEndpointCreated:
        """Add an endpoint to the API key's project. A project has at most 10.

        Args:
            url: An ``https`` URL, with no user name or password and no private IP address.
            tags: Monitor tags: the endpoint gets the deliveries of the project's monitors
                that carry any one of them. Leave it out, or send ``[]``, for every monitor.

        Returns:
            A :class:`WebhookEndpointCreated`. Keep its ``secret``: it is not shown again.

        Raises:
            APIStatusError: ``409`` (``too_many_endpoints``): the project has 10 already;
                ``422`` (``invalid_request``): the url or a tag is not correct.
        """
        data = self._client._post(_ROOT, _create_body(url, tags))
        return WebhookEndpointCreated.model_validate(data)

    def list(self) -> list[WebhookEndpoint]:
        """The project's endpoints, newest first, without their secrets."""
        return _list_from(self._client._get(_ROOT))

    def get(self, endpoint_id: str) -> WebhookEndpoint:
        """Get one endpoint, without its secret.

        Raises:
            NotFoundError: It does not exist, was deleted, or is another project's.
        """
        return WebhookEndpoint.model_validate(self._client._get(_path(endpoint_id)))

    def update(
        self, endpoint_id: str, *, url: str | None = None, tags: list[str] | None = None
    ) -> WebhookEndpoint:
        """Change the url, which keeps the secret, or replace the tags. Send at least one.

        Args:
            endpoint_id: The endpoint id.
            url: A new ``https`` URL.
            tags: The endpoint's new tags, in place of all of them. ``[]`` takes every
                monitor; leave it out to keep them.
        """
        body = _update_body(url, tags)
        data = self._client._request("PATCH", _path(endpoint_id), body)
        return WebhookEndpoint.model_validate(data)

    def delete(self, endpoint_id: str) -> None:
        """Delete an endpoint."""
        self._client._request("DELETE", _path(endpoint_id))

    def rotate(self, endpoint_id: str) -> WebhookEndpointRotated:
        """A new secret. For 24 hours, until ``previous_until``, each delivery is signed with
        the new secret and the one it replaces; a second rotate within them drops the older.

        Returns:
            A :class:`WebhookEndpointRotated`. Keep its ``secret``: it is not shown again.
        """
        data = self._client._post(_path(endpoint_id, "/rotate"), {})
        return WebhookEndpointRotated.model_validate(data)


class AsyncWebhookEndpointsResource:
    """Asynchronous webhook endpoint operations bound to an
    :class:`~telem.async_client.AsyncTelem`."""

    def __init__(self, client: AsyncTelem) -> None:
        self._client = client

    async def create(self, url: str, *, tags: list[str] | None = None) -> WebhookEndpointCreated:
        """Add an endpoint. See :meth:`WebhookEndpointsResource.create`."""
        data = await self._client._post(_ROOT, _create_body(url, tags))
        return WebhookEndpointCreated.model_validate(data)

    async def list(self) -> list[WebhookEndpoint]:
        """The project's endpoints, newest first, without their secrets."""
        return _list_from(await self._client._get(_ROOT))

    async def get(self, endpoint_id: str) -> WebhookEndpoint:
        """Get one endpoint. See :meth:`WebhookEndpointsResource.get`."""
        return WebhookEndpoint.model_validate(await self._client._get(_path(endpoint_id)))

    async def update(
        self, endpoint_id: str, *, url: str | None = None, tags: list[str] | None = None
    ) -> WebhookEndpoint:
        """Change the url or replace the tags. See :meth:`WebhookEndpointsResource.update`."""
        body = _update_body(url, tags)
        data = await self._client._request("PATCH", _path(endpoint_id), body)
        return WebhookEndpoint.model_validate(data)

    async def delete(self, endpoint_id: str) -> None:
        """Delete an endpoint."""
        await self._client._request("DELETE", _path(endpoint_id))

    async def rotate(self, endpoint_id: str) -> WebhookEndpointRotated:
        """A new secret. See :meth:`WebhookEndpointsResource.rotate`."""
        data = await self._client._post(_path(endpoint_id, "/rotate"), {})
        return WebhookEndpointRotated.model_validate(data)
