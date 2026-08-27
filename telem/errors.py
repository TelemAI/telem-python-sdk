"""Exception hierarchy for the Telem SDK."""

from __future__ import annotations


class TelemError(Exception):
    """Base class for all errors raised by the Telem SDK.

    Attributes:
        message: Human-readable error message.
        status_code: The HTTP status code that triggered the error, if any.
        body: The parsed response body (dict) or raw text that triggered the error, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: object | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body


class AuthError(TelemError):
    """Raised on HTTP 401 and 403 responses (missing or invalid credentials)."""


class BadRequestError(TelemError):
    """Raised on HTTP 400 responses (e.g. an unknown provider name)."""


class NotFoundError(TelemError):
    """Raised on HTTP 404 responses."""


class APIStatusError(TelemError):
    """Raised on any other non-2xx response (including 5xx server errors)."""


class TelemServerVersionError(TelemError):
    """Raised when the server does not echo the normalized schema version the SDK requires."""


class TelemConnectionError(TelemError):
    """Raised when the request never produced an HTTP response (timeout or transport failure)."""


def _detail_from_body(body: object) -> str | None:
    """Extract a human-readable message from a parsed JSON error body, if present.

    ``POST /v1/fetch`` serializes errors as
    ``{"error": {"code", "message", "param"?}}`` — the ``message`` is extracted here
    (the stable ``code`` stays available on the exception's ``body``). Every other
    endpoint, ``/v1/interactions`` included, keeps FastAPI's ``detail``: a string, or
    (validation errors) a *list* whose items' ``msg`` values are joined with ``", "``
    so the error reads as a message rather than a dumped object. Both shapes are
    handled here because one client talks to both.
    """
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
        detail = body.get("detail")
        if isinstance(detail, str) and detail:
            return detail
        if isinstance(detail, list) and detail:
            messages = [
                item["msg"]
                for item in detail
                if isinstance(item, dict) and isinstance(item.get("msg"), str)
            ]
            return ", ".join(messages) or str(detail)
    return None


def raise_for_status(status_code: int, body: object) -> None:
    """Raise the appropriate :class:`TelemError` subclass for a non-2xx response.

    Maps status codes to exception types:
    400 -> :class:`BadRequestError`, 401/403 -> :class:`AuthError`,
    404 -> :class:`NotFoundError`, any other non-2xx -> :class:`APIStatusError`.
    2xx status codes are treated as success and do not raise.

    The error message includes any string ``detail`` field found in the JSON body.

    Args:
        status_code: The HTTP status code of the response.
        body: The parsed JSON body (dict), raw text, or ``None``.
    """
    if 200 <= status_code < 300:
        return

    detail = _detail_from_body(body)
    if status_code == 400:
        message = detail or "Bad request"
        raise BadRequestError(message, status_code=status_code, body=body)
    if status_code in (401, 403):
        message = detail or "Authentication failed"
        raise AuthError(message, status_code=status_code, body=body)
    if status_code == 404:
        message = detail or "Not found"
        raise NotFoundError(message, status_code=status_code, body=body)

    message = detail or f"Unexpected status code {status_code}"
    raise APIStatusError(message, status_code=status_code, body=body)
