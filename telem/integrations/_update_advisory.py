"""Client update advisory for the Python SDK integrations.

When a search response carries a recommended ``telem-sdk`` version newer than the
one installed, the integration tells the application maintainer to bump their pin
-- at most once per process, on the developer's own channel. It never touches the
text a model reads, and it never updates anything on its own.

Only the openai and hermes integrations use this. langchain has no developer
notice channel and no per-call state to dedup against, so it is deliberately left
out. The version comparison and the update command are held to the same
checked-in fixtures the other surfaces run, so the message a Python user sees
cannot drift from the rest of the fleet.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from telem import _version
from telem._version_advisory import is_behind

#: The advisory map key every Python integration reads -- a fixed literal, never
#: the trajectory harness id (openai / hermes / langgraph). The server advertises
#: the SDK's recommended version under exactly this key.
_ADVISORY_KEY = "sdk"

#: The env var that turns the advisory off entirely, matching the plugin surfaces.
_OPT_OUT_ENV = "TELEM_NO_UPDATE_NOTICE"

#: How a maintainer updates the SDK. Pinned by a test against the shared command
#: fixtures so this literal cannot drift from what every other surface emits; it
#: is inlined rather than read at runtime because the fixtures do not ship in the
#: wheel.
_SDK_UPDATE_COMMAND = "telem-sdk can be updated with: pip install -U telem-sdk"

#: Recommended versions already advised in this process. Keyed by the version
#: string so a long-lived host notifies at most once per release, and a same
#: version seen again on a later call stays silent. In-memory only -- the SDK does
#: not persist a stamp the way the plugins do.
_ADVISED: set[str] = set()


def _opt_out() -> bool:
    """True when the caller has opted out of the advisory via the env var.

    Trimmed before the ``"1"`` compare so a stray surrounding space does not defeat
    the lever, matching the TS surfaces' ``(env ?? "").trim() === "1"``.
    """
    return (os.environ.get(_OPT_OUT_ENV) or "").strip() == "1"


def _message(recommended: str) -> str:
    """The maintainer-facing advisory naming the recommended version and command."""
    return (
        f"A newer telem-sdk is available: {recommended} "
        f"(installed {_version.__version__}); bump your telem-sdk pin. "
        f"{_SDK_UPDATE_COMMAND}"
    )


def recommended_map(response: Any) -> dict[str, Any] | None:
    """Pull the advisory's recommended-version map off a search response, or None.

    The response carries the server's parsed body on ``raw``; the advisory rides
    there as ``client_advisory.recommended``. Anything missing or the wrong shape
    -- an older server, a ``null`` field, a non-dict -- yields None (no notice),
    never an error.
    """
    raw = getattr(response, "raw", None)
    if not isinstance(raw, dict):
        return None
    advisory = raw.get("client_advisory")
    if not isinstance(advisory, dict):
        return None
    recommended = advisory.get("recommended")
    return recommended if isinstance(recommended, dict) else None


def maybe_warn(recommended: dict[str, Any] | None, emit: Callable[[str], None]) -> None:
    """Emit the update advisory on ``emit`` if this build is behind, at most once.

    ``recommended`` is the advisory's recommended-version map; the installed SDK
    version is compared against its ``sdk`` entry with the shared comparator.
    Stays silent on opt-out, on a missing / blank / malformed value, on an
    equal-or-newer build, and on any recommended version already advised in this
    process. ``emit`` is the caller's own developer channel -- a warning for
    openai, a log call for hermes -- and is never the text a model reads.
    """
    if _opt_out():
        return
    if not isinstance(recommended, dict):
        return
    version = recommended.get(_ADVISORY_KEY)
    if not isinstance(version, str) or version == "":
        return
    if not is_behind(_version.__version__, version):
        return
    if version in _ADVISED:
        return
    _ADVISED.add(version)
    # The caller's channel can itself raise -- openai passes ``warnings.warn``,
    # which RAISES under ``-W error`` / ``PYTHONWARNINGS=error`` / a pytest
    # ``filterwarnings = error`` app. A notice must never propagate into the search
    # turn, so the emit is wrapped exactly as the TS surfaces' outer try/catch is.
    # The version is already recorded as advised above, so a raising channel still
    # dedups once-per-process.
    try:
        emit(_message(version))
    except Exception:
        pass
