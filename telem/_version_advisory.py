"""Client update advisory: decide whether this build is behind the recommended one.

A faithful mirror of the shared TypeScript comparator. Both implementations run the
same checked-in corpus, so a string-compare bug that ranks ``"0.2.10"`` below
``"0.2.9"`` fails a test loudly instead of shipping on one surface.

Two rules the whole feature relies on:
    * numeric-segment compare -- ``"0.2.10"`` is NEWER than ``"0.2.9"``, never older.
    * malformed input on either side degrades to ``False`` (silence), never an
      exception, so a stray version value can never crash a client turn.
"""

from __future__ import annotations

import re

_NUMERIC = re.compile(r"^\d+$")

_Parsed = tuple[list[int], list[str]]


def _parse(value: object) -> _Parsed | None:
    """Parse ``x.y.z`` / ``x.y.z-pre.tags``; build metadata after ``+`` is ignored.

    Returns ``(release, prerelease)`` or ``None`` for anything that is not a run of
    numeric release segments -- the signal every caller turns into silence.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text == "":
        return None
    plus = text.find("+")
    if plus != -1:
        text = text[:plus]
    dash = text.find("-")
    core = text if dash == -1 else text[:dash]
    pre_text = "" if dash == -1 else text[dash + 1 :]
    release: list[int] = []
    for part in core.split("."):
        if not _NUMERIC.match(part):
            return None
        release.append(int(part))
    # A dash with nothing after it is malformed, not a stable release.
    if dash != -1 and pre_text == "":
        return None
    pre = [] if pre_text == "" else pre_text.split(".")
    for identifier in pre:
        if identifier == "":
            return None
    return release, pre


def _compare_pre(a: list[str], b: list[str]) -> int:
    """-1 / 0 / 1 for a<b / a==b / a>b over prerelease identifier lists."""
    # A stable release (no prerelease) outranks any prerelease of the same core.
    if not a and not b:
        return 0
    if not a:
        return 1
    if not b:
        return -1
    for x, y in zip(a, b):
        x_num = bool(_NUMERIC.match(x))
        y_num = bool(_NUMERIC.match(y))
        if x_num and y_num:
            diff = int(x) - int(y)
            if diff != 0:
                return -1 if diff < 0 else 1
        elif x_num != y_num:
            # A numeric identifier has lower precedence than an alphanumeric one.
            return -1 if x_num else 1
        elif x != y:
            return -1 if x < y else 1
    if len(a) == len(b):
        return 0
    return -1 if len(a) < len(b) else 1


def _compare(a: _Parsed, b: _Parsed) -> int:
    """-1 / 0 / 1 for a<b / a==b / a>b over two already-parsed versions."""
    a_release, a_pre = a
    b_release, b_pre = b
    width = max(len(a_release), len(b_release))
    for i in range(width):
        x = a_release[i] if i < len(a_release) else 0
        y = b_release[i] if i < len(b_release) else 0
        if x != y:
            return -1 if x < y else 1
    return _compare_pre(a_pre, b_pre)


def is_behind(local: str, recommended: str) -> bool:
    """Is ``local`` strictly older than ``recommended`` -- should this client notify?

    Malformed input on either side returns ``False`` (silent), never raises.
    """
    a = _parse(local)
    b = _parse(recommended)
    if a is None or b is None:
        return False
    return _compare(a, b) < 0
