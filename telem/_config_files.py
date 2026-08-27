"""Python mirror of the shared Telem config core (its TypeScript source).

The unified config
is read by TypeScript harness plugins *and* by Python readers — the MCP server, the skill
scripts, and (for credentials only) the SDK. This module is the Python half: the same
directory resolution, the same tolerance rules, the same "empty means absent" rule, the
same warning semantics, stdlib ``json`` only.

**It is a mirror, not an independent implementation.** A shared corpus is the
conformance gate; its own suite runs every case through
this module and the TypeScript suite runs the same cases through the TS
side. A behavior change belongs in a fixture first.

Private on purpose: nothing here is a supported public API, and the SDK's own stance is
unchanged — it reads ``~/.telem/credentials.json`` and NEVER a repo-local options file
(a library must not let a checked-out repository steer an arbitrary program's spend).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Callable, NamedTuple

#: Dot-directory, not a bare ``telem/``: ``telem/`` is a live Python package name.
TELEM_DIR_NAME = ".telem"
CONFIG_FILE_NAME = "telem.json"
#: Machine-written, 0600, never valid inside an options file.
CREDENTIALS_FILE_NAME = "credentials.json"
#: Relocates the user directory, ``GH_CONFIG_DIR``-style: the value IS the directory.
CONFIG_DIR_ENV = "TELEM_CONFIG_DIR"

Env = Mapping[str, str]


class TelemDir(NamedTuple):
    """A resolved directory, plus a warning when a requested relocation was refused."""

    path: str
    warning: str | None = None


class FileRead(NamedTuple):
    """A parsed config file. ``data is None`` means ABSENT, for any reason."""

    data: dict[str, Any] | None
    warning: str | None = None


# --------------------------------------------------------------------------- #
# Coercion (mirror of the shared config contract)
# --------------------------------------------------------------------------- #
#: The ONE blank definition both readers trim. A CLOSED, EXPLICIT LIST — never either
#: runtime's own idea of whitespace:
#:
#:   U+0020 space   U+0009 tab   U+000A LF   U+000D CR   U+000B VT   U+000C FF
#:   U+00A0 no-break space       U+FEFF byte-order mark  U+3000 ideographic space
#:
#: The six ASCII characters are the common floor. The other three are on the list because
#: they are what a PASTE actually deposits: a value copied out of a web page, a PDF, or a
#: CJK editor arrives padded with NBSP, a BOM, or an ideographic space, and forwarding one
#: to the server turns ``"tier": " max "`` into a 400. The shipped plugins' ``.trim()`` ate
#: all three, and dropping to ASCII-only would have been a silent regression against them.
#:
#: It stops there, and it is a LIST rather than a call to either runtime, because the
#: runtimes disagree: ``str.strip()`` also eats U+001C-1F, U+0085 (NEL), U+1680,
#: U+2000-200A, U+2028/9, U+202F and U+205F but NOT U+FEFF; JS
#: ``String.prototype.trim`` eats U+FEFF and all of those except U+001C-1F and U+0085.
#: Either runtime's definition would make the same file resolve to two different configs —
#: exactly what this mirror exists to prevent — so neither is used. Adding a character
#: means adding it to BOTH lists and to the shared corpus, deliberately.
#:
#: Consequence, pinned by the shared corpus: a lone U+00A0, U+FEFF or U+3000 is ABSENT on
#: both sides; a lone U+0085 or U+2028 is CONTENT on both. The mirror is ``trimBlank`` in
#: the shared config contract. Written as escapes on purpose: an invisible literal is
#: not a readable definition.
_BLANK_CHARS = " \t\n\r\v\f\u00a0\ufeff\u3000"


def _trim_blank(value: str) -> str:
    """Strip leading/trailing characters of the shared blank list (``_BLANK_CHARS``)."""
    return value.strip(_BLANK_CHARS)


def normalize_empty(value: Any) -> Any | None:
    """The normative "empty means absent" rule: ``[]``/``{}``/``""``/null are absent.

    Applies to EVERY key regardless of coercion class, so a value whose type does not
    match its key (``fullContent: {}``) is absent rather than falsely truthy. A string of
    nothing but ``_BLANK_CHARS`` counts as empty. ``False``/``0`` are values, not
    emptiness.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value if _trim_blank(value) else None
    if isinstance(value, (list, tuple, dict)):
        return value if len(value) else None
    return value


def as_name(value: Any) -> str | None:
    """A trimmed non-empty string, else absent."""
    raw = normalize_empty(value)
    if not isinstance(raw, str):
        return None
    return _trim_blank(raw) or None


def as_name_list(value: Any) -> list[str] | None:
    """A list of trimmed non-empty strings, else absent. Non-string items are dropped."""
    raw = normalize_empty(value)
    # ``bool`` is an ``int``, and a string is a sequence: only a real list/tuple qualifies.
    if not isinstance(raw, (list, tuple)):
        return None
    names = [_trim_blank(item) for item in raw if isinstance(item, str)]
    names = [name for name in names if name]
    return names or None


def as_flag(value: Any) -> bool | None:
    """A real boolean, else absent."""
    raw = normalize_empty(value)
    return raw if isinstance(raw, bool) else None


def as_overrides_map(value: Any) -> dict[str, dict[str, Any]] | None:
    """``{"<provider>": {<raw params>}}``: names trimmed, non-object entries dropped.

    The params themselves are passed to the server verbatim, so their shape is
    deliberately not policed here. A provider literally named ``__proto__`` is an
    ordinary key of an ordinary ``dict`` — the TS mirror had to build its map on
    ``Object.create(null)`` to say the same thing, and the corpus pins both.
    """
    raw = normalize_empty(value)
    if not isinstance(raw, dict):
        return None
    out: dict[str, dict[str, Any]] = {}
    for name, params in raw.items():
        provider = _trim_blank(name) if isinstance(name, str) else ""
        if provider and isinstance(params, dict):
            out[provider] = params
    return out or None


COERCERS: dict[str, Callable[[Any], Any]] = {
    "name": as_name,
    "nameList": as_name_list,
    "flag": as_flag,
    "overridesMap": as_overrides_map,
}

#: The JSON type a well-formed value of each coercion class has.
COERCION_JSON_TYPE = {
    "name": "string",
    "nameList": "array",
    "flag": "boolean",
    "overridesMap": "object",
}


class OptionSpec(NamedTuple):
    """One option key. Mirrors an entry of ``TELEM_OPTIONS`` in ``options.ts``."""

    key: str
    json_type: str
    coercion: str
    env: str | None
    env_aliases: tuple[str, ...]


#: The six option keys of the contract, in the TS table's order. Descriptions live on the TS
#: side only — they are interview/schema copy, and duplicating prose is how prose drifts;
#: its own suite pins this table against the generated schema.
TELEM_OPTIONS: tuple[OptionSpec, ...] = (
    OptionSpec("tier", "string", "name", "TELEM_TIER", ()),
    OptionSpec("fields", "array", "nameList", "TELEM_FIELDS", ()),
    OptionSpec(
        "providersInclude", "array", "nameList", "TELEM_PROVIDERS_INCLUDE", ("TELEM_PROVIDERS",)
    ),
    OptionSpec("providersExclude", "array", "nameList", "TELEM_PROVIDERS_EXCLUDE", ()),
    OptionSpec("fullContent", "boolean", "flag", "TELEM_FULL_CONTENT", ()),
    OptionSpec("providerOverrides", "object", "overridesMap", None, ()),
)

TELEM_OPTION_KEYS: tuple[str, ...] = tuple(option.key for option in TELEM_OPTIONS)


def csv_value(raw: str | None) -> list[str] | None:
    """A comma-separated env value: items stripped, empties dropped, all-empty is unset.

    An env var can therefore never express an explicit empty list.
    """
    if raw is None:
        return None
    return as_name_list(raw.split(","))


def option_from_env(spec: OptionSpec, env: Env) -> Any | None:
    """Read one key from an env bag; deprecated aliases rank strictly below the primary."""
    if spec.env is None:
        return None
    for name in (spec.env, *spec.env_aliases):
        raw = env.get(name)
        if raw is None:
            continue
        if spec.coercion == "nameList":
            value = csv_value(raw)
        elif spec.coercion == "flag":
            # Only exactly "1" turns a flag on; env can never turn one off.
            value = True if raw == "1" else None
        elif spec.coercion == "name":
            value = as_name(raw)
        else:
            value = None
        if value is not None:
            return value
    return None


# --------------------------------------------------------------------------- #
# The file layer (mirror of the shared config contract)
# --------------------------------------------------------------------------- #
def _home_dir(env: Env) -> str:
    """``HOME``, else ``USERPROFILE`` (Windows), else the OS's own answer."""
    return env.get("HOME") or env.get("USERPROFILE") or os.path.expanduser("~")


def _is_inside(candidate: str, project_root: str) -> bool:
    """Is ``candidate`` the project root or inside it?

    A LEXICAL check on absolute paths: no symlink resolution (``realpath`` on a path that
    does not exist yet behaves differently in Node and Python, and the shared corpus pins
    the two together) and no case folding. It is a guardrail against a repo redirecting
    config at itself, not a security boundary.
    """
    root = os.path.abspath(project_root)
    target = os.path.abspath(candidate)
    return target == root or target.startswith(root + os.sep)


def resolve_telem_dir(env: Env, project_root: str | None = None) -> TelemDir:
    """The user-level Telem directory: ``~/.telem``, or ``TELEM_CONFIG_DIR`` when set.

    A relocation pointing INSIDE ``project_root`` is refused with a warning (r1
    config-redirect-into-repo): a checked-in ``.telem`` must never promote itself to the
    user level. Callers with no project (the SDK) pass no root and are not checked.
    """
    fallback = os.path.join(_home_dir(env), TELEM_DIR_NAME)
    requested = _trim_blank(env.get(CONFIG_DIR_ENV) or "")
    if not requested:
        return TelemDir(fallback)
    if project_root and _is_inside(requested, project_root):
        return TelemDir(
            fallback,
            f"[telem] ignoring {CONFIG_DIR_ENV}={requested}: it points inside the project "
            f"({os.path.abspath(project_root)}); using {fallback} instead",
        )
    # Relative values resolve against the process cwd, as with every other tool that
    # takes a directory from the environment.
    return TelemDir(requested if os.path.isabs(requested) else os.path.abspath(requested))


def project_config_path(project_root: str) -> str:
    """``<project_root>/.telem/telem.json``."""
    return os.path.join(project_root, TELEM_DIR_NAME, CONFIG_FILE_NAME)


def user_config_path(env: Env, project_root: str | None = None) -> TelemDir:
    """``<telem dir>/telem.json``, plus any warning from resolving that directory."""
    directory = resolve_telem_dir(env, project_root)
    return TelemDir(os.path.join(directory.path, CONFIG_FILE_NAME), directory.warning)


def credentials_path(env: Env, project_root: str | None = None) -> TelemDir:
    """``<telem dir>/credentials.json``, plus any warning from resolving that directory."""
    directory = resolve_telem_dir(env, project_root)
    return TelemDir(os.path.join(directory.path, CREDENTIALS_FILE_NAME), directory.warning)


#: Everything a read may throw that must become a warning instead. ``ValueError`` covers
#: both a JSON parse error and ``open()`` on a path with an embedded NUL; ``RecursionError``
#: is what ``json.loads`` raises past ~9,998 nesting levels and is NOT a ``ValueError``, so
#: without it a hostile file escaped the "never raises" contract all the way through
#: ``Telem()``. The TS mirror's bare ``catch`` already covers its equivalents.
_READ_ERRORS = (ValueError, OSError, UnicodeDecodeError, RecursionError)

#: Byte-order mark. Windows Notepad writes one; ``json.loads`` rejects it.
_BOM = "\ufeff"


def _reject_json_constant(name: str) -> Any:
    """Refuse ``NaN``/``Infinity``/``-Infinity``, which Python's parser accepts by default.

    JSON has no such literals, and ``JSON.parse`` throws on them, so accepting them here
    made the same file usable in Python and malformed in TypeScript.
    """
    raise ValueError(f"{name} is not valid JSON")


def read_telem_file(path: str) -> FileRead:
    """Read one config file. It NEVER raises.

    A missing, unreadable, malformed, or non-object file is simply ABSENT and the next
    precedence level supplies the keys instead — a stray keystroke in telem.json can never
    fail a search. A MISSING file is silent (not having one is the normal case); anything
    else returns exactly one warning, so a broken file is visible without being noisy.

    Parsing is strict ``json.loads``: no comments, no trailing commas (r3 dropped JSONC
    precisely to keep this line identical in both languages), and no ``NaN``/``Infinity``.

    Bytes are read raw and decoded strictly, mirroring the TS side's fatal ``TextDecoder``:
    a file that is not valid UTF-8 is ABSENT with a "not valid UTF-8" warning rather than
    silently used. Reading in binary also keeps universal-newline translation out of the
    picture, so both readers see the same characters.

    Exactly ONE leading U+FEFF is then stripped, so a Notepad-written config works; a file
    with two byte-order marks stays malformed, identically in both languages.

    Host-parser limits are documented in the shared config contract (number range and
    nesting depth). They are the two places the mirrors cannot agree, and neither is
    policed: ``providerOverrides`` params are verbatim passthrough.
    """
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except (FileNotFoundError, NotADirectoryError):
        return FileRead(None)
    except OSError as error:
        return FileRead(None, f"[telem] ignoring {path}: {error.strerror or 'unreadable'}")
    except ValueError:  # e.g. an embedded NUL byte in the path
        return FileRead(None, f"[telem] ignoring {path}: unreadable")

    try:
        raw = data.decode("utf-8")
    except UnicodeDecodeError:
        return FileRead(None, f"[telem] ignoring {path}: not valid UTF-8")
    if raw.startswith(_BOM):
        raw = raw[1:]

    try:
        parsed = json.loads(raw, parse_constant=_reject_json_constant)
    except _READ_ERRORS as error:
        # An exception instance is always truthy, so test the MESSAGE: a RecursionError can
        # carry an empty one, and a warning that trails off after the colon says nothing.
        detail = str(error) or type(error).__name__
        return FileRead(None, f"[telem] ignoring {path}: {detail}")
    if not isinstance(parsed, dict):
        return FileRead(None, f"[telem] ignoring {path}: expected a JSON object")
    return FileRead(parsed)


def file_signature(path: str) -> str | None:
    """A stat signature (``mtimeMs:size``) for a path, or ``None`` when it cannot be stat'd.

    The mirror of ``fileSignature`` in the shared config contract: readers key a parse
    cache on it, so a file is re-read — and a malformed one re-warned — once per EDIT, not
    once per call. Size rides along with the mtime because two writes can land in one
    millisecond.

    Milliseconds, from ``st_mtime_ns``, to match Node's ``stat.mtimeMs``. The STRING is not
    a cross-language protocol (each language caches with its own), but the semantics are:
    same two fields, same order, ``None`` on any failure.
    """
    try:
        info = os.stat(path)
    except (OSError, ValueError):  # ValueError: an embedded NUL byte in the path
        return None
    return f"{info.st_mtime_ns / 1_000_000}:{info.st_size}"


# --------------------------------------------------------------------------- #
# Per-key resolution (mirror of the shared config contract)
# --------------------------------------------------------------------------- #
class Resolution(NamedTuple):
    """Resolved options, where each came from, and everything that was ignored."""

    #: Only the keys that resolved; an absent key is simply not present.
    values: dict[str, Any]
    #: ``"project"`` | ``"user"`` | ``"env"`` per resolved key, keyed like ``values``.
    sources: dict[str, str]
    #: In layer order. Callers decide how to surface these; this module never prints.
    warnings: list[str]
    #: The user-level directory actually used (after any refused relocation).
    telem_dir: str


def resolve_options(env: Env, project_root: str | None = None) -> Resolution:
    """Resolve every option key against project file > user file > env.

    Precedence is per KEY, not per file: a project file that sets ``tier`` does not hide
    the user file's ``fields``. Deliberately NOT here: the legacy ``.opencode``/``.pi``
    files, the opencode plugin-tuple override, and the tier/fields tie-break — those are
    surface-specific and stay with their surface.
    """
    warnings: list[str] = []

    project_data: dict[str, Any] | None = None
    if project_root:
        project = read_telem_file(project_config_path(project_root))
        project_data = project.data
        if project.warning:
            warnings.append(project.warning)

    telem_dir = resolve_telem_dir(env, project_root)
    if telem_dir.warning:
        warnings.append(telem_dir.warning)
    user = read_telem_file(os.path.join(telem_dir.path, CONFIG_FILE_NAME))
    if user.warning:
        warnings.append(user.warning)

    layers = (("project", project_data), ("user", user.data))

    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for spec in TELEM_OPTIONS:
        coerce = COERCERS[spec.coercion]
        resolved: Any = None
        level: str | None = None
        for layer_level, data in layers:
            if not data or spec.key not in data:
                continue
            candidate = coerce(data[spec.key])
            if candidate is not None:
                resolved, level = candidate, layer_level
                break
        if level is None:
            from_env = option_from_env(spec, env)
            if from_env is not None:
                resolved, level = from_env, "env"
        if level is not None:
            values[spec.key] = resolved
            sources[spec.key] = level

    return Resolution(values, sources, warnings, telem_dir.path)


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def read_credentials(env: Env, project_root: str | None = None) -> dict[str, str]:
    """Read ``~/.telem/credentials.json`` into ``{"apiKey"?, "baseUrl"?}``. It NEVER raises.

    SILENT by design, unlike the options files: this is the SDK's path, and a library that
    prints to a host application's stderr because a file it was never told about is
    malformed is a library that gets vendored around. A missing, malformed, or partial
    file simply supplies nothing, and the caller's existing defaults stand.

    Never-raises is load-bearing here in a way it is not for the options readers: this runs
    inside ``Telem()``, so anything that escapes turns "the user has an odd HOME" into "the
    SDK cannot be constructed". The path resolution is inside the guard too — a RELATIVE
    ``TELEM_CONFIG_DIR`` reaches ``os.path.abspath``, which calls ``os.getcwd()``, which
    raises ``FileNotFoundError`` when the process's cwd has been deleted underneath it.
    """
    try:
        path = credentials_path(env, project_root).path
        read = read_telem_file(path)
    except _READ_ERRORS:
        return {}
    if not read.data:
        return {}
    out: dict[str, str] = {}
    for key in ("apiKey", "baseUrl"):
        value = as_name(read.data.get(key))
        if value is not None:
            out[key] = value
    return out
