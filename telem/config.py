"""The one public accessor for the unified Telem config files.

A project
carries ONE Telem config and whichever harness runs a search inside that project reads
it. :func:`resolve_search_options` is how a **Python** reader gets at it —
The Telem plugins and the bundled ``telem-search``
Claude Skill script both go through this function, and so can your own script.

Three levels, per KEY, top wins (levels 2, 5 and 6 of the spec's ladder — the harness
levels 1/3/4 are host-specific and stay with their hosts)::

    <project_root>/.telem/telem.json   # only when you pass project_root
    ~/.telem/telem.json                # TELEM_CONFIG_DIR relocates the directory
    TELEM_* environment variables

Per key, not per file: a project file that sets ``tier`` does not hide the user file's
``fields``.

**The project level is opt-in.** ``resolve_search_options()`` with no ``project_root``
reads no repo-local file at all, and the ``Telem``/``AsyncTelem`` clients never call
this function — a library must not let a checked-out repository steer an arbitrary
program's spend (the contract, "Python SDK: NEVER"). A *harness* whose working directory is
the user's project (the MCP server, a skill script run by the agent) passes
``project_root=os.getcwd()`` deliberately, and says so in its docs.

**It never raises.** A missing, unreadable, malformed, hostile or wrong-typed file is
ABSENT, and the next level supplies the key instead; everything ignored comes back in
:attr:`SearchOptionsResolution.warnings` for the caller to print (this module never
prints).

The reading itself lives in :mod:`telem._config_files`, the corpus-pinned Python mirror
of ``config-core``. What this module adds is the three composition rules that a *caller*
would otherwise have to reinvent, and that the TypeScript plugins apply in
the shared config contract: the tier/fields tie-break, the include/exclude
subtraction, and the ``providerOverrides`` membership guard. They run in that order,
because each one decides what the next one sees.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from telem._config_files import resolve_options

__all__ = ["SearchOptionsResolution", "resolve_search_options"]

#: Config key -> ``client.search()`` keyword argument. The ONE place the two
#: vocabularies meet: the config file speaks the cross-harness names of the contract, the
#: SDK speaks Python. Both ends are pinned by tests.
#:
#: Private and frozen on purpose. It is the implementation of
#: :meth:`SearchOptionsResolution.search_kwargs`, not a table callers should read or
#: extend — a caller who mutated it would silently re-route every later resolution in
#: the process, and a caller who read it would be depending on a mapping that changes
#: whenever the option table does.
_SEARCH_KWARGS: Mapping[str, str] = MappingProxyType(
    {
        "tier": "tier",
        "fields": "fields",
        "providersInclude": "providers_include",
        "providersExclude": "providers_exclude",
        "fullContent": "include_full_content",
        "providerOverrides": "provider_overrides",
    }
)

#: Precedence rank of each level, lower wins. Mirrors ``layerRank`` in
#: the shared config contract over the three levels a Python reader has, plus
#: ``"call"`` — the caller's own override, above every level, which is the harness
#: slot (opencode's plugin-tuple object, a CLI's ``--providers``) in Python form.
_SOURCE_RANK = {"call": 0, "project": 1, "user": 2, "env": 3}


@dataclass(frozen=True)
class SearchOptionsResolution:
    """Options resolved from the unified config, and everything ignored on the way.

    Frozen all the way down to its own fields: the containers are a read-only mapping
    and a tuple, so ``frozen=True`` means what it says. (The values *inside*
    ``values`` are the JSON the user wrote — lists and dicts, handed on verbatim,
    because they are about to be serialized as the request body.)
    """

    #: Only the keys that resolved, under the contract names (``providersInclude``, …).
    #: An absent key is simply not present — never ``None``.
    values: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    #: ``"call"`` | ``"project"`` | ``"user"`` | ``"env"`` per resolved key, keyed like
    #: ``values``. Recorded BEFORE the composition rules, so a key dropped by the
    #: tier/fields tie-break or the include/exclude subtraction still shows where it
    #: came from.
    sources: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    #: Human-readable, in resolution order: refused relocations, ignored files, and
    #: values the composition rules dropped. Print them to **stderr** — never into
    #: text a model reads.
    warnings: tuple[str, ...] = ()

    def search_kwargs(self) -> dict[str, Any]:
        """The resolved options as ``client.search()`` keyword arguments.

        Splat it into a call — ``client.search(query, **options.search_kwargs())`` —
        where it beats the client's own env-derived defaults, which is exactly
        "a config file outranks the environment", per key. Unresolved keys are absent
        rather than ``None``, so they leave the client's defaults alone.

        A fresh, ordinary ``dict`` each call: it is made to be splatted and edited.
        """
        return {_SEARCH_KWARGS[key]: value for key, value in self.values.items()}


def resolve_search_options(
    *,
    project_root: str | None = None,
    env: Mapping[str, str] | None = None,
    providers_include: Sequence[str] | None = None,
) -> SearchOptionsResolution:
    """Resolve the unified Telem config: project file > user file > ``TELEM_*`` env.

    Despite the name it resolves BOTH files and the environment — the whole three-level
    ladder — and returns what a search should run with.

    Args:
        project_root: The user's project directory, enabling the project level
            (``<project_root>/.telem/telem.json``). ``None`` — the default — reads no
            repo-local file at all. Harnesses whose cwd is the project pass
            ``os.getcwd()``.
        env: Environment to read, defaulting to :data:`os.environ`. Pass a plain dict
            to resolve against something other than the process environment.
        providers_include: A caller-supplied provider selection that outranks every
            level, for a surface that has one — a CLI's ``--providers`` flag, a host's
            own options slot. It is applied BEFORE the composition rules rather than
            after, so the rules see the include the request will actually carry: an
            override this selection licenses survives, one it strands is dropped, and a
            configured ``providersExclude`` is subtracted from it rather than sent
            alongside it (which the server answers with a 400). Empty or all-blank is
            ignored, exactly like an empty value in a file.

    Returns:
        A :class:`SearchOptionsResolution`. Never raises: everything that went wrong is
        a warning.
    """
    environ = os.environ if env is None else env
    try:
        resolution = resolve_options(environ, project_root)
    except Exception as error:  # pragma: no cover - defense in depth
        # `resolve_options` is already never-raises; this is the outer guard that
        # keeps an odd environment (a deleted cwd, an exotic HOME) from turning
        # "read the config" into "the tool call failed".
        return SearchOptionsResolution(warnings=(f"[telem] ignoring the Telem config: {error}",))

    values = dict(resolution.values)
    sources = dict(resolution.sources)
    warnings = list(resolution.warnings)

    selected = [
        name for name in (providers_include or []) if isinstance(name, str) and name.strip()
    ]
    if selected:
        values["providersInclude"] = selected
        sources["providersInclude"] = "call"

    _compose_tier_fields(values, sources, warnings)
    # BEFORE the overrides guard: the guard checks membership against the include the
    # request will actually carry, which is the SUBTRACTED one.
    _compose_providers(values, warnings)
    _guard_provider_overrides(values, warnings)
    return SearchOptionsResolution(
        values=MappingProxyType(values),
        sources=MappingProxyType(sources),
        warnings=tuple(warnings),
    )


def _rank(sources: Mapping[str, str], key: str) -> int:
    """Precedence rank of the level that supplied ``key``; unknown levels rank last."""
    return _SOURCE_RANK.get(sources.get(key, ""), len(_SOURCE_RANK))


def _compose_tier_fields(
    values: dict[str, Any], sources: Mapping[str, str], warnings: list[str]
) -> None:
    """A request may carry ``tier`` OR ``fields``, never both (the V2 contract is a 400).

    They resolve independently, so compose them: the MORE SPECIFIC level wins, and on a
    true same-level tie ``fields`` wins — it is the finer instrument, and "fields
    replaces tier" is the direction the server documents. Mirrors ``composeTierFields``
    in the shared config contract; announced, because silently dropping a value the
    user wrote down is how a config becomes a mystery.
    """
    tier = values.get("tier")
    fields = values.get("fields")
    if not isinstance(tier, str) or not isinstance(fields, list):
        return
    keep_tier = _rank(sources, "tier") < _rank(sources, "fields")
    del values["fields" if keep_tier else "tier"]
    kept = "keeping tier, ignoring fields" if keep_tier else "keeping fields, ignoring tier"
    warnings.append(
        f"[telem] both tier ({tier}) and fields ({', '.join(fields)}) are configured, "
        f"but a request may carry only one — {kept} "
        "(the more specific source wins; on a tie, fields)."
    )


def _compose_providers(values: dict[str, Any], warnings: list[str]) -> None:
    """When both provider halves resolve, ``exclude`` is subtracted from ``include``
    here and only the subtracted ``include`` is sent.

    The server's rule 3 answers ``search.providers.include`` and ``exclude`` naming the
    same provider with a 400, and answers an ``include`` that resolves to nothing with a
    400 as well. Both halves resolving is not exotic: they resolve PER KEY over the
    ladder, so a user file's ``providersExclude`` meets a project file's
    ``providersInclude`` the moment two people configure the thing they each cared
    about. Forwarding that pair verbatim made a checked-in config break EVERY search in
    the repository, with a message about the request rather than about the file.

    Subtracting is not a reinterpretation: ``include`` REPLACES the deployment's default
    set, so once it resolves it already names the whole candidate set, and "run
    these, minus those" is exactly ``include \\ exclude``. It is also what
    ``BaseClient._resolve_search_options`` has always computed for the same pair, so the
    wire body is now the same bytes from every surface rather than two dialects of one
    config file.

    An emptied ``include`` drops BOTH halves: the config states an empty provider set,
    which no search can run, so the deployment's default set runs — loudly — instead of
    every call failing. (``include: []`` on the wire is the server's 400 by design; that
    answer belongs to a caller who explicitly asked for nothing, not to a pair of config
    files that happened to cancel out.) A disjoint pair is silent — dropping a no-op is
    not news. ``exclude`` ALONE keeps riding verbatim: with no ``include`` the set it
    subtracts from is the deployment default, which this process cannot enumerate.

    Mirrors ``composeProviders`` in the shared config contract and in the pi skill's
    vendored reader; the shared corpus pins that the three agree on the byte.
    """
    include = values.get("providersInclude")
    exclude = values.get("providersExclude")
    if not isinstance(include, list) or not isinstance(exclude, list):
        return

    excluded = set(exclude)
    kept = [name for name in include if name not in excluded]
    dropped = [name for name in include if name in excluded]
    del values["providersExclude"]

    if kept:
        values["providersInclude"] = kept
        if not dropped:
            return
        warnings.append(
            f"[telem] providersExclude removes {', '.join(dropped)} from providersInclude: "
            f"this search runs {', '.join(kept)}. A request may not name the same provider in "
            "both halves, so the exclusion is applied to the config and only providersInclude "
            "is sent."
        )
        return

    del values["providersInclude"]
    warnings.append(
        f"[telem] providersExclude ({', '.join(exclude)}) removes every provider in "
        f"providersInclude ({', '.join(include)}): this config resolves to an EMPTY provider "
        "set, which no search can run, so BOTH halves are dropped and the deployment's default "
        "provider set is running instead."
    )


def _guard_provider_overrides(values: dict[str, Any], warnings: list[str]) -> None:
    """``providerOverrides`` applies only alongside ``providersInclude``, and only to
    providers named there. Anything else is DROPPED rather than forwarded.

    The reason is what a client can KNOW: an override must name a provider the request
    actually selects, and the resolved ``providersInclude`` is the only statement of
    that set this process holds. With ``providersInclude`` unset, the deployment's own
    default set is running — a set no client can enumerate — so no override can be
    checked against it, and an unverifiable override is not forwarded.

    It runs LAST, so "the resolved providersInclude" means the one the request will
    carry: an override on a provider the exclusion just subtracted away is stranded, and
    is dropped like any other.

    Mirrors ``guardProviderOverrides`` in the shared config contract byte for
    behavior, so the same config file means the same thing on every surface.
    """
    overrides = values.get("providerOverrides")
    if not isinstance(overrides, dict):
        return
    include = values.get("providersInclude")
    selected = set(include) if isinstance(include, list) else set()
    kept = {name: params for name, params in overrides.items() if name in selected}
    dropped = [name for name in overrides if name not in selected]
    if not dropped:
        return
    if kept:
        values["providerOverrides"] = kept
    else:
        del values["providerOverrides"]

    because = (
        "providerOverrides applies only alongside providersInclude, and providersInclude "
        "is not set, so the deployment's own default provider set is running"
        if include is None
        else "providerOverrides applies only to providers named in the resolved "
        f"providersInclude ({', '.join(include)})"
    )
    quoted = ", ".join(f'"{name}"' for name in dropped)
    them = "them" if len(dropped) > 1 else "it"
    warnings.append(
        f"[telem] ignoring providerOverrides for {', '.join(dropped)}: {because}. "
        f"Add {quoted} to providersInclude in the same config to use {them} — noting that "
        "providersInclude REPLACES the deployment's default provider set with exactly the "
        "providers you list, rather than adding to it."
    )
