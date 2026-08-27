"""Install the bundled ``telem-search`` Claude Agent Skill into a skills directory.

Claude Code loads skills from ``~/.claude/skills/<name>/`` (personal) or
``<project>/.claude/skills/<name>/`` (project-scoped) — never from site-packages —
so shipping the skill inside the wheel is not enough on its own: something has to
copy it out. That something is this module, exposed as the ``telem-install-skill``
console script (see ``[project.scripts]`` in ``pyproject.toml``) and runnable as
``python -m telem.install_skill`` when a ``--user`` install leaves the console
script off ``PATH``.

The bundled copy lives at ``telem/_skills/telem-search/`` and is located with
:mod:`importlib.resources`, never by walking ``__file__`` — the latter breaks for
zipped and editable installs.

Two properties matter more than anything else here, because ``--force`` deletes:

* **Every path is resolved before it is touched, and the resolved path is the one
  used.** A symlinked *parent* (a ``.claude`` symlink committed to a repository,
  say) would otherwise redirect the replacement outside the skills directory
  entirely, and the path reported to the user would not be the path written. For
  ``--project`` the resolved path is additionally asserted to still be inside the
  project, on the value that is actually written to.
* **The install is staged, then swapped, and the previous install is kept until the
  swap succeeds.** A failure or an interrupt at *any* point leaves a complete copy
  of the previous install on disk. If it cannot be put back automatically the error
  message says where it is; it is never deleted to tidy up after a failure.

The second point is why the previous install is parked in a sibling
``.telem-search.backup-*`` directory rather than inside the staging directory: the
staging directory is unconditionally removed on the way out, including when the way
out is a ``KeyboardInterrupt``, and a backup that a cleanup path can delete is not a
backup. A ``.telem-search.backup-*`` directory left behind means some run failed
after moving the old install aside; its contents are the previous install, and
nothing here will ever delete it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

# Same protocol, two homes: importlib.abc.Traversable was the only spelling
# until 3.11 added importlib.resources.abc and left the old name working. The
# 3.11+ path goes first so we stop depending on the deprecated alias the moment
# the floor rises. novermin: static analysis reads the `try` branch as a hard
# 3.11 requirement and cannot see the fallback.
try:  # Python 3.11+  # novermin
    from importlib.resources.abc import Traversable
except ModuleNotFoundError:  # Python 3.10
    from importlib.abc import Traversable

SKILL_NAME = "telem-search"
"""Directory name the skill is installed under; matches ``name:`` in SKILL.md."""

_ANCHOR = "telem"
_BUNDLE_DIR = "_skills"
# Byproducts of running the skill's script in place; never worth copying out.
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")

#: Prefix for the private directory the new copy is built in. Removed on the way out.
_STAGING_PREFIX = f".{SKILL_NAME}.tmp-"

#: Prefix for the directory the *previous* install is parked in during the swap.
#: Deliberately distinct from the staging prefix so the two can never be confused:
#: staging directories are disposable, backup directories hold the user's data.
_BACKUP_PREFIX = f".{SKILL_NAME}.backup-"

#: A staging directory older than this was orphaned by a killed run (SIGKILL, a
#: power cut) rather than by a run happening right now, so a later run can clear it.
#: Generous on purpose: deleting a *live* run's staging directory would break it.
_STALE_STAGING_SECONDS = 24 * 60 * 60

#: Mode for directories this command creates. This is frequently what first creates
#: ``~/.claude``, and a skills directory holds instructions an agent reads and acts
#: on, so another local user must not be able to drop files in it. ``mkdir`` applies
#: the umask on top, which can only clear further bits.
_DIR_MODE = 0o700

#: Files/directories whose presence means "this is the root of a project".
_PROJECT_MARKERS = (".git", ".claude")

EXIT_OK = 0
"""The skill was installed."""

EXIT_EXISTS = 1
"""The destination already existed and ``--force`` was not given."""

EXIT_ERROR = 3
"""Anything else went wrong. (``2`` is argparse's own usage-error code.)"""


class SkillInstallError(RuntimeError):
    """Base class for failures reported as a message rather than a traceback."""


class SkillExistsError(SkillInstallError):
    """The destination already exists and ``force`` was not requested."""


class UnsafePathError(SkillInstallError):
    """The destination cannot be trusted: a symlinked component, an unusable home."""


def bundled_skill() -> Traversable:
    """Return the packaged skill directory as an :mod:`importlib.resources` traversable."""
    return resources.files(_ANCHOR).joinpath(_BUNDLE_DIR, SKILL_NAME)


def _write_out(source: Traversable, target: Path) -> None:
    """Copy a Traversable tree onto the filesystem, applying :data:`_IGNORE`.

    ``iterdir()`` is walked once into a list because it is an iterator on some
    Traversable implementations, and the ignore callable needs the whole name
    list before any child is visited.
    """
    target.mkdir(parents=True, exist_ok=True)
    children = list(source.iterdir())
    ignored = _IGNORE(str(source), [child.name for child in children])
    for child in children:
        if child.name in ignored:
            continue
        if child.is_dir():
            _write_out(child, target / child.name)
        else:
            (target / child.name).write_bytes(child.read_bytes())


@contextmanager
def _skill_source(source: Traversable) -> Iterator[Path]:
    """Yield the packaged skill as a real directory on disk.

    ``resources.as_file()`` does exactly this — but only from 3.12, which is
    what this package's floor used to be. On 3.10 and 3.11 it raises
    ``IsADirectoryError`` for a directory-valued Traversable that is not already
    a filesystem path, which is precisely the zip import the module docstring
    promises to support. Lowering the floor without this would have made
    ``telem-install-skill`` fail for zipped installs on the two versions it was
    lowered to reach.

    When the Traversable already *is* a path — every ordinary pip install, and
    editable installs — it is handed back untouched, so the common case still
    goes through :func:`shutil.copytree` with its mode preservation intact and
    pays no temp-directory round trip. Only a non-filesystem Traversable takes
    the copy below, which reproduces what 3.12's ``as_file()`` does.
    """
    try:
        direct: Path | None = Path(os.fspath(source))  # type: ignore[arg-type]
    except TypeError:
        # Not backed by the filesystem (a zip import). Deliberately resolved
        # before the yield: a `yield` inside the `try` would catch a TypeError
        # raised by the *caller's* block and fall through to a second yield.
        direct = None
    if direct is not None:
        yield direct
        return
    with tempfile.TemporaryDirectory(prefix=_STAGING_PREFIX) as tmp:
        materialized = Path(tmp) / SKILL_NAME
        _write_out(source, materialized)
        yield materialized


def _home() -> Path:
    """Return the user's home directory, refusing the ways it can be meaningless.

    The test each rejection has to pass is not "is this unusual" but "would Claude
    Code read what we are about to write". Claude Code takes the home directory from
    the same ``HOME``, so a *nonexistent* ``HOME`` is fine — the tree created under it
    is the tree Claude Code will scan — and it is deliberately accepted. Two values
    are not:

    * ``HOME=""`` — :func:`os.path.expanduser` returns ``"/"`` for an empty (as
      opposed to unset — that falls back to the password database, which is fine)
      ``HOME``, so the install would quietly land in ``/.claude/skills``, succeed as
      root in a container, and be read by nothing.
    * ``HOME=some/relative/path`` — ``~`` then depends on the current directory, so
      the install lands under whatever directory this command happened to run in and
      Claude Code, started somewhere else, resolves the same ``HOME`` to a different
      place. Nothing would ever read it.
    """
    raw = os.environ.get("HOME")
    if raw is not None and not raw.strip():
        raise UnsafePathError(
            "HOME is set but empty, so '~' expands to '/' and the skill would be "
            "installed into /.claude/skills, where Claude Code never looks. Unset HOME "
            "(the password database is then used) or point it at a real home directory."
        )
    if raw is not None and not os.path.isabs(raw):
        raise UnsafePathError(
            f"HOME is set to the relative path {raw!r}, so '~' depends on the current "
            f"directory: the skill would be installed under {raw}/.claude/skills relative "
            f"to wherever this command ran, and read from wherever Claude Code is started "
            f"instead. Point HOME at an absolute path."
        )
    home = Path.home()
    if home.parent == home:
        raise UnsafePathError(
            f"the home directory resolves to {home}, so the skill would be installed "
            f"into {home / '.claude' / 'skills'}, where Claude Code never looks. Point "
            f"HOME at a real home directory."
        )
    return home


def _declared_skills_dir(*, project: bool) -> Path:
    """Return the skills directory as *written*, before any symlink resolution."""
    base = Path.cwd() if project else _home()
    return base / ".claude" / "skills"


def _assert_within(resolved: Path, root: Path, *, declared: Path) -> None:
    """Refuse ``resolved`` unless it is inside ``root``, naming what escaped.

    This is a *containment* check rather than a symlink check, and the difference is
    the point. Asking "is this component a symlink?" and then resolving separately
    tests one value and uses another: flip ``.claude`` between a real directory and a
    symlink in between and the answer that was checked is not the answer that is used.
    Asking "is the path I am about to write to inside the project?" tests the value
    that is used, so there is nothing left to flip — and it also stops refusing the
    legitimate in-project ``.claude -> config/claude``, which a symlink check cannot
    tell apart from a hostile one.
    """
    if resolved.is_relative_to(root):
        return
    raise UnsafePathError(
        f"{declared} resolves to {resolved}, which is outside {root}. Refusing to "
        f"install into a project whose .claude path leaves the project directory — a "
        f"repository can commit that symlink, which would point --force at an unrelated "
        f"directory. Remove the symlink, or install for all projects with plain "
        f"`telem-install-skill`."
    )


def skills_dir(*, project: bool = False) -> Path:
    """Return the **resolved** skills directory Claude Code scans, personal by default.

    Resolving here is what keeps ``--force`` honest: every later operation, including
    the removal, happens on the real directory, and the path reported to the user is
    the path actually written.

    The two scopes treat a resolved path that leaves its base differently on purpose:

    * ``--project`` **refuses** it. The working directory is whatever repository the
      user happened to clone, so a committed ``.claude -> /somewhere/else`` symlink is
      attacker-controlled input, and there is no legitimate reason for a repo-local
      skills directory to leave the repository. A symlink that stays *inside* the
      project (``.claude -> config/claude``) is fine and is followed.
    * The personal scope **resolves and reports**. Symlinking ``~/.claude`` into a
      dotfiles repository is a real, common setup, so refusing it would break working
      installs; the resolved target is printed instead.

    The refusal here is for the *reported* path;  :func:`install_skill` re-asserts
    containment on the path it actually writes to, which is what closes the window
    between the two.

    Args:
        project: Resolve ``.claude/skills`` under the current working directory
            instead of ``~/.claude/skills``.

    Raises:
        UnsafePathError: ``project`` is true and the path resolves outside the working
            directory, or the home directory is unusable.
        OSError: the working directory or home directory could not be read.
    """
    declared = _declared_skills_dir(project=project)
    resolved = declared.resolve()
    if project:
        # Path.cwd() comes from getcwd(), which is already fully resolved.
        _assert_within(resolved, Path.cwd(), declared=declared)
    return resolved


def _existing_kind(path: Path) -> str | None:
    """Return ``"symlink"``/``"directory"``/``"file"`` for an existing path, else ``None``.

    ``is_symlink`` is checked first and never followed, so a dangling symlink counts
    as existing (which it does, as far as creating something in its place goes).
    """
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    if path.exists():
        return "file"
    return None


def _exists_message(destination: Path, kind: str) -> str:
    """Explain what is in the way and what ``--force`` would actually do to it."""
    if kind == "symlink":
        return (
            f"{destination} already exists as a symlink to {os.readlink(destination)}; "
            f"pass --force to replace the symlink itself with a real copy of the bundled "
            f"skill. The target directory is left untouched, but edits made through the "
            f"link would stop being picked up."
        )
    if kind == "file":
        return (
            f"{destination} already exists and is not a directory; pass --force to "
            f"replace it with the bundled skill."
        )
    return (
        f"{destination} already exists; pass --force to replace it "
        f"(this would overwrite any local edits to the skill)."
    )


def _make_dirs(path: Path) -> None:
    """Create ``path`` and its missing ancestors, each with an owner-only mode.

    ``Path.mkdir(parents=True)`` creates the *ancestors* with the default ``0o777``
    mode — it does not pass ``mode`` down the recursion — so ``~/.claude`` would
    inherit the ambient umask and be world-writable under ``umask 000``. Each level
    is therefore created explicitly.
    """
    missing: list[Path] = []
    current = path
    while not current.exists() and not current.is_symlink():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent

    if not current.is_dir():
        raise UnsafePathError(
            f"{current} exists and is not a directory, so {path} cannot be created. "
            f"Move it out of the way and re-run."
        )
    for directory in reversed(missing):
        # exist_ok: a concurrent run may have created it since the walk above.
        directory.mkdir(mode=_DIR_MODE, exist_ok=True)


def _resolve_destination(destination: Path) -> Path:
    """Return ``destination`` with its parent resolved and its own name kept as-is.

    Resolving the parent is the whole point: it is what a symlinked component would
    otherwise hide. The leaf is deliberately *not* resolved — the install replaces the
    name itself, so a symlink there must be renamed, never followed.
    """
    name = destination.name
    if not name or name in (".", ".."):
        raise UnsafePathError(f"{destination} is not a usable skill directory path.")
    return destination.parent.resolve() / name


@contextmanager
def _staging_area(root: Path) -> Iterator[Path]:
    """Yield a private temp directory inside ``root``, removed on the way out.

    Inside ``root`` rather than the system temp directory so the swap below is a
    same-filesystem rename, and so everything this function can ever delete lives
    under the resolved skills directory.

    Nothing that must survive a failure may be put in here: the removal below runs on
    every exit path, including :class:`KeyboardInterrupt`. The previous install goes
    to :func:`_reserve_backup` instead.
    """
    work = Path(tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=root))
    try:
        yield work
    finally:
        # ignore_errors: cleanup must never replace the real error with its own.
        shutil.rmtree(work, ignore_errors=True)


def _sweep_stale_staging(root: Path) -> None:
    """Remove staging directories left behind by runs that were killed outright.

    A ``KeyboardInterrupt`` unwinds through :func:`_staging_area` and cleans up after
    itself, but ``SIGKILL`` and a power cut do not, and the leftovers accumulate in the
    user's skills directory forever. They only ever contain a fresh copy of the bundled
    skill, so removing them loses nothing.

    Only directories older than :data:`_STALE_STAGING_SECONDS` are touched, because a
    *concurrent* run's staging directory is seconds old and deleting it would break a
    working install. Backup directories are never swept — they hold the user's data.
    """
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    cutoff = time.time() - _STALE_STAGING_SECONDS
    for entry in entries:
        if not entry.name.startswith(_STAGING_PREFIX) or entry.is_symlink():
            continue
        try:
            if entry.stat().st_mtime >= cutoff:
                continue
        except OSError:
            continue
        # ignore_errors: a leftover we cannot clear must not fail the install.
        shutil.rmtree(entry, ignore_errors=True)


def _reserve_backup(root: Path) -> Path:
    """Return a fresh, not-yet-existing path to park the previous install at.

    The path is a name *inside* a private directory rather than a name next to the
    destination: :func:`tempfile.mkdtemp` is what makes it collision-free, and the
    inner name has to not exist for :func:`os.replace` to be able to move either a
    directory or a file onto it. The directory is in ``root`` so the move is a
    same-filesystem rename that cannot half-finish.
    """
    holder = Path(tempfile.mkdtemp(prefix=_BACKUP_PREFIX, dir=root))
    return holder / SKILL_NAME


def _discard_backup(backup: Path) -> None:
    """Remove a backup that is no longer needed, never at the cost of the real result."""
    shutil.rmtree(backup.parent, ignore_errors=True)


def _restore_backup(backup: Path, destination: Path, cause: BaseException) -> None:
    """Put the previous install back, or say where it is if that is impossible.

    Raising over ``cause`` is deliberate. ``cause`` is why the install failed, which
    the user can retry; this is why their existing skill is not where they left it,
    which they cannot work out on their own — and an unrecoverable state nobody is
    told about is the same as data loss. That is also why the ``except`` is
    ``BaseException``: a second ``Ctrl-C``, landing while the first one is being
    recovered from, is exactly when the user most needs to be told where their skill
    went, and letting it propagate would print a bare ``KeyboardInterrupt`` instead.

    The filesystem, not the exception, decides which of those happened: ``os.replace``
    is one atomic rename, so an interrupt raised *after* it landed leaves nothing to
    report and the original ``cause`` is allowed through untouched.
    """
    try:
        os.replace(backup, destination)
    except BaseException as failure:
        if _existing_kind(backup) is None:
            # The rename did land; only what came after it was interrupted.
            _discard_backup(backup)
            return
        raise SkillInstallError(
            f"the install failed ({cause!r}) and {destination} could not be put back "
            f"({failure!r}). Nothing was lost: the previous install is complete at "
            f"{backup}, and no later run will delete it. Restore it with:\n"
            f"    mv {backup} {destination}"
        ) from cause
    _discard_backup(backup)


def install_skill(destination: Path, *, force: bool = False, within: Path | None = None) -> Path:
    """Copy the bundled skill to ``destination``.

    The copy is staged into a sibling temp directory and swapped in with
    :func:`os.replace`. The previous install is moved aside into a second sibling
    directory first and deleted only once the swap has succeeded, so it survives a
    read-only parent, a full disk, a concurrent run, an ``OSError`` from the swap
    itself, and a ``Ctrl-C`` landing anywhere in the window — recovery is attempted
    automatically, and when even that fails the raised message says which path holds
    the previous install. ``--force`` can never recurse into anything outside the
    resolved parent directory (an existing symlink is renamed, so its target is never
    followed).

    Args:
        destination: Directory to create, e.g. ``~/.claude/skills/telem-search``.
            Its parent is resolved first; the returned path is what was written.
        force: Replace ``destination`` if it already exists. Without this an
            existing path is left untouched and :class:`SkillExistsError` is raised —
            silently clobbering a customized skill is not acceptable.
        within: Refuse to write anywhere outside this directory. Passed for
            ``--project``, where the destination comes from a repository that may have
            committed a ``.claude`` symlink; checked against the *resolved* path this
            call will actually write to.

    Returns:
        The resolved destination path.

    Raises:
        SkillExistsError: ``destination`` exists and ``force`` is false — including
            when it came into existence while the copy was running.
        UnsafePathError: ``destination`` is not a usable path, resolves outside
            ``within``, or something that is not a directory sits where a parent
            directory has to be created.
        SkillInstallError: the swap failed *and* the previous install could not be put
            back; the message names the directory holding it.
        OSError: the copy or the swap failed (permissions, disk space, ...).
    """
    destination = _resolve_destination(destination)
    if within is not None:
        _assert_within(destination, within.resolve(), declared=destination)
    root = destination.parent

    kind = _existing_kind(destination)
    if kind is not None and not force:
        raise SkillExistsError(_exists_message(destination, kind))

    _make_dirs(root)
    _sweep_stale_staging(root)
    backup: Path | None = None
    with _staging_area(root) as work:
        staged = work / SKILL_NAME
        # Materializes a real directory even when the package is a zip import —
        # see _skill_source for why this is not resources.as_file().
        with _skill_source(bundled_skill()) as source:
            shutil.copytree(source, staged, ignore=_IGNORE)

        # Probed a second time because copying takes long enough for a skill to be
        # created in the window, and `force` has to gate the destructive step, not
        # merely an earlier reading of it. Without this re-check a skill written
        # during the copy is moved aside and deleted by a `force=False` call that
        # then reports success.
        kind = _existing_kind(destination)
        if kind is not None and not force:
            raise SkillExistsError(_exists_message(destination, kind))

        swapped = False
        try:
            if kind is not None:
                backup = _reserve_backup(root)
                os.replace(destination, backup)
            os.replace(staged, destination)
            swapped = True
        except BaseException as exc:
            # BaseException, not OSError: a Ctrl-C in this window is the likeliest way
            # to reach it, and it must not be the one path that skips the restore.
            if backup is not None and not swapped:
                if _existing_kind(backup) is not None:
                    _restore_backup(backup, destination, exc)
                else:
                    # The move aside never happened, so the reserved directory is
                    # empty litter -- and claiming a backup exists would be a lie.
                    _discard_backup(backup)
            raise
    if backup is not None:
        _discard_backup(backup)
    return destination


def _warn(prog: str, message: str) -> None:
    """Write a non-fatal note to stderr, so stdout stays just the installed path."""
    print(f"{prog}: {message}", file=sys.stderr)


def _warn_if_not_project_root(prog: str, cwd: Path) -> None:
    """Warn when ``--project`` was run somewhere Claude Code will not read.

    ``--project`` writes to ``./.claude/skills``, and Claude Code only reads the
    ``.claude`` of the directory it was started in. Run from ``<project>/src/deep``
    the install genuinely succeeds and is genuinely never loaded. The directory is
    not corrected automatically: guessing an ancestor would silently write outside
    the directory the user named.
    """
    if any((cwd / marker).exists() for marker in _PROJECT_MARKERS):
        return
    _warn(
        prog,
        f"warning: {cwd} contains no {' or '.join(_PROJECT_MARKERS)}, so it may not be "
        f"the project root. Claude Code only loads project skills from the .claude of "
        f"the directory it was started in — re-run from that directory, or drop "
        f"--project to install for every project.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Install the bundled skill (console script entry point).

    Returns:
        ``0`` on success; ``1`` when the destination already exists and ``--force``
        was not given; ``3`` for every other failure — an unusable path, a symlinked
        component in a ``--project`` path, or an :class:`OSError` such as a
        permission denial or a non-directory in the way. (``argparse`` exits ``2``
        by itself for a usage error.)
    """
    parser = argparse.ArgumentParser(
        prog="telem-install-skill",
        description=(
            f"Install the bundled {SKILL_NAME} Claude Agent Skill into a skills "
            "directory Claude Code scans."
        ),
    )
    parser.add_argument(
        "--project",
        action="store_true",
        help="install into ./.claude/skills (this project only) instead of ~/.claude/skills",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing installation instead of failing",
    )
    args = parser.parse_args(argv)

    try:
        within = None
        if args.project:
            # Captured once and passed down, so the containment check inside
            # install_skill runs against the directory this invocation started in.
            within = Path.cwd()
            _warn_if_not_project_root(parser.prog, within)
        declared = _declared_skills_dir(project=args.project) / SKILL_NAME
        destination = skills_dir(project=args.project) / SKILL_NAME
        if destination != declared:
            _warn(parser.prog, f"note: {declared} resolves to {destination}")
        if args.force and destination.is_symlink():
            _warn(
                parser.prog,
                f"note: {destination} is a symlink to {os.readlink(destination)}; "
                f"replacing the symlink with a real directory. The target is left as it "
                f"is, so edits made through the link will no longer be picked up.",
            )
        installed = install_skill(destination, force=args.force, within=within)
    except SkillExistsError as exc:
        _warn(parser.prog, str(exc))
        return EXIT_EXISTS
    except SkillInstallError as exc:
        _warn(parser.prog, str(exc))
        return EXIT_ERROR
    except OSError as exc:
        _warn(parser.prog, f"could not install the skill: {exc}")
        return EXIT_ERROR
    print(f"Installed the {SKILL_NAME} skill to {installed}")
    return EXIT_OK


if __name__ == "__main__":  # `python -m telem.install_skill`.
    sys.exit(main())
