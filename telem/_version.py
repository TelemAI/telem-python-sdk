"""The single place the package version is written.

Everything else derives from this: ``telem.__version__`` re-exports it, ``_base``
builds the default User-Agent from it, and ``pyproject.toml`` reads it at build
time via ``[tool.hatch.version]`` rather than carrying its own copy.

That indirection exists because the copies drifted. The default User-Agent read
``telem-sdk/0.1.0`` in the wheels published as 0.1.0, 0.1.1 AND 0.1.2 — three
releases identifying themselves as the first one — because a version bump meant
editing one string in ``pyproject.toml`` and another in ``__init__.py``, and the
third one here had no reason to be looked at. Bumping a release should be a
one-line change; if you are editing a version string anywhere else, something has
regressed back to the shape that caused that.
"""

__version__ = "0.1.6"
