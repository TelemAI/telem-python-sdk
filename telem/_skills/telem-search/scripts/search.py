"""One-shot web search against a Telem backend via the telem-sdk Python client.

Companion to the telem-search skill (see ../SKILL.md). Self-contained by design:
stdlib, the public ``telem`` package, and ``httpx`` (a hard dependency of
telem-sdk) only, so the skill directory keeps working when copied out of this
repository.

Run it from the skill directory with the interpreter that has telem-sdk installed,
or standalone with ``uv run --with telem-sdk scripts/search.py`` (from a checkout,
``uv run --with /path/to/telem-python-sdk`` instead)::

    export TELEM_API_KEY=...          # required
    # TELEM_BASE_URL defaults to the hosted https://router.telem.ai; set it to
    # set it only if Telem gave you a different endpoint.
    python scripts/search.py "query" --goal "..."
    python scripts/search.py "q1" "q2" --session <id>
    python scripts/search.py "q" --providers exa,tavily

Search options come from the unified Telem config, read fresh on every invocation
(the script is one-shot, so "per call" and "per process" are the same thing):

    <cwd>/.telem/telem.json   the project file — run this from the project root
    ~/.telem/telem.json       the user file
    TELEM_*                   the environment

per key, top wins. That is `telem.resolve_search_options()`, the SDK's one public
accessor for those files — this script stays stdlib + public `telem` API only, so it
keeps working when the skill directory is copied out of the repository. `--providers`
is passed to it as the call-level selection, so it outranks every level and the
composition rules run once against it. Anything the config layer had to ignore is
printed to stderr, never mixed into the results.

Multiple queries are batched into ONE interaction sharing ONE session and are
rendered as '### Query N' sections. Pass --session on every follow-up search for
the same task; pass --goal only on the first search of a new task.
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx  # hard dependency of telem-sdk, so safe for a standalone copy

from telem import SearchResponse, SearchResult, Telem, TelemError, resolve_search_options

# The formatting below is intentionally duplicated from the stdio MCP server
# (format_search_text): importing that private module here would break the
# skill when this directory is copied out of the repository.


def _block(result: SearchResult, max_len: int) -> str:
    """Render one result as a URL/Title/Content block, content truncated to max_len."""
    return f"URL: {result.url}\nTitle: {result.title}\nContent: {result.content[:max_len]}"


def format_results(response: SearchResponse, max_len: int) -> str:
    """Flat blocks for a single query; '### Query N: <query>' sections for a batch."""
    labels: dict[int, str] = {}
    grouped: dict[int, list[SearchResult]] = {}
    for run in response.by_provider:
        grouped.setdefault(run.batch_index, []).extend(run.results)
        if run.batch_index not in labels and run.query:
            labels[run.batch_index] = run.query

    if not any(grouped.values()):
        return "No results found."

    if len(grouped) <= 1:
        blocks = [_block(r, max_len) for run in response.by_provider for r in run.results]
        return "\n\n".join(blocks)

    sections: list[str] = []
    for index in sorted(grouped):
        label = labels.get(index, "")
        heading = f"### Query {index + 1}: {label}" if label else f"### Query {index + 1}"
        blocks = [_block(r, max_len) for r in grouped[index]]
        body = "\n\n".join(blocks) if blocks else "No results found."
        sections.append(f"{heading}\n{body}")
    return "\n\n".join(sections)


def search_options(providers: list[str] | None) -> dict[str, object]:
    """Config-file/env options as ``search()`` kwargs, with ``--providers`` on top.

    The flag is the most specific thing the user said, so it replaces any resolved
    ``providersInclude`` — and it is handed to the resolver as that selection rather
    than pasted over the answer afterwards. That ordering is the whole point: the
    composition rules (the ``providerOverrides`` membership guard, the include/exclude
    subtraction) run ONCE, against the include this call will actually carry.

    Applying them first and substituting after was subtly wrong in both directions: an
    override the flag LICENSES had already been dropped against the file's include and
    could not be brought back, and the advice printed for it ("add it to
    providersInclude") described a config the flag had just overruled.
    """
    # The project layer is opt-in in the SDK, and a skill script is exactly the case it
    # is meant for: the agent runs it in the user's project directory.
    resolved = resolve_search_options(project_root=os.getcwd(), providers_include=providers)
    for warning in resolved.warnings:
        print(warning, file=sys.stderr)
    return resolved.search_kwargs()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("queries", nargs="+", help="one or more search queries (batched)")
    parser.add_argument("--session", default=None, help="session id from a previous search")
    parser.add_argument("--goal", default=None, help="the current task (first search only)")
    parser.add_argument(
        "--providers",
        default=None,
        help="comma-separated provider names (default: the Telem config, else server picks)",
    )
    parser.add_argument(
        "--max-len", type=int, default=8000, help="per-result content cap in characters"
    )
    args = parser.parse_args()

    providers = None
    if args.providers:
        providers = [item.strip() for item in args.providers.split(",") if item.strip()] or None
    max_len = args.max_len if args.max_len > 0 else 8000
    query = args.queries[0] if len(args.queries) == 1 else args.queries

    try:
        response = Telem().search(
            query,
            # Continuation calls: the backend already knows the session's goal.
            goal=None if args.session else args.goal,
            session=args.session,
            # Call-level, so a config file outranks the client's env-derived defaults.
            **search_options(providers),
        )
    except TelemError as exc:
        sys.exit(f"telem search failed: {exc.message}")
    except httpx.HTTPError as exc:
        sys.exit(f"telem search failed: could not reach the Telem backend ({exc})")

    print(f"Telem search session: {response.session_id}\n")
    print(format_results(response, max_len))


if __name__ == "__main__":
    main()
