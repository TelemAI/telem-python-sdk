---
name: telem-search
description: "Searches the web through the Telem search orchestration API using the telem-sdk Python package: single searches, batched multi-query searches (several queries run concurrently server-side as one interaction), session continuation across follow-up searches, and provider selection and discovery. Use when the user asks to search the web, research a topic, or gather sources, and TELEM_API_KEY is set (TELEM_BASE_URL is optional and defaults to the hosted router)."
---

# Telem Search (Python SDK)

Search the web through a Telem backend by calling the `telem-sdk` Python package
directly — no MCP server needed. `AsyncTelem` mirrors `Telem` with the exact same
API awaited, for use inside async code.

## Setup

`telem-sdk` is on PyPI, so either of these works:

```bash
pip install telem-sdk
# or, without installing:
uv run --with telem-sdk python your_script.py
```

Working from a checkout of the repository instead:

```bash
cd /path/to/telem-python-sdk && pip install -e.
uv run --with /path/to/telem-python-sdk python your_script.py
```

If you are reading this file from an installed skill directory, `telem-sdk` is
already installed somewhere — run scripts with that same interpreter.

The client reads its credentials from the environment at construction:

| Env var          | Meaning                                                                            |
|------------------|------------------------------------------------------------------------------------|
| `TELEM_BASE_URL` | API base URL. **Defaults to the hosted `https://router.telem.ai`** — searches leave this machine and the `TELEM_API_KEY` bearer token goes with them. Set this only if Telem gave you a different endpoint. |
| `TELEM_API_KEY`  | Bearer token; required by current backends for searches and all session endpoints  |

Both also fall back to `~/.telem/credentials.json`, which the Telem installer writes,
so a logged-in user need not export anything.

## Search options: the unified Telem config

Search options live in **one config every Telem harness reads** — not in a per-tool
setting. Resolve them with `telem.resolve_search_options` and splat them into the call:

```python
import os, sys
from telem import Telem, resolve_search_options

options = resolve_search_options(project_root=os.getcwd)  # project layer is opt-in
for warning in options.warnings:                          # never into your answer
    print(warning, file=sys.stderr)

resp = Telem.search("query", goal="...", **options.search_kwargs)
```

Three levels, **per key**, top wins — a project file that sets `tier` does not hide the
user file's `fields`:

| Level | Where | Notes |
|---|---|---|
| project | `<project>/.telem/telem.json` | Only when you pass `project_root`; usually the user's cwd, and often committed to the repo |
| user | `~/.telem/telem.json` | `TELEM_CONFIG_DIR` relocates the directory |
| env | `TELEM_TIER`, `TELEM_FIELDS`, `TELEM_PROVIDERS_INCLUDE`, `TELEM_PROVIDERS_EXCLUDE`, `TELEM_FULL_CONTENT` | csv for the list ones; `1` only for the flag |

File keys are `tier`, `fields`, `providersInclude`, `providersExclude`, `fullContent`,
`providerOverrides`. An empty value (`[]`, `{}`, `""`) reads exactly like an omitted
key, and an unknown key is ignored rather than rejected.

Reading never raises: a missing, malformed or hostile file is simply absent and the
next level supplies the key, with a line in `options.warnings` saying what was ignored.

Three composition rules `resolve_search_options` applies for you, the same ones the
opencode and pi plugins apply, in this order:

1. A request may carry `tier` **or** `fields`, never both — the more specific level
   wins; on a tie, `fields`.
2. When **both** `providersInclude` and `providersExclude` resolve, the exclusion is
   subtracted from the include and only `providersInclude` is sent: the server rejects
   a request naming the same provider in both halves, and an include already replaces
   the deployment's default set. If the subtraction empties the include, both halves
   are dropped and the deployment's default providers run — a config can never resolve
   to an empty provider set, which would fail every search. `providersExclude` on its
   own is forwarded verbatim.
3. `providerOverrides` applies only alongside the resolved `providersInclude` and only
   to providers named there — anything else is dropped rather than sent, because the
   server rejects an override naming a provider the request did not select.

Each rule that drops something says so in `options.warnings`.

The **client itself never reads these files** — a library must not let a checked-out
repository steer an arbitrary program's spend. It does read
`TELEM_PROVIDERS_INCLUDE`/`TELEM_PROVIDERS_EXCLUDE`/`TELEM_TIER`/`TELEM_FIELDS`/`TELEM_FULL_CONTENT`
as constructor defaults, which is why the resolved options go in as **call** arguments:
call beats default, which is exactly file-beats-env. (`TELEM_PROVIDERS` is a deprecated
alias of `TELEM_PROVIDERS_INCLUDE`, honored by `resolve_search_options` and by no other
part of the SDK.)

## Single search

```python
from telem import Telem

client = Telem  # reads TELEM_BASE_URL / TELEM_API_KEY from the environment
resp = client.search(
    "postgres jsonb indexing strategies",
    goal="Help the user pick an indexing strategy for a jsonb-heavy Postgres schema.")
for r in resp.results:
    print(r.url)
    print(r.title)
    print(r.content)
```

`resp.session_id` identifies the session for follow-up searches; `resp.status` is
`succeeded`, `partially_succeeded`, or `failed`.

## Batched search

Passing a list of queries batches them into ONE interaction — the backend runs
them concurrently. Group the output by query via `run.batch_index` / `run.query`
over `resp.by_provider`:

```python
resp = client.search(["fastapi background tasks", "celery vs arq"], session=session_id)

for run in resp.by_provider:
    print(f"Query {run.batch_index + 1}: {run.query} [{run.provider}]")
    for r in run.results:
        print(" -", r.title, r.url)
```

The flat `resp.results` stays the concatenation across all runs. A one-element
list behaves exactly like a plain string.

## Session continuation

One Telem session represents one **task**, not one query topic.
Sessions group a task's searches server-side.

- Set `goal` only on the FIRST search of a task: a 3-4 sentence paragraph
  describing that task. The backend stores it with the session.
- Pass `session=resp.session_id` on every follow-up search for the same task —
  even when the queries move to a different aspect or subtopic of that task.
  Never pass `goal` again once you have a session id; it is already known.
- Start a new session (fresh `goal`, no `session`) only for a genuinely new task.
- If you no longer have the session id — after a context compaction, for example —
  do not invent one: start a fresh search with no `session`, restate the task in
  `goal`, and continue from there.
- When starting a new goal, make exactly ONE first search and wait for its
  `session_id` before issuing more — parallel first calls (e.g. `AsyncTelem`
  under `asyncio.gather`) each open a separate session. Batch multiple initial
  queries into one call instead; once you have the id, parallel follow-ups with
  `session=` are fine.

```python
first = client.search("rust async runtimes", goal="Survey async Rust for a blog post ...")
followup = client.search("tokio vs async-std benchmarks", session=first.session_id)
```

## Provider selection and discovery

List the configured providers (no auth needed), then request specific ones:

```python
for p in client.providers:
    print(p.name, p.active_by_default, p.description)

resp = client.search("query", providers_include=["exa"], session=session_id)
```

Omit `providers_include` to use the deployment's defaults, or let the Telem config
supply it (see "Search options" above). An explicit `providers_include=` on the call
outranks the config. An unknown provider name raises `BadRequestError`.

## Result fields

- `resp.results` — flat `list[SearchResult]` across all providers, in provider
  order; each has `url`, `title`, `summary`, `excerpt`, `full_content`,
  `publish_date`, `rank`, `source`, `provider`, and `raw` (the verbatim server
  envelope row). `r.content` is a legacy read-only alias (`summary`, else
  `full_content["content"]`, else `""`).
- `resp.by_provider` — per-provider `ProviderRun` breakdown with `status`,
  `error`, `latency_ms`, `batch_index`, and `query`. Partial failures are
  preserved: a failed provider appears with its `error` while others carry results.
- `resp.status` — `succeeded` / `partially_succeeded` / `failed`.

## Errors

All SDK errors subclass `TelemError` and carry `.message`, `.status_code`, `.body`:

| Status        | Exception         | Typical cause                              |
|---------------|-------------------|--------------------------------------------|
| 400           | `BadRequestError` | Unknown provider name, malformed input     |
| 401 / 403     | `AuthError`       | Missing/invalid token — check `TELEM_API_KEY` |
| 404           | `NotFoundError`   | Unknown session or endpoint                |
| other non-2xx | `APIStatusError`  | Backend/provider failure (5xx)             |

## Helper script

`scripts/search.py` (in this skill directory) is a self-contained one-shot CLI.
Run it from the skill directory, with the interpreter that has `telem-sdk`
installed:

```bash
python scripts/search.py "query" --goal "..."
# from a checkout, without installing:
uv run --with /path/to/telem-python-sdk python scripts/search.py "query" --goal "..."
```

It supports multiple queries (batched), `--session`, `--providers exa,dummy`, and
`--max-len`, and prints the session id first so follow-ups can continue it.

It reads the same unified config on every invocation, taking the project file from the
directory you run it in — so run it from the project root — and `--providers` overrides
the config's provider selection for that one run. Config warnings go to stderr, so
piping stdout keeps the results clean.

## Installing this skill

This skill ships inside the `telem-sdk` package. Installing the package and
running its console script copies it into a location Claude Code scans:

```bash
pip install telem-sdk            # or `pip install -e .` from a checkout
telem-install-skill              # ~/.claude/skills/telem-search — personal, all projects
telem-install-skill --project    # ./.claude/skills/telem-search — this project only
python -m telem.install_skill    # same thing, when the console script isn't on PATH
```

Add `--force` to replace an existing installation; without it the command
refuses to overwrite one and exits `1` (any other failure exits `3`). Run
`--project` from the project root — Claude Code only reads the `.claude` of the
directory it was started in.
