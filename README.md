# Telem Python SDK

A typed Python client for the Telem search orchestration API. It wraps the backend's
interaction endpoint and reads back the server's normalized envelope — one result shape for
every provider — with both synchronous and asynchronous clients.

Full documentation, including quickstarts for every surface (SDK, MCP, OpenClaw, OpenCode,
Pi): [docs.telem.ai](https://docs.telem.ai) (launching soon).

## Installation

```bash
pip install telem-sdk
```

Requires Python 3.10+ (since 0.1.3 — 0.1.2 and earlier declare 3.12+, and pip
will not select them on 3.10 or 3.11). Working from a checkout instead? `uv sync`
— see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quickstart

```python
from telem import Telem

client = Telem  # reads TELEM_API_KEY and TELEM_BASE_URL from the environment
results = client.search("best python http client").results
for r in results:
    print(r.title, r.url)
```

## Configuration

The client is credential-agnostic and resolves configuration from arguments, then the
environment, then — for credentials only — `~/.telem/credentials.json`, then defaults:

| Setting             | Argument                       | Env var                    | Default                    |
|---------------------|--------------------------------|----------------------------|----------------------------|
| API key             | `api_key`                      | `TELEM_API_KEY`            | `~/.telem/credentials.json`, else none (anonymous) |
| Base URL            | `base_url`                     | `TELEM_BASE_URL`           | `~/.telem/credentials.json`, else `https://router.telem.ai` |
| Result tier         | `default_tier`                 | `TELEM_TIER`               | the server's (`default`)   |
| Explicit fields     | `default_fields`               | `TELEM_FIELDS` (csv)       | unset (the tier decides)   |
| Provider allow-list | `default_providers_include`    | `TELEM_PROVIDERS_INCLUDE` (csv) | the deployment's set  |
| Provider deny-list  | `default_providers_exclude`    | `TELEM_PROVIDERS_EXCLUDE` (csv) | unset                 |
| Full page content   | `default_include_full_content` | `TELEM_FULL_CONTENT` (`1` only) | off                   |

When an API key is set, requests carry an `Authorization: Bearer <key>` header. Anonymous
access works for most endpoints locally; `sessions.list` requires a token.

`~/.telem/credentials.json` is the machine-written credentials file (`{"apiKey": "tlm_…",
"baseUrl": "https://…"}`) that the guided installer writes, so an installed user never has
to export anything. It supplies each of the two values only when that value is still
unresolved, and it is opened only when at least one of them is — so a client given both
its key and its base URL by argument or env var never touches the disk. A missing or
malformed file simply supplies nothing, and nothing is printed. `TELEM_CONFIG_DIR`
relocates the directory. It is the SDK's **only** config file — repo-local
`.telem/telem.json` search options, which the Telem plugins do read, are
deliberately never read here: a library must not let a checked-out repository steer an
arbitrary program's spend.

Every search default resolves as **call argument → constructor argument → env var →
unset**. A csv env var is split on commas with items stripped and empties dropped, so an
all-empty value reads as unset: only a constructor argument can express an explicit empty
list (`default_fields=[]`, a deliberate "send nothing" the server rejects). `TELEM_TIER`
and `TELEM_FULL_CONTENT` follow the same rule — `TELEM_FULL_CONTENT` enables full content
for exactly the value `1`. `TELEM_PROVIDERS` is NOT read by the SDK: it belongs to the
the Telem plugins, where it survives as a
**deprecated alias of the provider allow-list**, ranked below `TELEM_PROVIDERS_INCLUDE`
and below both config files. Export `TELEM_PROVIDERS_INCLUDE` everywhere.

`num_results`, `include_raw` and `provider_overrides` are deliberately call-level only:
per-call intent, an audit knob and a surgical escape hatch respectively.

```python
client = Telem(api_key="tlm_...", base_url="https://router.telem.ai", default_tier="extended")
```

The request timeout defaults to **60 s** (the server grants provider timeouts that long at
the `max` tier and with full content); pass `timeout=` to change it.

## Search

`search` performs a single round trip — one Search operation — and returns the server's
**normalized envelope** for every provider that ran — every provider's rows come back in
one shape, whatever its own API looks like:

```python
resp = client.search(
    "climate policy 2026",
    tier="extended",             # minimalist | default | extended | max
    providers_include=["exa"],   # omit to use the deployment's default provider set
    num_results=10,              # rows PER PROVIDER (server default 5, range 1.20)
    include_raw=True,            # also attach each provider's own response body
    goal="brief the user",       # merged into request metadata
    context="follow-up query",   # merged into request metadata
)

resp.results        # flattened list[SearchResult]: providers in run order, rows in envelope order
resp.by_provider    # list[ProviderRun] — the primary surface; keeps partial failures
resp.session_id     # continue the conversation by passing session=resp.session_id
resp.status         # "succeeded" | "partially_succeeded" | "failed"
resp.normalized_schema_version   # the contract the server answered with
```

The full option set is `tier`, `fields`, `providers_include`, `providers_exclude`,
`provider_overrides`, `num_results`, `include_raw`, `include_full_content`, plus
`goal`/`context`/`session`/`metadata`. `None` means unset (fall through to the client
default, then to the server's own default); `[]`, `{}`, `False` and `True` are all
explicit values and are sent verbatim. Nothing is pre-validated client-side — tier names,
field names and the `num_results` bounds are the server's call and come back as
`BadRequestError`.

Two options compose rather than stack: a `fields` list **replaces** any `tier` (the level
that set it wins, and `fields` wins a tie), and when both provider halves are set the
excluded names are subtracted from the allow-list, which then fully determines the set.

`provider_overrides` is the per-provider escape hatch: raw parameters merged into ONE
provider's request body, keyed by provider name and written in **that provider's own
vocabulary**, not the SDK's:

```python
resp = client.search("climate policy 2026", provider_overrides={"exa": {"numResults": 2}})
```

An overridden provider gets its `raw` payload attached automatically, so you can see what
the override actually did.

Each `SearchResult` exposes `url`, `title`, `summary`, `excerpt`, `full_content`,
`publish_date`, `rank`, `thumbnail`, `favicon`, `source` (an object: `domain`/`name`/
`author`), `enrichments`, `fetch_meta`, plus `provider` and `raw` (the verbatim envelope
row). `result.content` survives as a **legacy alias** — `summary`, else
`full_content["content"]`, else `""` — as a property, not a stored field: it never appears
in `model_dump`.

## Providers

```python
for p in client.providers:
    print(p.name, p.active_by_default, p.normalized, p.tiers)
```

`normalized` marks the providers that return the V2 envelope (they are the ones a search
can select), and `tiers` lists the tier names each of them serves.

## Sessions

```python
client.sessions.list                      # list[SessionSummary] (requires an API key)
client.sessions.history(session_id)         # short history
client.sessions.history(session_id, full=True)  # detailed history
client.sessions.results(session_id)         # aggregated websearch preprocessor results
```

## LangChain and LangGraph integration

Install the optional integration to create a native LangChain tool:

```bash
pip install "telem-sdk[langchain]"
```

```python
from telem import Telem
from telem.integrations.langchain import create_telem_search_tool

client = Telem
telem_search = create_telem_search_tool(
    client,
    providers_include=["exa"],
    num_results=5)
```

Only `query` is exposed in the model's tool schema. Search policy — tier, providers,
result count, metadata, and the other `search` options — is fixed by application code
when the tool is created. The tool returns compact text to the model and keeps the full
typed `SearchResponse` in `ToolMessage.artifact` for application code.

### LangChain agent

Install `langchain` and the package for your model provider. The current `create_agent`
runtime uses LangGraph internally and accepts the Telem tool directly:

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

agent = create_agent(ChatOpenAI(model="gpt-5.4-mini"), tools=[telem_search])
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What changed today?"}]},
    config={"configurable": {"thread_id": "application-conversation-42"}})
```

### LangGraph ToolNode

Applications using the lower-level graph API can pass the same tool to `ToolNode`; there
is no separate LangGraph implementation:

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([telem_search])
```

Pass an `AsyncTelem` client to create an async-only tool and run the graph with `ainvoke`.
Telem's typed API errors propagate to LangGraph, whose `ToolNode` error policy can handle
them normally.

## Agent integration (OpenAI wrap)

`client.wrap` patches an OpenAI client in place (exa-style — the same object is
returned) so the model can call Telem search as a `telem_search` tool. One wrapped
client = one agent conversation = one Telem session; the wrap sends query lineage so
searches from one conversation stay grouped. Requires the extra:
`pip install telem-sdk[openai]`.

```python
from openai import OpenAI
from telem import Telem

client = Telem
wrapped = client.wrap(OpenAI)   # patched in place; same object returned

r = wrapped.chat.completions.create(model="gpt-4o", messages=msgs)
r.telem_responses                    # list[SearchResponse]; empty if no search ran
wrapped.telem_conversation_id        # this agent's conversation identity
wrapped.telem_session_id             # the backend session, adopted from the first search
wrapped.telem_messages               # recorded conversation snapshot
```

By default the request's `tools` are **replaced** with the `telem_search` tool (caller
tools are dropped, exa parity) and search rounds complete inside `create`. Agents
with their own tools switch to loop integration:

```python
run_tool = wrapped.wrap_tool_runner(run_my_tool)  # merges tools; create stops auto-completing

r = wrapped.chat.completions.create(model="gpt-4o", messages=msgs, tools=my_tools)
if r.choices[0].message.tool_calls:
    msgs.append(r.choices[0].message)  # the assistant message that made the tool calls
    for tc in r.choices[0].message.tool_calls:
        msgs.append(run_tool(tc))  # telem_search handled by the SDK, others by run_my_tool
```

Pass `use_telem="none"` on a call to skip Telem for that call. `AsyncTelem.wrap`
mirrors this for `AsyncOpenAI`; the async runner accepts sync or async tool functions.

Wrap the subagent **at the moment you delegate**, passing the parent wrapped client.
The wrap freezes the parent's conversation right then and carries it as the child's
newest ancestor:

```python
root = client.wrap(OpenAI, goal="answer the user task")
root.chat.completions.create(model="gpt-4o", messages=root_msgs)

# Inside the parent's tool handler, when it decides to spawn a researcher:
subagent = client.wrap(OpenAI, parent=root, conversation_id="research-1")
subagent.chat.completions.create(model="gpt-4o", messages=subagent_msgs)
```

Every search the subagent runs then carries its lineage back to the parent, so parent
and child stitch into one graph. Nested subagents work the same way by passing the
spawning subagent as `parent`.

> **Wrap the child when you delegate, not at startup.** The freeze happens at `wrap`
> time. A child wrapped before its parent has spoken records a delegation with empty
> parent context, and the wrap cannot detect that.

**Conversation identity.** `conversation_id` is auto-minted per wrapped client. Supply
your own whenever the conversation outlives one client object — a web server wrapping a
fresh client per request must pass a stable thread id, or every request looks like a new
conversation. `context_window_id` is the matching override for the context-window
generation; by default the wrap anchors on the first message, so trimming or summarizing
the history starts a new generation on its own.

Streaming is not supported yet: `stream=True` calls emit a `UserWarning` and bypass
Telem entirely (no `telem_search` tool, no session tracking, no reply recording).

## Agent Skill

`telem-search` is an installable [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills)
that teaches an agent to search the web by calling this SDK directly — single and
batched searches, session continuation, provider selection — with no MCP server
involved. It ships inside the package (`telem/_skills/telem-search/`); Claude Code
only loads skills from `~/.claude/skills/` or a project's `.claude/skills/`, so a
console script copies it there:

```bash
telem-install-skill              # ~/.claude/skills/telem-search — all projects
telem-install-skill --project    # ./.claude/skills/telem-search — this project only
```

Pass `--force` to replace an existing installation; without it the command refuses
to overwrite one and exits non-zero. The console script comes from the package, so
install it first: `pip install telem-sdk`, or `pip install -e .` from a checkout.

It ships a self-contained one-shot CLI at `scripts/search.py` inside the installed
skill directory.

The skill reads the same unified config as the Telem plugins, through
the SDK's one public accessor — `telem.resolve_search_options(project_root=os.getcwd)`
→ `options.search_kwargs`, splatted into `client.search(...)`. `--providers` goes in
as that call's `providers_include`, so the composition rules run once against the
selection the request actually carries. The project level is
opt-in (that `project_root` argument is the opt-in); the client itself still reads no
repo-local file. Resolution is per invocation, notices go to stderr, and a malformed
file is ignored rather than fatal.

## Async

`AsyncTelem` mirrors `Telem`; every request method is a coroutine:

```python
import asyncio
from telem import AsyncTelem

async def main:
    async with AsyncTelem as client:
        resp = await client.search("best python http client")
        print(len(resp.results))

asyncio.run(main)
```

## Errors

All errors derive from `TelemError` and carry `.message`, `.status_code`, and `.body`:

| Status         | Exception          |
|----------------|--------------------|
| 400            | `BadRequestError`  |
| 401 / 403      | `AuthError`        |
| 404            | `NotFoundError`    |
| other non-2xx  | `APIStatusError`   |

Two failures never reach a status code: a request that produced no HTTP response at all
raises `TelemConnectionError`, and a search answered by a pre-V2 server raises
`TelemServerVersionError`.

```python
from telem import Telem, BadRequestError

try:
    Telem.search("hi", providers_include=["does-not-exist"])
except BadRequestError as exc:
    print(exc.status_code, exc.message)
```

## License

Copyright (c) 2026 Telem AI. Licensed under the [Apache License, Version 2.0](LICENSE).
