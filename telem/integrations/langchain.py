"""LangChain tool integration for Telem search."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
    from langchain_core.tools import BaseTool, StructuredTool
    from langgraph.prebuilt import ToolRuntime
except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
    raise ImportError(
        "The LangChain integration requires optional dependencies; "
        'install them with `pip install "telem-sdk[langchain]"`.'
    ) from exc

from telem.async_client import AsyncTelem
from telem.client import Telem
from telem.integrations._langchain_trajectory import create_child_config, trajectory_metadata
from telem.integrations._tools import DEFAULT_RESULT_MAX_LEN, format_search_results
from telem.models import SearchResponse

_DEFAULT_DESCRIPTION = "Search the web for relevant, up-to-date information."


class TelemSearchInput(BaseModel):
    """Validated search input, including LangGraph's hidden injected runtime."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    query: str = Field(description="The query to search for.")
    runtime: ToolRuntime = Field(default=None)  # type: ignore[assignment]


def create_telem_child_config(
    parent_runtime: ToolRuntime,
    *,
    child_thread_id: str,
    child_context_window_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return LangGraph config that links a child agent to its parent snapshot.

    Call this from the parent tool that starts the child graph. The helper freezes the
    parent's active message branch and carries any existing ancestors into the child.
    """
    return create_child_config(
        parent_runtime,
        child_thread_id=child_thread_id,
        child_context_window_id=child_context_window_id,
        config=config,
    )


def create_telem_search_tool(
    client: Telem | AsyncTelem,
    *,
    name: str = "telem_search",
    description: str = _DEFAULT_DESCRIPTION,
    result_max_len: int = DEFAULT_RESULT_MAX_LEN,
    tier: str | None = None,
    fields: list[str] | None = None,
    providers_include: list[str] | None = None,
    providers_exclude: list[str] | None = None,
    provider_overrides: dict[str, dict[str, Any]] | None = None,
    num_results: int | None = None,
    include_raw: bool | None = None,
    include_full_content: bool | None = None,
    goal: str | None = None,
    context: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    trace_metadata: dict[str, Any] | None = None,
) -> BaseTool:
    """Create a LangChain tool backed by a Telem client.

    Only ``query`` is exposed to the model. All other arguments are application-owned
    policy forwarded to every :meth:`Telem.search` call. The tool's model-facing content
    is a compact text rendering; its LangChain artifact is the typed
    :class:`SearchResponse`.

    When LangGraph injects ``ToolRuntime``, the active message branch and trajectory-v5
    identity are sent with every search. A stable ``configurable.thread_id`` keeps one
    conversation fingerprint across invocations. Direct tool invocation has no graph state
    and therefore keeps the history-free request behavior.

    ``tags`` and ``trace_metadata`` annotate the LangChain tool run. They are sent to
    configured callback handlers, not to Telem. Use ``metadata`` for additional metadata;
    generated trajectory-v5 keys take precedence over colliding caller keys.

    A tool built from :class:`Telem` supports ``invoke`` (and ``ainvoke`` through
    LangChain's executor fallback). A tool built from :class:`AsyncTelem` supports
    ``ainvoke`` and intentionally has no synchronous implementation.
    """
    search_kwargs = {
        "tier": tier,
        "fields": fields,
        "providers_include": providers_include,
        "providers_exclude": providers_exclude,
        "provider_overrides": provider_overrides,
        "num_results": num_results,
        "include_raw": include_raw,
        "include_full_content": include_full_content,
        "goal": goal,
        "context": context,
        "metadata": metadata,
    }

    def runtime_search_kwargs(runtime: ToolRuntime | None) -> dict[str, Any]:
        if runtime is None:
            return search_kwargs
        return {
            **search_kwargs,
            "metadata": trajectory_metadata(metadata, runtime),
        }

    if isinstance(client, AsyncTelem):

        async def async_search(
            query: str,
            runtime: ToolRuntime = None,  # type: ignore[assignment]
        ) -> tuple[str, SearchResponse]:
            response = await client.search(query, **runtime_search_kwargs(runtime))
            return format_search_results(response, result_max_len), response

        return StructuredTool.from_function(
            coroutine=async_search,
            name=name,
            description=description,
            args_schema=TelemSearchInput,
            response_format="content_and_artifact",
            tags=tags,
            metadata=trace_metadata,
        )

    if isinstance(client, Telem):

        def search(
            query: str,
            runtime: ToolRuntime = None,  # type: ignore[assignment]
        ) -> tuple[str, SearchResponse]:
            response = client.search(query, **runtime_search_kwargs(runtime))
            return format_search_results(response, result_max_len), response

        return StructuredTool.from_function(
            func=search,
            name=name,
            description=description,
            args_schema=TelemSearchInput,
            response_format="content_and_artifact",
            tags=tags,
            metadata=trace_metadata,
        )

    raise TypeError("client must be an instance of Telem or AsyncTelem")


__all__ = [
    "TelemSearchInput",
    "create_telem_child_config",
    "create_telem_search_tool",
]
