#!/usr/bin/env python3
"""MCP server exposing Mnemon memory tools for third-party agents.

Environment variables:
  MNEMON_MCP_MODEL          Optional LLM model for consolidation
  MNEMON_MCP_EMBED_MODEL    Embedding model (default: text-embedding-3-small)
  MNEMON_MCP_EMBED_DIM      Embedding dimensions (default: 1536)
"""

from __future__ import annotations

import os
from typing import Any

from mnemon.services import (
    MemoryService,
    episodes_resource_uri,
    facts_resource_uri,
    known_resource_uris,
    profile_resource_uri,
    qualify_tool_name,
    read_resource,
    state_resource_uri,
)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing MCP dependency. Install with: pip install 'mnemon[mcp]'") from exc


mcp = FastMCP("mnemon-memory")
_service: MemoryService | None = None
_namespace = os.getenv("MNEMON_MCP_NAMESPACE", "mnemon")


async def _get_service() -> MemoryService:
    global _service
    if _service is None:
        model = os.getenv("MNEMON_MCP_MODEL")
        embedding_model = os.getenv("MNEMON_MCP_EMBED_MODEL", "text-embedding-3-small")
        embedding_dim = int(os.getenv("MNEMON_MCP_EMBED_DIM", "1536"))
        _service = await MemoryService.create_default(
            model=model,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )
    return _service


@mcp.tool(name=qualify_tool_name(_namespace, "memory_write"))
async def memory_write(
    content: str,
    agent_id: str = "mcp-agent",
    session_id: str | None = None,
    tags: list[str] | None = None,
    importance: float | None = None,
) -> dict[str, Any]:
    """Store a memory episode for later recall."""
    service = await _get_service()
    return await service.write_memory(
        content=content,
        agent_id=agent_id,
        session_id=session_id,
        tags=tags,
        importance=importance,
    )


@mcp.tool(name=qualify_tool_name(_namespace, "memory_retrieve"))
async def memory_retrieve(
    query: str,
    top_k: int = 5,
    min_score: float = 0.01,
) -> dict[str, Any]:
    """Retrieve episodic and semantic memory relevant to a query."""
    service = await _get_service()
    return await service.retrieve_memory(query=query, top_k=top_k, min_score=min_score)


@mcp.tool(name=qualify_tool_name(_namespace, "memory_profile_recall"))
async def memory_profile_recall(
    query: str,
    top_k: int = 5,
    scope_type: str = "all",
    scope_id: str | None = None,
) -> dict[str, Any]:
    """Retrieve scoped memories alongside current and historical profile facts."""
    service = await _get_service()
    return await service.profile_recall(
        query=query,
        top_k=top_k,
        scope_type=scope_type,
        scope_id=scope_id,
    )


@mcp.tool(name=qualify_tool_name(_namespace, "memory_explain_fact"))
async def memory_explain_fact(triple_id: str) -> dict[str, Any]:
    """Explain one semantic fact by tracing it back to source episodes."""
    service = await _get_service()
    return await service.explain_fact(triple_id)


@mcp.tool(name=qualify_tool_name(_namespace, "memory_causal_trace"))
async def memory_causal_trace(
    episode_id: str | None = None,
    outcome_query: str | None = None,
    max_depth: int = 10,
) -> dict[str, Any]:
    """Trace the causal episode chain behind a memory or outcome query."""
    service = await _get_service()
    return await service.causal_trace(
        episode_id=episode_id,
        outcome_query=outcome_query,
        max_depth=max_depth,
    )


@mcp.tool(name=qualify_tool_name(_namespace, "memory_consolidate"))
async def memory_consolidate() -> dict[str, Any]:
    """Run one consolidation cycle (episodes to semantic facts)."""
    service = await _get_service()
    return await service.consolidate()


@mcp.tool(name=qualify_tool_name(_namespace, "memory_state"))
async def memory_state() -> dict[str, Any]:
    """Get aggregate stats for memory stores."""
    service = await _get_service()
    return await service.state()


@mcp.tool(name=qualify_tool_name(_namespace, "memory_resources_list"))
async def memory_resources_list() -> dict[str, Any]:
    """List available memory resources (tool fallback when MCP resources are unavailable)."""
    return {
        "resources": known_resource_uris(_namespace),
    }


@mcp.tool(name=qualify_tool_name(_namespace, "memory_resources_read"))
async def memory_resources_read(uri: str) -> dict[str, Any]:
    """Read a memory resource by URI (tool fallback)."""
    service = await _get_service()
    try:
        payload = await read_resource(service, _namespace, uri)
    except ValueError as exc:
        return {
            "error": str(exc),
            "known_resources": known_resource_uris(_namespace),
        }
    return payload


def _register_resources_if_supported() -> None:
    """Register MCP resources when the server implementation supports them."""
    resource_decorator = getattr(mcp, "resource", None)
    if resource_decorator is None:
        return

    @resource_decorator(state_resource_uri(_namespace))
    async def _resource_state() -> str:
        service = await _get_service()
        payload = await read_resource(service, _namespace, state_resource_uri(_namespace))
        return str(payload["text"])

    @resource_decorator(episodes_resource_uri(_namespace))
    async def _resource_recent_episodes() -> str:
        service = await _get_service()
        payload = await read_resource(service, _namespace, episodes_resource_uri(_namespace))
        return str(payload["text"])

    @resource_decorator(facts_resource_uri(_namespace))
    async def _resource_recent_facts() -> str:
        service = await _get_service()
        payload = await read_resource(service, _namespace, facts_resource_uri(_namespace))
        return str(payload["text"])

    @resource_decorator(profile_resource_uri(_namespace))
    async def _resource_profile() -> str:
        service = await _get_service()
        payload = await read_resource(service, _namespace, profile_resource_uri(_namespace))
        return str(payload["text"])


_register_resources_if_supported()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
