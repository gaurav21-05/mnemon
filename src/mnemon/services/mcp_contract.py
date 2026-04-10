"""MCP naming and resource-contract helpers for Mnemon service adapters."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mnemon.services.memory_service import MemoryService

RAW_TOOL_NAMES: tuple[str, ...] = (
    "memory_write",
    "memory_retrieve",
    "memory_profile_recall",
    "memory_explain_fact",
    "memory_causal_trace",
    "memory_consolidate",
    "memory_state",
)


def qualify_tool_name(namespace: str, raw_name: str) -> str:
    """Build a stable qualified MCP tool name."""
    clean_ns = namespace.strip().replace(" ", "_") or "mnemon"
    return f"{clean_ns}.{raw_name}"


def state_resource_uri(namespace: str) -> str:
    clean_ns = namespace.strip().replace(" ", "_") or "mnemon"
    return f"memory://{clean_ns}/state"


def episodes_resource_uri(namespace: str) -> str:
    clean_ns = namespace.strip().replace(" ", "_") or "mnemon"
    return f"memory://{clean_ns}/episodes/recent"


def facts_resource_uri(namespace: str) -> str:
    clean_ns = namespace.strip().replace(" ", "_") or "mnemon"
    return f"memory://{clean_ns}/facts/recent"


def profile_resource_uri(namespace: str) -> str:
    clean_ns = namespace.strip().replace(" ", "_") or "mnemon"
    return f"memory://{clean_ns}/profile/current"


def known_resource_uris(namespace: str) -> list[str]:
    return [
        state_resource_uri(namespace),
        episodes_resource_uri(namespace),
        facts_resource_uri(namespace),
        profile_resource_uri(namespace),
    ]


async def read_resource(service: MemoryService, namespace: str, uri: str) -> dict[str, Any]:
    """Read one synthetic MCP resource backed by current memory state."""
    if uri == state_resource_uri(namespace):
        payload = await service.state()
        return {
            "uri": uri,
            "mime_type": "application/json",
            "text": json.dumps(payload, indent=2),
        }

    if uri == episodes_resource_uri(namespace):
        docs = await service.episodic._document_store.query(filters={}, limit=10)
        payload = {
            "count": len(docs),
            "episodes": docs,
        }
        return {
            "uri": uri,
            "mime_type": "application/json",
            "text": json.dumps(payload, indent=2, default=str),
        }

    if uri == facts_resource_uri(namespace):
        payload = await service.recent_facts(limit=10)
        return {
            "uri": uri,
            "mime_type": "application/json",
            "text": json.dumps(payload, indent=2),
        }

    if uri == profile_resource_uri(namespace):
        payload = await service.profile_snapshot()
        return {
            "uri": uri,
            "mime_type": "application/json",
            "text": json.dumps(payload, indent=2),
        }

    raise ValueError(f"Unknown resource URI: {uri}")
